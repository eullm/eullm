// moe_cache_sim — the expert choices of a MoE model replayed through the
// eviction policies of llama.cpp's expert cache (`--moe-cache`,
// src/llama-moe-cache.cpp), to choose a policy before writing it into the
// cache (docs/lumi/allocation-plan.md, c10).
//
// The cache keeps the most used experts of the layers whose experts live in
// host RAM in a fixed number of VRAM slots shared by all those layers. At
// every layer of every step the experts the router chose must all have a
// slot; a miss is copied over the CPU-GPU link into a slot taken from
// another expert. Upstream takes the least recently used. The layers come in
// the same order every step, so when one step wants more experts than there
// are slots, the least recently used expert is always one the next layers
// want: on LUMI with 32 requests the cache found 0.85% of the experts it was
// asked for (Qwen3-235B) and copied every other one.
//
//   moe_cache_sim run TRACE --slots N [--expert-mib M] [--gbps G]
//   moe_cache_sim synth [model and workload options] --slots N
//   moe_cache_sim synth ... --out TRACE      (write the trace, run nothing)
//
// A trace is what `LLAMA_MOE_CACHE_TRACE=FILE` writes (patch 0008): a header,
// then one record per layer the cache planned. `synth` makes one up from a
// routing model with the knobs below, to be calibrated against the hit rates
// LUMI measured, until real traces are there.
//
// Build: c++ -O2 -std=c++17 -o moe_cache_sim moe_cache_sim.cpp

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

// ---------------------------------------------------------------- traces

constexpr uint32_t TRACE_MAGIC   = 0x43454f4d; // "MOEC"
constexpr uint32_t TRACE_VERSION = 1;

struct trace {
    int32_t n_layer  = 0; // layers of the model
    int32_t n_expert = 0;
    int32_t n_used   = 0; // experts per token
    int32_t n_slots  = 0; // the cache's own slot count, 0 if unknown
    std::vector<int32_t> layers;  // the cached layers, in order
    std::vector<int32_t> il;      // [plan] layer
    std::vector<size_t>  offset;  // [plan + 1] into ids
    std::vector<int32_t> ids;     // the experts chosen, every token's in turn

    size_t n_plans() const { return il.size(); }
    void add(int32_t l, const int32_t * p, size_t n) {
        if (offset.empty()) {
            offset.push_back(0);
        }
        il.push_back(l);
        ids.insert(ids.end(), p, p + n);
        offset.push_back(ids.size());
    }
};

void write_trace(const trace & t, const std::string & path) {
    FILE * f = fopen(path.c_str(), "wb");
    if (!f) {
        throw std::runtime_error("cannot write " + path);
    }
    const int32_t n_layers = (int32_t) t.layers.size();
    const uint32_t head[2] = { TRACE_MAGIC, TRACE_VERSION };
    fwrite(head, sizeof(head), 1, f);
    const int32_t hp[5] = { t.n_layer, t.n_expert, t.n_used, t.n_slots, n_layers };
    fwrite(hp, sizeof(hp), 1, f);
    fwrite(t.layers.data(), sizeof(int32_t), t.layers.size(), f);
    for (size_t i = 0; i < t.n_plans(); ++i) {
        const int32_t rec[2] = { t.il[i], (int32_t) (t.offset[i + 1] - t.offset[i]) };
        fwrite(rec, sizeof(rec), 1, f);
        fwrite(t.ids.data() + t.offset[i], sizeof(int32_t), rec[1], f);
    }
    fclose(f);
}

trace read_trace(const std::string & path) {
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) {
        throw std::runtime_error("cannot read " + path);
    }
    trace t;
    uint32_t head[2];
    int32_t hp[5];
    if (fread(head, sizeof(head), 1, f) != 1 || head[0] != TRACE_MAGIC || head[1] != TRACE_VERSION ||
        fread(hp, sizeof(hp), 1, f) != 1) {
        fclose(f);
        throw std::runtime_error(path + " is not an expert cache trace of version 1");
    }
    t.n_layer = hp[0]; t.n_expert = hp[1]; t.n_used = hp[2]; t.n_slots = hp[3];
    t.layers.resize(hp[4]);
    if (fread(t.layers.data(), sizeof(int32_t), t.layers.size(), f) != t.layers.size()) {
        fclose(f);
        throw std::runtime_error(path + ": truncated header");
    }
    std::vector<int32_t> buf;
    int32_t rec[2];
    while (fread(rec, sizeof(rec), 1, f) == 1) {
        buf.resize(rec[1]);
        if (fread(buf.data(), sizeof(int32_t), buf.size(), f) != buf.size()) {
            break; // a trace cut by the end of the job: keep what is whole
        }
        t.add(rec[0], buf.data(), buf.size());
    }
    fclose(f);
    return t;
}

// ---------------------------------------------------------------- policies

// What all policies share: the slot of every (layer, expert), the key of
// every slot, and the distinct experts of one plan (as plan() in llama.cpp).
struct policy {
    int32_t n_layer, n_expert, n_slots;
    std::vector<int32_t> slot_of; // [layer*n_expert + expert], -1 if not cached
    std::vector<int32_t> key_of;  // [slot], -1 if empty
    std::vector<uint32_t> seen;
    uint32_t seen_gen = 0;
    std::vector<int32_t> uniq;

    policy(int32_t n_layer, int32_t n_expert, int32_t n_slots) :
        n_layer(n_layer), n_expert(n_expert), n_slots(n_slots),
        slot_of((size_t) n_layer*n_expert, -1), key_of(n_slots, -1), seen(n_expert, 0) {}
    virtual ~policy() = default;
    virtual const char * name() const = 0;

    void distinct(const int32_t * ids, size_t n) {
        if (++seen_gen == 0) {
            std::fill(seen.begin(), seen.end(), 0);
            seen_gen = 1;
        }
        uniq.clear();
        for (size_t i = 0; i < n; ++i) {
            if (seen[ids[i]] != seen_gen) {
                seen[ids[i]] = seen_gen;
                uniq.push_back(ids[i]);
            }
        }
        if (uniq.size() > (size_t) n_slots) {
            throw std::runtime_error("a layer chose more experts than the cache has slots");
        }
    }

    // give the slot s to expert e of layer il
    void assign(int32_t s, int32_t il, int32_t e) {
        if (key_of[s] >= 0) {
            slot_of[key_of[s]] = -1;
        }
        key_of[s] = il*n_expert + e;
        slot_of[key_of[s]] = s;
    }

    // plan index i (for the policies that look ahead), the layer and its choices
    virtual void plan(size_t i, int32_t il, const int32_t * ids, size_t n, size_t & hits, size_t & misses) = 0;
};

// Upstream: one list of the slots, least recently used at the head.
struct lru_list {
    std::vector<int32_t> prev, next;
    int32_t head = -1, tail = -1;

    explicit lru_list(int32_t n) : prev(n), next(n) {
        for (int32_t s = 0; s < n; ++s) {
            prev[s] = s - 1;
            next[s] = s + 1 < n ? s + 1 : -1;
        }
        head = 0;
        tail = n - 1;
    }
    void touch(int32_t s) {
        if (s == tail) {
            return;
        }
        if (prev[s] >= 0) {
            next[prev[s]] = next[s];
        } else {
            head = next[s];
        }
        prev[next[s]] = prev[s];
        prev[s] = tail;
        next[s] = -1;
        next[tail] = s;
        tail = s;
    }
};

// llama.cpp's moe_cache_lru::plan, unchanged.
struct policy_lru : policy {
    lru_list lru;
    policy_lru(int32_t l, int32_t e, int32_t s) : policy(l, e, s), lru(s) {}
    const char * name() const override { return "lru"; }

    void plan(size_t, int32_t il, const int32_t * ids, size_t n, size_t & hits, size_t & misses) override {
        distinct(ids, n);
        for (int32_t e : uniq) {
            const int32_t s = slot_of[(size_t) il*n_expert + e];
            if (s >= 0) {
                lru.touch(s);
                hits++;
            }
        }
        std::sort(uniq.begin(), uniq.end());
        for (int32_t e : uniq) {
            if (slot_of[(size_t) il*n_expert + e] >= 0) {
                continue;
            }
            const int32_t s = lru.head;
            assign(s, il, e);
            lru.touch(s);
            misses++;
        }
    }
};

// llama.cpp's moe_cache_slru::plan since #30637 (b11777 and later), unchanged:
// a probation and a protected LRU list, a hit moves its slot to the protected
// one, which holds at most half the slots (its least recently used slot goes
// back to probation); a miss takes the least recently used slot of probation,
// then of the protected list, skipping the experts this layer chose, and the
// new expert starts in probation.
struct policy_slru : policy {
    std::vector<int32_t> prev, next;
    int32_t head[2] = { -1, -1 };
    int32_t tail[2] = { -1, -1 };
    std::vector<bool> is_protected;
    int32_t n_protected = 0;

    policy_slru(int32_t l, int32_t e, int32_t s) : policy(l, e, s), prev(s), next(s), is_protected(s, false) {
        for (int32_t i = 0; i < s; ++i) {
            prev[i] = i - 1;
            next[i] = i + 1 < s ? i + 1 : -1;
        }
        head[0] = 0;
        tail[0] = s - 1;
    }
    const char * name() const override { return "slru"; }

    void move(int32_t s, bool protect) {
        const bool from = is_protected[s];
        if (from == protect && s == tail[from]) {
            return;
        }
        if (prev[s] >= 0) {
            next[prev[s]] = next[s];
        } else {
            head[from] = next[s];
        }
        if (next[s] >= 0) {
            prev[next[s]] = prev[s];
        } else {
            tail[from] = prev[s];
        }
        prev[s] = tail[protect];
        next[s] = -1;
        if (tail[protect] >= 0) {
            next[tail[protect]] = s;
        } else {
            head[protect] = s;
        }
        tail[protect] = s;
        is_protected[s] = protect;
        n_protected += int(protect) - int(from);
    }
    void touch(int32_t s) {
        move(s, true);
        if (n_protected > n_slots/2) {
            move(head[1], false);
        }
    }
    int32_t victim(int32_t il) const {
        for (int segment = 0; segment < 2; ++segment) {
            for (int32_t s = head[segment]; s >= 0; s = next[s]) {
                const int32_t key = key_of[s];
                if (key < 0 || key / n_expert != il || seen[key % n_expert] != seen_gen) {
                    return s;
                }
            }
        }
        throw std::runtime_error("no SLRU victim");
    }

    void plan(size_t, int32_t il, const int32_t * ids, size_t n, size_t & hits, size_t & misses) override {
        distinct(ids, n);
        for (int32_t e : uniq) {
            const int32_t s = slot_of[(size_t) il*n_expert + e];
            if (s >= 0) {
                touch(s);
                hits++;
            }
        }
        std::sort(uniq.begin(), uniq.end());
        for (int32_t e : uniq) {
            if (slot_of[(size_t) il*n_expert + e] >= 0) {
                continue;
            }
            const int32_t s = victim(il);
            assign(s, il, e);
            move(s, false);
            misses++;
        }
    }
};

// The proposal: the least recently used slot while it is stale, i.e. its
// expert was not chosen during the last pass over the layers (its own layer
// came and went without it); when every slot was used within the last pass,
// the step wants more experts than the cache holds, and the slot taken is
// the most recently used one outside the current layer: the expert of the
// layer just computed, whose next use is a whole pass away, instead of the
// one the next layer is about to ask for. Below its capacity the cache
// behaves as LRU; above it, the slots filled early in the pass stay, and the
// rest of the pass streams through the slots last taken.
struct policy_stale_mru : policy {
    lru_list lru;
    std::vector<uint64_t> last_t; // [slot] plan counter of the last use
    uint64_t t = 0;
    uint64_t pass_len;

    policy_stale_mru(int32_t l, int32_t e, int32_t s, uint64_t pass_len) :
        policy(l, e, s), lru(s), last_t(s, 0), pass_len(pass_len) {}
    const char * name() const override { return "stale-mru"; }

    void plan(size_t, int32_t il, const int32_t * ids, size_t n, size_t & hits, size_t & misses) override {
        ++t;
        distinct(ids, n);
        int32_t first = -1; // the oldest slot of this layer's, at the tail of the list
        for (int32_t e : uniq) {
            const int32_t s = slot_of[(size_t) il*n_expert + e];
            if (s >= 0) {
                lru.touch(s);
                last_t[s] = t;
                if (first < 0) {
                    first = s;
                }
                hits++;
            }
        }
        std::sort(uniq.begin(), uniq.end());
        for (int32_t e : uniq) {
            if (slot_of[(size_t) il*n_expert + e] >= 0) {
                continue;
            }
            int32_t s = lru.head;
            const bool stale = key_of[s] < 0 || t - last_t[s] >= pass_len;
            if (!stale) {
                // fewer distinct experts than slots, so a slot precedes this layer's
                s = first >= 0 ? lru.prev[first] : lru.tail;
            }
            assign(s, il, e);
            lru.touch(s);
            last_t[s] = t;
            if (first < 0) {
                first = s;
            }
            misses++;
        }
    }
};

// Counts that decay, kept for every expert of every layer whether it is in
// the cache or not (as Strata keeps them): each choice adds 1 to a count
// halved every `half_life` passes, and the slot whose expert has the lowest
// count, outside the current layer, goes. An expert seldom chosen gets a
// slot for its layer and is the next to lose it; one chosen often stays,
// whatever the order of the layers.
struct policy_lfu : policy {
    std::vector<double> count;    // [layer*n_expert + expert] at last_t
    std::vector<uint64_t> last_t;
    std::vector<uint32_t> in_plan; // [slot]
    uint32_t gen = 0;
    double log_d; // ln of the decay per plan
    uint64_t t = 0;
    std::set<std::pair<double, int32_t>> order; // the slots by their expert's key, ln(count) - t*ln(d)
    std::vector<double> slot_key;

    policy_lfu(int32_t l, int32_t e, int32_t s, double half_life_plans) :
        policy(l, e, s), count((size_t) l*e, 0), last_t((size_t) l*e, 0), in_plan(s, 0),
        log_d(std::log(0.5)/half_life_plans), slot_key(s, -1e300) {
        for (int32_t i = 0; i < s; ++i) {
            order.insert({ slot_key[i], i }); // empty slots first
        }
    }
    const char * name() const override { return "lfu-decay"; }

    double chosen(size_t k) {
        count[k] = count[k]*std::exp(log_d*(double) (t - last_t[k])) + 1.0;
        last_t[k] = t;
        return std::log(count[k]) - (double) t*log_d;
    }

    void rekey(int32_t s, double key) {
        order.erase({ slot_key[s], s });
        slot_key[s] = key;
        order.insert({ key, s });
        in_plan[s] = gen;
    }

    void plan(size_t, int32_t il, const int32_t * ids, size_t n, size_t & hits, size_t & misses) override {
        ++t;
        ++gen;
        distinct(ids, n);
        std::vector<double> keys(uniq.size());
        for (size_t j = 0; j < uniq.size(); ++j) {
            keys[j] = chosen((size_t) il*n_expert + uniq[j]);
        }
        for (size_t j = 0; j < uniq.size(); ++j) {
            const int32_t s = slot_of[(size_t) il*n_expert + uniq[j]];
            if (s >= 0) {
                rekey(s, keys[j]);
                hits++;
            }
        }
        for (size_t j = 0; j < uniq.size(); ++j) {
            if (slot_of[(size_t) il*n_expert + uniq[j]] >= 0) {
                continue;
            }
            int32_t s = -1;
            for (const auto & k : order) {
                if (in_plan[k.second] != gen) {
                    s = k.second;
                    break;
                }
            }
            assign(s, il, uniq[j]);
            rekey(s, keys[j]);
            misses++;
        }
    }
};

// stale-mru when a pass overflows the cache, but among the stale slots the
// one whose expert has the lowest decayed count (kept for every expert, as
// lfu-decay) instead of the least recently used: on the c10 traces of
// 16-10-2026 the counts won where the steps fit and lost where they did not.
// `window` bounds how many stale slots, from the least recently used, are
// compared (0: all of them), as a cache in llama.cpp could afford per miss.
// With `behind`, a full pass also goes by the counts: the miss takes the
// slot with the lowest count among those used earlier in this pass (their
// next use is a pass away), the most recently used only if there is none
// ("lfu-pass").
struct policy_stale_lfu : policy {
    lru_list lru;
    std::vector<uint64_t> last_t;  // [slot] plan of the last use
    std::vector<double> count;     // [layer*n_expert + expert] at count_t
    std::vector<uint64_t> count_t;
    uint64_t t = 0;
    uint64_t pass_len;
    double log_d;
    int32_t window;
    bool behind;
    std::string label;
    int32_t last_il = INT32_MAX;
    uint64_t pass_start = 0;

    policy_stale_lfu(int32_t l, int32_t e, int32_t s, uint64_t pass_len, double half_life_plans, int32_t window,
                     bool behind) :
        policy(l, e, s), lru(s), last_t(s, 0), count((size_t) l*e, 0), count_t((size_t) l*e, 0),
        pass_len(pass_len), log_d(std::log(0.5)/half_life_plans), window(window), behind(behind),
        label(std::string(behind ? "lfu-pass" : "stale-lfu") + (window > 0 ? "/" + std::to_string(window) : "")) {}
    const char * name() const override { return label.c_str(); }

    double score(size_t k) const { return count[k]*std::exp(log_d*(double) (t - count_t[k])); }

    void plan(size_t, int32_t il, const int32_t * ids, size_t n, size_t & hits, size_t & misses) override {
        ++t;
        if (il <= last_il) {
            pass_start = t;
        }
        last_il = il;
        distinct(ids, n);
        for (int32_t e : uniq) {
            const size_t k = (size_t) il*n_expert + e;
            count[k] = score(k) + 1.0;
            count_t[k] = t;
        }
        int32_t first = -1;
        for (int32_t e : uniq) {
            const int32_t s = slot_of[(size_t) il*n_expert + e];
            if (s >= 0) {
                lru.touch(s);
                last_t[s] = t;
                if (first < 0) {
                    first = s;
                }
                hits++;
            }
        }
        std::sort(uniq.begin(), uniq.end());
        for (int32_t e : uniq) {
            if (slot_of[(size_t) il*n_expert + e] >= 0) {
                continue;
            }
            int32_t s = lru.head;
            if (key_of[s] >= 0 && t - last_t[s] < pass_len) {
                s = first >= 0 ? lru.prev[first] : lru.tail;
                if (behind && last_t[s] >= pass_start) {
                    // the slots used earlier in this pass, the latest first
                    double best = score(key_of[s]);
                    int32_t seen = 1;
                    for (int32_t c = lru.prev[s]; c >= 0 && last_t[c] >= pass_start && (window == 0 || seen < window);
                         c = lru.prev[c], ++seen) {
                        const double sc = score(key_of[c]);
                        if (sc < best) {
                            best = sc;
                            s = c;
                        }
                    }
                }
            } else if (key_of[s] >= 0) {
                // the stale slots are the head of the list; the lowest count among them
                double best = score(key_of[s]);
                int32_t seen = 1;
                for (int32_t c = lru.next[s]; c >= 0 && (window == 0 || seen < window); c = lru.next[c], ++seen) {
                    if (key_of[c] < 0) {
                        s = c;
                        break;
                    }
                    if (t - last_t[c] < pass_len) {
                        break;
                    }
                    const double sc = score(key_of[c]);
                    if (sc < best) {
                        best = sc;
                        s = c;
                    }
                }
            }
            assign(s, il, e);
            lru.touch(s);
            last_t[s] = t;
            if (first < 0) {
                first = s;
            }
            misses++;
        }
    }
};

// Belady's: the slot whose expert is next chosen furthest in the future
// goes. Needs the whole trace, so it is a bound, not a policy.
struct policy_opt : policy {
    // [plan] next plan index at which each of the plan's distinct experts is chosen again, in uniq order
    const std::vector<std::vector<std::pair<int32_t, size_t>>> & next_use;
    std::vector<size_t> next_of; // [slot]
    std::vector<uint32_t> in_plan;
    uint32_t gen = 0;
    std::set<std::pair<size_t, int32_t>, std::greater<>> order; // furthest first

    policy_opt(int32_t l, int32_t e, int32_t s, const std::vector<std::vector<std::pair<int32_t, size_t>>> & nu) :
        policy(l, e, s), next_use(nu), next_of(s, SIZE_MAX), in_plan(s, 0) {
        for (int32_t i = 0; i < s; ++i) {
            order.insert({ SIZE_MAX, i });
        }
    }
    const char * name() const override { return "belady (bound)"; }

    void plan(size_t i, int32_t il, const int32_t * ids, size_t n, size_t & hits, size_t & misses) override {
        ++gen;
        distinct(ids, n);
        std::map<int32_t, size_t> nxt(next_use[i].begin(), next_use[i].end());
        auto set_next = [&](int32_t s, size_t v) {
            order.erase({ next_of[s], s });
            next_of[s] = v;
            order.insert({ v, s });
            in_plan[s] = gen;
        };
        for (int32_t e : uniq) {
            const int32_t s = slot_of[(size_t) il*n_expert + e];
            if (s >= 0) {
                set_next(s, nxt[e]);
                hits++;
            }
        }
        for (int32_t e : uniq) {
            if (slot_of[(size_t) il*n_expert + e] >= 0) {
                continue;
            }
            int32_t s = -1;
            for (const auto & k : order) {
                if (in_plan[k.second] != gen) {
                    s = k.second;
                    break;
                }
            }
            assign(s, il, e);
            set_next(s, nxt[e]);
            misses++;
        }
    }
};

std::vector<std::vector<std::pair<int32_t, size_t>>> next_uses(const trace & t) {
    std::vector<std::vector<std::pair<int32_t, size_t>>> res(t.n_plans());
    std::vector<size_t> next((size_t) t.n_layer*t.n_expert, SIZE_MAX);
    std::vector<uint32_t> seen(t.n_expert, 0);
    uint32_t gen = 0;
    for (size_t i = t.n_plans(); i-- > 0;) {
        ++gen;
        const int32_t il = t.il[i];
        for (size_t j = t.offset[i]; j < t.offset[i + 1]; ++j) {
            const int32_t e = t.ids[j];
            if (seen[e] != gen) {
                seen[e] = gen;
                res[i].push_back({ e, next[(size_t) il*t.n_expert + e] });
            }
        }
        for (const auto & p : res[i]) {
            next[(size_t) il*t.n_expert + p.first] = i;
        }
    }
    return res;
}

// ---------------------------------------------------------------- synthetic routing

// Each layer has a ranking of its experts, the prompt's, and an expert of
// rank r is chosen with weight (r+1)^-alpha. A sequence keeps each expert
// of its previous token with probability `keep`, and every token `drift`
// pairs of ranks swap in each layer, so the experts in use change slowly.
// With `shared` the sequences answer the same prompt (the campaign's
// throughput points do) and share the layer's ranking; a fraction `own` of
// their draws comes from a ranking of their own.
struct synth_opts {
    int32_t n_layer = 94, n_expert = 128, n_used = 8;
    int32_t batch = 1, tokens = 150, rounds = 1;
    double alpha = 1.0, keep = 0.5, own = 0.2;
    int32_t drift = 2;
    bool shared = true;
    uint64_t seed = 1;
};

trace synth(const synth_opts & o) {
    std::mt19937_64 rng(o.seed);
    std::vector<double> cdf(o.n_expert);
    double acc = 0.0;
    for (int32_t r = 0; r < o.n_expert; ++r) {
        acc += std::pow(r + 1.0, -o.alpha);
        cdf[r] = acc;
    }
    std::uniform_real_distribution<double> u01(0.0, 1.0);
    auto rank = [&]() {
        return (int32_t) (std::lower_bound(cdf.begin(), cdf.end(), u01(rng)*acc) - cdf.begin());
    };
    auto shuffled = [&]() {
        std::vector<int32_t> p(o.n_expert);
        for (int32_t i = 0; i < o.n_expert; ++i) {
            p[i] = i;
        }
        std::shuffle(p.begin(), p.end(), rng);
        return p;
    };

    trace t;
    t.n_layer = o.n_layer; t.n_expert = o.n_expert; t.n_used = o.n_used;
    for (int32_t l = 0; l < o.n_layer; ++l) {
        t.layers.push_back(l);
    }
    std::vector<std::vector<int32_t>> layer_rank(o.n_layer);
    for (auto & p : layer_rank) {
        p = shuffled();
    }
    std::uniform_int_distribution<int32_t> any_rank(0, o.n_expert - 1);
    std::vector<int32_t> ids;
    std::vector<uint8_t> taken(o.n_expert);
    for (int32_t round = 0; round < o.rounds; ++round) {
        // a round: `batch` requests started together, `tokens` tokens each
        std::vector<std::vector<std::vector<int32_t>>> own_rank(o.batch, std::vector<std::vector<int32_t>>(o.n_layer));
        std::vector<std::vector<std::vector<int32_t>>> prev(o.batch, std::vector<std::vector<int32_t>>(o.n_layer));
        for (auto & seq : own_rank) {
            for (auto & p : seq) {
                p = shuffled();
            }
        }
        for (int32_t tok = 0; tok < o.tokens; ++tok) {
            for (int32_t l = 0; l < o.n_layer; ++l) {
                ids.clear();
                for (int32_t b = 0; b < o.batch; ++b) {
                    std::fill(taken.begin(), taken.end(), 0);
                    std::vector<int32_t> cur;
                    for (int32_t e : prev[b][l]) {
                        if (u01(rng) < o.keep) {
                            cur.push_back(e);
                            taken[e] = 1;
                        }
                    }
                    while ((int32_t) cur.size() < o.n_used) {
                        const bool mine = !o.shared || u01(rng) < o.own;
                        const int32_t e = (mine ? own_rank[b][l] : layer_rank[l])[rank()];
                        if (!taken[e]) {
                            taken[e] = 1;
                            cur.push_back(e);
                        }
                    }
                    prev[b][l] = cur;
                    ids.insert(ids.end(), cur.begin(), cur.end());
                }
                t.add(l, ids.data(), ids.size());
                for (int32_t k = 0; k < o.drift; ++k) {
                    std::swap(layer_rank[l][any_rank(rng)], layer_rank[l][any_rank(rng)]);
                    for (int32_t b = 0; b < o.batch; ++b) {
                        std::swap(own_rank[b][l][any_rank(rng)], own_rank[b][l][any_rank(rng)]);
                    }
                }
            }
        }
    }
    return t;
}

// ---------------------------------------------------------------- run

struct run_opts {
    int32_t slots = 0;
    double expert_mib = 0.0; // bytes of one expert (all its tensors), for the copies
    double gbps = 22.0;      // the link, GB/s (LUMI measured 16-24)
    double half_life = 4.0;  // passes, lfu-decay
    bool opt = true;
    bool csv = false;
};

void run(const trace & t, const run_opts & o, const std::string & label) {
    const int32_t slots = o.slots > 0 ? o.slots : t.n_slots;
    if (slots <= 0) {
        throw std::runtime_error("--slots is needed (the trace does not say)");
    }
    const uint64_t pass_len = t.layers.size();
    const double total_experts = (double) pass_len*t.n_expert;

    size_t n_ids = 0;
    for (size_t i = 0; i < t.n_plans(); ++i) {
        n_ids += t.offset[i + 1] - t.offset[i];
    }
    const double passes = (double) t.n_plans()/pass_len;
    const double tokens_per_pass = n_ids/(double) t.n_used/t.n_plans();

    std::vector<std::unique_ptr<policy>> ps;
    ps.emplace_back(new policy_lru(t.n_layer, t.n_expert, slots));
    ps.emplace_back(new policy_slru(t.n_layer, t.n_expert, slots));
    ps.emplace_back(new policy_stale_mru(t.n_layer, t.n_expert, slots, pass_len));
    ps.emplace_back(new policy_lfu(t.n_layer, t.n_expert, slots, o.half_life*pass_len));
    ps.emplace_back(new policy_stale_lfu(t.n_layer, t.n_expert, slots, pass_len, o.half_life*pass_len, 0, false));
    ps.emplace_back(new policy_stale_lfu(t.n_layer, t.n_expert, slots, pass_len, o.half_life*pass_len, 0, true));
    ps.emplace_back(new policy_stale_lfu(t.n_layer, t.n_expert, slots, pass_len, o.half_life*pass_len, 32, true));
    std::vector<std::vector<std::pair<int32_t, size_t>>> nu;
    if (o.opt) {
        nu = next_uses(t);
        ps.emplace_back(new policy_opt(t.n_layer, t.n_expert, slots, nu));
    }

    if (!o.csv) {
        printf("%s: %zu plans (%.0f passes of %.1f tokens), %d slots = %.1f%% of %.0f experts\n",
            label.c_str(), t.n_plans(), passes, tokens_per_pass, slots, 100.0*slots/total_experts, total_experts);
    }
    for (auto & p : ps) {
        size_t hits = 0, misses = 0;
        for (size_t i = 0; i < t.n_plans(); ++i) {
            p->plan(i, t.il[i], t.ids.data() + t.offset[i], t.offset[i + 1] - t.offset[i], hits, misses);
        }
        const double demand = (double) (hits + misses)/passes;
        const double copied = (double) misses/passes;
        const double gib = copied*o.expert_mib/1024.0;
        const double ms = o.expert_mib > 0 ? copied*o.expert_mib*1048576.0/(o.gbps*1e9)*1e3 : 0.0;
        if (o.csv) {
            printf("%s,%s,%d,%.1f,%.2f,%.0f,%.0f,%.3f,%.1f\n", label.c_str(), p->name(), slots, tokens_per_pass,
                100.0*hits/(hits + misses), demand, copied, gib, ms);
        } else {
            printf("  %-15s hits %6.2f%%  per pass: %6.0f experts wanted, %6.0f copied", p->name(),
                100.0*hits/(hits + misses), demand, copied);
            if (o.expert_mib > 0) {
                printf(" = %6.2f GiB, %7.1f ms at %.0f GB/s", gib, ms, o.gbps);
            }
            printf("\n");
        }
    }
}

[[noreturn]] void usage() {
    fprintf(stderr,
        "usage: moe_cache_sim run TRACE [--slots N] [--expert-mib M] [--gbps G] [--half-life P] [--no-opt] [--csv]\n"
        "       moe_cache_sim synth [--layers L] [--experts E] [--used K] [--batch B] [--tokens T] [--rounds R]\n"
        "                           [--alpha A] [--keep P] [--own P] [--drift D] [--independent] [--seed S]\n"
        "                           (--out TRACE | --slots N [run options])\n");
    exit(2);
}

} // namespace

int main(int argc, char ** argv) {
    if (argc < 2) {
        usage();
    }
    const std::string cmd = argv[1];
    run_opts ro;
    synth_opts so;
    std::string path, out, label;
    for (int i = 2; i < argc; ++i) {
        const std::string a = argv[i];
        auto val = [&]() -> const char * {
            if (i + 1 >= argc) {
                usage();
            }
            return argv[++i];
        };
        if (a == "--slots") ro.slots = atoi(val());
        else if (a == "--expert-mib") ro.expert_mib = atof(val());
        else if (a == "--gbps") ro.gbps = atof(val());
        else if (a == "--half-life") ro.half_life = atof(val());
        else if (a == "--no-opt") ro.opt = false;
        else if (a == "--csv") ro.csv = true;
        else if (a == "--label") label = val();
        else if (a == "--layers") so.n_layer = atoi(val());
        else if (a == "--experts") so.n_expert = atoi(val());
        else if (a == "--used") so.n_used = atoi(val());
        else if (a == "--batch") so.batch = atoi(val());
        else if (a == "--tokens") so.tokens = atoi(val());
        else if (a == "--rounds") so.rounds = atoi(val());
        else if (a == "--alpha") so.alpha = atof(val());
        else if (a == "--keep") so.keep = atof(val());
        else if (a == "--own") so.own = atof(val());
        else if (a == "--drift") so.drift = atoi(val());
        else if (a == "--independent") so.shared = false;
        else if (a == "--seed") so.seed = strtoull(val(), nullptr, 10);
        else if (a == "--out") out = val();
        else if (cmd == "run" && path.empty() && a[0] != '-') path = a;
        else usage();
    }
    try {
        if (cmd == "run") {
            if (path.empty()) {
                usage();
            }
            run(read_trace(path), ro, label.empty() ? path : label);
        } else if (cmd == "synth") {
            trace t = synth(so);
            if (!out.empty()) {
                t.n_slots = ro.slots;
                write_trace(t, out);
            } else {
                if (label.empty()) {
                    label = "synth L" + std::to_string(so.n_layer) + " E" + std::to_string(so.n_expert) +
                            " b" + std::to_string(so.batch);
                }
                run(t, ro, label);
            }
        } else {
            usage();
        }
    } catch (const std::exception & e) {
        fprintf(stderr, "moe_cache_sim: %s\n", e.what());
        return 1;
    }
    return 0;
}
