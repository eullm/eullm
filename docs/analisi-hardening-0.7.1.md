# Analisi hardening — 0.7.1 (2026-08-23)

Rapporto di revisione puntuale, complementare a
[`backlog-fix-e-hardening.md`](backlog-fix-e-hardening.md). Nasce da una
sessione di debugging su hardware reale (RTX 5070 Ti, Blackwell/sm_120) del
divario di prestazioni e VRAM tra eullm e `llama-cli` stock su un GGUF
Qwen3.8-27B ibrido, che ha poi innescato una revisione a tavolino su quattro
fronti in parallelo: **sicurezza del perimetro**, **correttezza dello
scheduler di inferenza**, **riscontro del backlog contro il codice 0.7.1**, e
**igiene del codice**.

Metodo: ogni voce cita file e riga ed è stata verificata leggendo il codice
attorno, non solo la riga segnalata — vale la regola del backlog («nessuna
voce senza riferimento al sorgente»). Le voci marcate CONFERMATO sono
riproducibili leggendo il codice; PLAUSIBILE richiede una condizione al
contorno (metadati malformati, frammentazione di token multi-byte) non ancora
costruita come test.

Baseline: Engine 0.7.1 · llama.cpp `e79e4bf6` (b10405) · 288 unit test verdi.

---

## Chiuse in questa sessione

- [x] **S-01 · Path traversal nei lookup per nome del model store** *(P1, medium,
  CONFERMATO)* — commit `8d18260`.
  Il campo `model` di una richiesta API raggiungeva `store.gguf_path` senza
  validazione: `api::resolve_model` (`api/mod.rs:857`) filtra la scorciatoia
  launch-model, le forme-percorso opt-in e il join su `/models`, ma allo step 5
  passava il nome grezzo. `gguf_path` faceva `root.join(name)` senza
  `is_safe_filename`, quindi `{"model":"../../../../tmp/x"}` risolveva fuori
  dallo store — caricando un `.gguf` là, e via lo stesso pattern in `delete()`
  **rimuovendo** una directory fuori dallo store, oltre a fare da oracolo di
  esistenza per `.gguf`/`manifest.json` ovunque il processo possa leggere. Fix:
  un helper unico `safe_model_dir` (strip prefisso `eullm/`, `is_safe_filename`,
  join) su tutti e sei i metodi che condividevano il pattern
  (`gguf_path`/`mmproj_path`/`get`/`exists`/`delete` falliscono in sicurezza;
  `model_path`, che riceve solo id fidati, mappa un nome pericoloso su un
  placeholder non-escaping). Test di regressione con pesi piantati fuori dallo
  store. Nella stessa passata corretto un `mmproj_path` case-sensitive che
  divergeva dagli altri tre siti di detection mmproj.

- [x] **S-02 · Checkpoint a lunghezza piena campionava da logit stantii** *(P1,
  high, CONFERMATO)* — commit `6773093`.
  `best_checkpoint` (`scheduler.rs:613`) accettava un checkpoint i cui token
  **uguagliano** la richiesta. Ripristinarlo poneva `effective_reuse_len =
  tokens.len()`, così il loop di decodifica di `prefill_sequence` su
  `tokens[reuse_len..]` era vuoto: nessuna decodifica, nessuna riga di logit
  fresca, e `sampler.sample(&ctx, -1)` leggeva i logit che il contesto condiviso
  conteneva per ultimi — la distribuzione di un'altra sequenza o del turno
  precedente. Si innescava su modello ibrido/ricorrente (`rs_seq=0`) con
  `--ctx-checkpoints` attivo alla ri-sottomissione di un prompt identico. Fix:
  richiedere un prefisso **stretto** (`c.tokens.len() < tokens.len()`),
  l'equivalente-checkpoint del cap a `len-1` che `pick_slot` e il fast-path
  testuale già applicano — resta sempre ≥1 token da decodificare. Nessun test
  copriva il restore da checkpoint; aggiunto.

- [x] **S-03 · Due commenti/valori resi obsoleti dal revert n_ubatch 1024→512**
  *(P3, docs)* — commit `f8d73f7`.
  Il commento di `COMPUTE_BUFFER_RESERVE_BYTES` (`fit.rs:655`) dichiarava
  `n_ubatch=1024` e «nessuna misura reale»; entrambi ora falsi (n_ubatch è 512,
  e il buffer è stato misurato a ~507 MiB via la tabella memory-breakdown).
  Corretto il commento e agganciato a H2-H per la ri-calibrazione (valore
  lasciato a 640 MiB: abbassarlo è un cambio da validare su hardware, un
  under-reserve va in OOM a caricamento). Il probe multimodale
  (`inference/mod.rs:1523`) usava `n_batch.min(1024)` sul ramo testo mentre il
  percorso reale usa `min(512)`: allineato, e commento corretto.

- [x] **S-04 · `n_outputs_max` non limitato — ~300 MiB di compute buffer sprecati**
  *(P2)* — commit `e8803a1` (già in `docs/backlog` come follow-up di H3-Y).
  Causa del divario di compute buffer contro `llama-cli` (507 vs 164 MiB su un
  27B): con `n_outputs_max` al default (0 = n_batch), la riserva del grafo a
  caricamento dimensiona l'output LM-head come `[n_vocab, min(n_ubatch,
  n_outputs_max)]` nel compute buffer del device — ~300 MiB per un vocabolario
  ~150k, per righe di logit che lo scheduler non chiede mai (legge un logit per
  sequenza per passo). Impostato a `max_batch_size` su entrambi i percorsi di
  contesto dello scheduler. Contesto embedding non toccato (le embedding pooled
  emettono ogni token).

---

## Aperte — sicurezza / correttezza

- [ ] **S-05 · Desync `CachedSlot.text`/`tokens` via decoder UTF-8 in streaming**
  *(P2, medium, CONFERMATO logica)* `scheduler.rs:1841-1854`.
  `raw_generated_pieces[i]` è output di `token_to_piece` attraverso un decoder
  **stateful**: un token che finisce a metà carattere produce un pezzo corto o
  vuoto, i cui byte vengono portati nel pezzo del token successivo. Se la
  generazione termina (stop / max-token) subito dopo un token del genere,
  `resident_text` (che scarta gli ultimi *pezzi*) e `resident_tokens` (che
  tronca a `n_past`) divergono: i byte penzolanti del token residente non sono
  in `slot.text`. Un prompt successivo che combacia con `slot.text` via
  `text_prefix_match` (`:987`) riusa `slot.tokens` + il suffisso ri-tokenizzato,
  e la KV contiene byte spuri (mezza emoji) che il testo del prompt non ha →
  generazione condizionata su un prompt che il client non ha mai inviato,
  output sbagliato silenzioso. `keep = raw_generated_pieces.len().saturating_sub(
  extra)` scarta interi pezzi, non può esprimere un riporto a livello di byte.
  Nessun test multi-byte esercita la coppia slot-caching. Fix candidato:
  troncare `resident_text` sul confine di byte del token residente, o invalidare
  il riuso testuale quando l'ultimo pezzo è incompleto.

- [ ] **S-06 · `Done` e tail-flush persi su canale pieno (`try_send`)** *(P3,
  low-med, CONFERMATO)* `scheduler.rs:1592, 1913`.
  La backpressure trattiene il testo dei token in `pending`, ma se il client si
  blocca finché il canale da 256 slot si riempie, sia il flush della coda a
  max-token (`try_send(StreamEvent::Token(tail))`) sia `send_done`
  (`try_send(StreamEvent::Done{...})`) vengono scartati. Un client lento ma vivo
  che poi svuota riceve uno stream che finisce senza `done` finale e senza il
  testo di coda. Il test di backpressure (`:2509`) copre solo il percorso token,
  non `Done`. Fix candidato: usare il percorso bloccante/`pending` anche per
  `Done` e il tail.

- [ ] **S-07 · `repeat_last_n: -1` di Ollama disabilita silenziosamente la
  penalità** *(P3, low, PLAUSIBILE)* `sampling.rs:65`, `api/routes.rs:256`.
  `repeat_last_n` passa non clampato dal JSON a `LlamaSampler::penalties`; la
  llama.cpp vendorizzata fa `penalty_last_n = std::max(penalty_last_n, 0)`,
  mentre Ollama documenta `-1` come «usa num_ctx». Un client che chiede la
  penalità più forte non ne ottiene nessuna. Fix candidato: mappare `-1` alla
  dimensione del contesto prima di passarlo.

- [ ] **S-08 · `max_tokens == 0` diverge tra i due motori** *(P3, low,
  CONFERMATO)* `scheduler.rs:1216`, `inference/mod.rs:1858`.
  Sul percorso scheduler il primo token è campionato, contato ed emesso *prima*
  del check `tokens_generated >= max_tokens`, quindi `num_predict: 0` produce 1
  token; il motore sequenziale con `while ... < max_tokens` ne produce 0.
  Comportamento divergente per la stessa richiesta. Va prima deciso quale sia
  la semantica corretta di `num_predict: 0` (Ollama: caricamento senza
  generazione) e allineati i due motori.

- [ ] **S-09 · Metadati interi con segno negativi reinterpretati come `u32::MAX`**
  *(P3, low, PLAUSIBILE)* `fit.rs:236-247, 330-338`.
  `read_uint_as_u64` legge INT8/16/32/64 come byte senza segno; `*slot =
  u32::try_from(v).ok().or(Some(u32::MAX))`. Un `full_attention_interval`
  negativo (corrotto) diventa `u32::MAX` → `kv_paying_layers = 1` → KV
  massicciamente sotto-stimata → l'OOM che `--fit` esiste per prevenire.
  Richiede metadati malformati. Fix candidato: trattare un intero con segno
  negativo come metadato assente (`None`) invece di clamparlo.

- [ ] **S-10 · Elisione filtro e stop nello stesso `process_piece`** *(P3, low,
  PLAUSIBILE)* `inference/output.rs:89-111`.
  Se una sequenza-filtro si completa nello *stesso* `process_piece` di uno stop
  (marker assemblati su più pezzi), il check di stop precede l'elisione del
  filtro: `</think>` può trapelare nonostante `think:false`, o simmetricamente
  la rimozione del filtro può ricomporre `pending` in una stringa contenente una
  stop-sequence che l'holdback del prefisso proprio poi emette senza fermarsi.
  Serve che frammenti di marker arrivino insieme in un pezzo — raro perché gli
  stop-marker sono di solito token singoli, da cui PLAUSIBILE. Non testato.

- [ ] **S-11 · Richieste in coda scartate senza evento `Error` allo shutdown**
  *(P4, nit, CONFERMATO)* `scheduler.rs:963-968, 1344-1350`.
  Le richieste ancora in coda (non ancora attive) allo shutdown o alla
  disconnessione vengono droppate senza un `StreamEvent::Error`, quindi il
  client non distingue «rifiutata» da «mai arrivata».

---

## Aperte — dal riscontro del backlog contro 0.7.1

Voci del backlog storico ancora valide, verificate contro il codice 0.7.1 (ID
originali conservati):

- [ ] **H1-F · Perimetro dell'Hub** — la metà porta è sistemata (default 3000),
  la metà sicurezza no: l'Hub lega `0.0.0.0:{port}` senza allowlist/auth/CORS
  (`hub/src/main.rs`, solo validazione slug).
- [ ] **H2-F · Irrobustire il patcher GGUF** — `gguf_patch.rs`: allocazioni da
  lunghezze u64 fornite dal file senza bound (`:216`, `:249`).
- [ ] **H2-H · Ri-calibrare `COMPUTE_BUFFER_RESERVE_BYTES`** — ora con una misura
  reale disponibile (~507 MiB pre-fix S-04, atteso più basso dopo), il valore
  640 MiB è probabilmente ~2x sovradimensionato; ricalibrare su hardware dopo la
  validazione di S-04. Commento già corretto (S-03).
- [ ] **H3-E · Reproducibilità pipeline Forge** — nessun lockfile,
  dipendenze `>=` senza upper bound (`forge/pyproject.toml`).
- [ ] **H3-I · Unificare i due cataloghi** — l'Hub hardcoda sette modelli
  `eullm/*` inesistenti separati da `catalog/v1/catalog.json`.
- [ ] **H3-J · Igiene audit trail** — `AuditLogger` costruito per richiesta,
  file riaperto per scrittura, nessun fsync/rotazione, `read_all`/`count`
  leggono l'intero file.
- [ ] **H3-S · Backend GPU caricati a runtime** — il re-vendor b10405 ha portato
  la feature upstream `dynamic-backends` (`GGML_BACKEND_DL`), non usata: il primo
  passo dell'item è ora più economico.
- [ ] **H3-T · Sizing batch del probe multimodale** — parità multimodale intatta;
  il ramo testo è stato allineato (S-03). Resta la ri-validazione hardware per
  famiglia.
- [ ] **H4-C · `--fit` scelga anche il contesto** — nessuna selezione automatica
  del contesto: ogni funzione di sizing prende `ctx_size` come input.
- [ ] **H4-D · Context shift** — assente: la generazione finisce a
  `done_reason=length` quando la finestra si riempie.
- [ ] **H4-E · La stima KV del banner ignora gli ibridi** —
  `estimate_kv_memory` addebita KV uniformemente su `n_layer` mentre `fit.rs`
  sconta via `full_attention_interval`: due calcoli per un numero.
- [ ] **H4-F(b) · Collisione quant HF** — il sintomo (a) è risolto (quant
  nell'id); resta (b): un nome repo nudo ri-scarica il quant default invece di
  offrire la directory già presente.
- [ ] **H4-G · Documentazione tecnica indietro rispetto al codice** —
  `docs/engine.md:325` dice ancora «8 slot paralleli (default)» / `--batch-size
  8` (il default è 1 dalla 0.6.36).

Chiudibili: **H3-R** (bump b10200 con patch a mano) è superata da H3-Y (re-vendor
pulito b10405, patch eliminate); chiudere con rimando.

**Nessuna regressione** trovata sui fix chiusi dal bump llama.cpp b10405 o dal
revert n_ubatch: verificati H2-T (Metal-off), H2-L (auto-correzione flash-attn
q4_0), H2-U (filtri Harmony), H2-AA (key_length nella stima KV), H3-V (margini
del probe), H3-X/H4-A (FFI del wrapper chat template) — tutti intatti.

---

## Igiene del codice

Albero insolitamente pulito: **zero codice morto** (verificato ogni simbolo
contro l'intero albero), **zero marker TODO/FIXME/HACK**, nessun blocco di
codice commentato. Da valutare:

- [ ] **G-01 · Callback di progresso download duplicato ×4** *(P3)* `main.rs:1051`
  ha l'helper `download_progress()`, ma tre re-implementazioni inline restano
  (`:1248`, `:1393`, `:1502`) e **divergono**: usano `downloaded - last` grezzo
  dove l'helper usa `saturating_sub` (panic da underflow in debug se un download
  ripreso riporta un offset minore). Fix: chiamare l'helper nei tre punti.
- [ ] **G-02 · Blocchi response-JSON + audit-entry duplicati in routes.rs** *(P4)*
  ogni endpoint ricostruisce l'oggetto risposta ~20 righe + entry audit una
  volta per percorso invece di una per endpoint (`generate` ×2, `chat` ×3,
  `chat_completions` ×3). È anche perché quelle funzioni superano le 250 righe.

Clippy pedantic (424 warning in `engine/src`, esclusi i lint-rumore): meritano
un'occhiata `sha256_file` con buffer da 1 MiB sullo stack (`registry/mod.rs:98`)
e ~120 cast numerici, di cui quelli nell'aritmetica di sizing VRAM
(`fit.rs:782,878,885`, `api/mod.rs:954`) sono i soli che vale la pena guardare.

---

## Verificato solido (evidenza di revisione, non di salto)

- **Perimetro API** — `auth.rs`: segreti solo come digest SHA-256, confronto
  timing-safe che scandisce tutte le chiavi senza early-return, config
  malformata fatale all'avvio, mai eco del segreto; `origin.rs`: loopback per
  uguaglianza non suffisso (`localhost.evil.example` respinto), `null`/vuoto mai
  ammessi; `ip_allowlist.rs`: default loopback-only, env malformata restringe
  non allarga; middleware: ordine auth→ip→origin→cors corretto, CORS su metodi
  non-safe applicato *prima* dell'handler, body cappato a 64 MB, override slot
  con bound rigidi.
- **Guardia SSRF** (`tools/guard.rs`) — ogni indirizzo risolto deve passare;
  connessione pinnata via `ClientBuilder::resolve` all'indirizzo validato
  (rebinding DNS chiuso); redirect disabilitati e seguiti a mano con
  ri-validazione a ogni hop; forme IPv4-mapped/compat/NAT64/6to4 spacchettate e
  giudicate sull'IPv4 incapsulato; letterali decimali/esadecimali/ottali
  normalizzati; body con cap rigido.
- **Hub** (`hub/src/main.rs`) — `download_model` valida lo slug **e**
  canonicalizza verificando `starts_with(canonical_root)` prima di leggere;
  risposta in streaming; endpoint compliance/model-card fanno 404 per nomi
  sconosciuti (nessuna attestazione fabbricata).
- **FFI C++ EuLLM** (`wrapper_common.cpp`) — out-param azzerati in testa, tutti
  gli input null-checked, ogni percorso in `catch`, nessun leak degli out-param
  già allocati sui percorsi d'errore; lato Rust (`model.rs`) le CString restano
  vive per tutta la chiamata, i C-string di ritorno liberati via
  `llama_rs_string_free`.
- **Scheduler** — `n_outputs_max` limitato correttamente su ogni sito
  `logits=true`; contabilità `n_past` corretta in ogni ramo terminale; ogni
  rimozione da `active` accoppiata a esattamente un push in `idle_slots`
  (invariante `active + idle == max_batch_size`); riciclo `seq_id` con trim
  della coda stantia anche a `reuse_len=0`; fallback F16 identici al primario
  tranne i tipi di cache; clamp `num_ctx`/`effective_ctx` corretto su entrambi i
  motori, nessuna divisione per zero (`batch_size > 0` gate a monte).
- **Aritmetica `fit.rs`** — tutto in byte, nessun mix MiB/byte, divisioni
  guardate, sottrazioni di riserve con `saturating_sub`/`checked_sub`; parser
  GGUF con ogni lettura bounds-checked (nessun panic su input troncato).
