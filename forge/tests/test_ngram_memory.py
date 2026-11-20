"""The hashed n-gram memory (eullm_forge.memory) on a tiny model, on CPU.

What must hold before an hour of GPU goes into it:
* at scale zero the model with the memory answers exactly as without it;
* the gradient reaches the memory and nothing of the frozen model;
* the table is sparse: a step changes only the rows its batch read;
* the file written is the memory read back, bit for bit;
* decoding one token at a time hashes the same n-grams as the whole sequence.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from eullm_forge.memory import NGramMemory, NGramMemoryConfig, attach, decoder_layers  # noqa: E402


def tiny_model():
    cfg = transformers.Qwen3Config(vocab_size=97, hidden_size=32, intermediate_size=64,
                                   num_hidden_layers=3, num_attention_heads=4,
                                   num_key_value_heads=2, head_dim=8, max_position_embeddings=64)
    torch.manual_seed(0)
    model = transformers.Qwen3ForCausalLM(cfg).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def tiny_memory(**kw):
    cfg = NGramMemoryConfig(vocab_size=97, hidden_size=32, layer=1, rows_per_head=101,
                            embed_dim=64, **kw)
    return NGramMemory(cfg)


def tokens(b=2, t=12, seed=1):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 97, (b, t), generator=g)


def test_finds_the_decoder_layers():
    assert len(decoder_layers(tiny_model())) == 3


def test_at_scale_zero_the_model_is_unchanged():
    model, ids = tiny_model(), tokens()
    with torch.no_grad():
        before = model(input_ids=ids).logits
        handle = attach(model, tiny_memory())
        after = model(input_ids=ids).logits
    assert torch.equal(before, after)
    handle.detach()


def test_with_scale_the_memory_changes_the_output():
    model, ids, mem = tiny_model(), tokens(), tiny_memory()
    with torch.no_grad():
        before = model(input_ids=ids).logits
        mem.scale.fill_(1.0)
        attach(model, mem)
        after = model(input_ids=ids).logits
    assert not torch.allclose(before, after)


def test_gradient_reaches_the_memory_only_and_the_table_sparsely():
    model, ids, mem = tiny_model(), tokens(), tiny_memory()
    with torch.no_grad():
        mem.scale.fill_(0.5)
    attach(model, mem)
    model.train()
    loss = model(input_ids=ids, labels=ids).loss
    loss.backward()
    assert all(p.grad is None for p in model.parameters())
    grad = mem.table.weight.grad
    assert grad is not None and grad.is_sparse
    touched = torch.unique(grad.coalesce().indices())
    assert 0 < touched.numel() <= ids.numel() * mem.config.n_heads
    assert mem.scale.grad is not None and mem.key_proj.weight.grad is not None


def test_scale_zero_still_learns():
    """At scale zero only the scale has a gradient; one step moves it off zero."""
    model, ids, mem = tiny_model(), tokens(), tiny_memory()
    attach(model, mem)
    model.train()
    model(input_ids=ids, labels=ids).loss.backward()
    assert mem.scale.grad is not None and mem.scale.grad.abs().item() > 0


def test_one_step_moves_only_the_rows_read():
    model, ids, mem = tiny_model(), tokens(), tiny_memory()
    with torch.no_grad():
        mem.scale.fill_(0.5)
    attach(model, mem)
    model.train()
    opt = torch.optim.SparseAdam([mem.table.weight], lr=1e-2)
    before = mem.table.weight.detach().clone()
    model(input_ids=ids, labels=ids).loss.backward()
    opt.step()
    changed = (mem.table.weight.detach() != before).any(dim=1).nonzero().flatten()
    read = torch.unique(mem.row_ids(ids))
    assert set(changed.tolist()) <= set(read.tolist())
    assert changed.numel() > 0


def test_save_and_load_are_the_same_memory(tmp_path):
    mem = tiny_memory()
    with torch.no_grad():
        mem.scale.fill_(0.3)
        mem.table.weight[5] += 1.0
    mem.save(tmp_path / "lex-memory")
    back = NGramMemory.load(tmp_path / "lex-memory")
    assert back.config == mem.config
    for (k, a), (_, b) in zip(sorted(mem.state_dict().items()), sorted(back.state_dict().items())):
        assert torch.equal(a, b), k
    ids = tokens()
    assert torch.equal(mem.row_ids(ids), back.row_ids(ids))


def test_hashes_depend_on_the_seed_only():
    ids = tokens()
    assert torch.equal(tiny_memory().row_ids(ids), tiny_memory().row_ids(ids))
    assert not torch.equal(tiny_memory().row_ids(ids), tiny_memory(seed=7).row_ids(ids))


def test_rows_stay_inside_each_heads_slice():
    mem, ids = tiny_memory(), tokens(t=40)
    rows = mem.row_ids(ids)
    lo = mem.head_offsets.view(1, 1, -1)
    hi = lo + mem.head_sizes.view(1, 1, -1)
    assert ((rows >= lo) & (rows < hi)).all()
    assert rows.max() < mem.table.num_embeddings


def test_incremental_decoding_matches_the_whole_sequence():
    """Token by token with the carried history and conv state = the full pass."""
    mem = tiny_memory()
    with torch.no_grad():
        mem.scale.fill_(1.0)
    ids = tokens(b=1, t=10)
    hidden = torch.randn(1, 10, 32)
    with torch.no_grad():
        whole, _ = mem(ids, hidden)
        hist, conv, parts = None, None, []
        for i in range(10):
            out, conv = mem(ids[:, i:i + 1], hidden[:, i:i + 1], hist, conv)
            parts.append(out)
            prev = ids[:, :i + 1]
            pad = torch.full((1, max(0, 2 - prev.shape[1])), 97)
            hist = torch.cat([pad, prev], dim=1)[:, -2:]
    assert torch.allclose(whole, torch.cat(parts, dim=1), atol=1e-5)


def test_generate_with_the_memory_runs_and_is_deterministic():
    model, mem = tiny_model(), tiny_memory()
    with torch.no_grad():
        mem.scale.fill_(1.0)
    attach(model, mem)
    ids = tokens(b=1, t=6)
    with torch.no_grad():
        a = model.generate(ids, max_new_tokens=5, do_sample=False)
        b = model.generate(ids, max_new_tokens=5, do_sample=False)
    assert torch.equal(a, b)


def test_bad_shapes_are_refused():
    with pytest.raises(ValueError):
        NGramMemoryConfig(vocab_size=10, hidden_size=8, embed_dim=10).check()
    with pytest.raises(ValueError):
        NGramMemoryConfig(vocab_size=10, hidden_size=8, ngram_size=1).check()


def test_generate_with_cache_equals_generate_without():
    """The carried history in HF's own decoding loop: cached = recomputed each step."""
    model, mem = tiny_model(), tiny_memory()
    with torch.no_grad():
        mem.scale.fill_(2.0)
        mem.table.weight.mul_(10)
    attach(model, mem)
    ids = tokens(b=2, t=7, seed=3)
    with torch.no_grad():
        cached = model.generate(ids, max_new_tokens=8, do_sample=False, use_cache=True)
        whole = model.generate(ids, max_new_tokens=8, do_sample=False, use_cache=False)
    assert torch.equal(cached, whole)


def test_left_padding_hashes_as_no_token():
    model, mem = tiny_model(), tiny_memory()
    with torch.no_grad():
        mem.scale.fill_(1.0)
    attach(model, mem)
    ids = tokens(b=1, t=5)
    padded = torch.cat([torch.zeros(1, 3, dtype=torch.long), ids], dim=1)
    mask = torch.cat([torch.zeros(1, 3, dtype=torch.long),
                      torch.ones(1, 5, dtype=torch.long)], dim=1)
    with torch.no_grad():
        a = model.generate(ids, max_new_tokens=4, do_sample=False)
        b = model.generate(padded, attention_mask=mask, max_new_tokens=4, do_sample=False)
    assert torch.equal(a[:, 5:], b[:, 8:])
