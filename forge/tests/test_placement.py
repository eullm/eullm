"""The device map for a model whose n-gram table stays off the GPUs."""

from __future__ import annotations

from eullm_forge.eval.placement import spread_device_map

NGRAM = "ple.ple_embedding.ngram_embedding.weight"


def _names(n_layers=8):
    names = ["model.language_model.embed_tokens.weight", "model.visual.blocks.0.attn.weight",
             "lm_head.weight"]
    for i in range(n_layers):
        names += [f"model.language_model.layers.{i}.mlp.experts.gate_up_proj",
                  f"model.language_model.layers.{i}.self_attn.q_proj.weight"]
    names += ["model.language_model.layers.2.ple.key_proj.weight",
              "model.language_model.layers.2.ple.ple_embedding.ngram_heads_offsets",
              f"model.language_model.layers.2.{NGRAM}"]
    return names


def test_layers_are_spread_evenly_and_the_table_left_out():
    dm = spread_device_map(_names(), 4, [NGRAM])
    assert dm["model.language_model.layers.0"] == 0
    assert dm["model.language_model.layers.3"] == 1
    assert dm["model.language_model.layers.5"] == 2
    assert dm["model.language_model.layers.7"] == 3
    assert dm["lm_head.weight"] == 3 and dm["model.language_model.embed_tokens.weight"] == 0
    # The table's layer is mapped tensor by tensor, the table itself not at all.
    assert "model.language_model.layers.2" not in dm
    assert dm["model.language_model.layers.2.ple.ple_embedding.ngram_heads_offsets"] == 1
    assert dm["model.language_model.layers.2.ple.key_proj.weight"] == 1
    # No key is the table or a module above it, so nothing sends it to a GPU.
    table = f"model.language_model.layers.2.{NGRAM}"
    assert not any(table == k or table.startswith(k + ".") for k in dm)
    assert set(dm.values()) == {0, 1, 2, 3}
