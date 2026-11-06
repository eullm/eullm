"""Where a model too big for one GPU goes: its decoder layers spread evenly over the GPUs.

transformers' ``device_map="auto"`` keeps room on the first GPU for the
largest layer. Qwen3.8-Flash-Next's layer 2 holds the n-gram table (~95 GiB),
which transformers then keeps on the CPU (``_no_placement_params``), but the
room is kept all the same: on 2026-11-04 the first GPU got nothing, the other
three ~78 GiB each, and the first step that needed memory ran out of it
(probe 2690543, teacher eval 2690720). For a model with such a parameter this
module writes the map itself: layer i on GPU i * n / L, everything outside the
layers on the first GPU but the output head on the last, and the parameters
transformers keeps off the GPUs left out of the map, as it does. The layer
that holds one of those is mapped tensor by tensor, as transformers does.
Any other model keeps ``"auto"``.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

_LAYER = re.compile(r"^(.*?\blayers\.(\d+))\.")


def spread_device_map(tensor_names: Iterable[str], n_gpus: int,
                      keep_off: Iterable[str] = ()) -> dict[str, int]:
    """A device map placing the decoder layers evenly over ``n_gpus``.

    ``tensor_names`` are a model's parameters and buffers; those ending with
    one of ``keep_off`` get no entry (transformers leaves them on the CPU),
    and the layer holding one is mapped by tensor, so that nothing else of it
    goes where they are.
    """
    names = list(tensor_names)
    keep_off = tuple(keep_off)
    layers = [int(m.group(2)) for n in names if (m := _LAYER.match(n))]
    n_layers = max(layers) + 1 if layers else 1
    split = {m.group(1) for n in names if n.endswith(keep_off) and (m := _LAYER.match(n))}
    # The modules that hold the layers ("model", "model.language_model", ...):
    # everything else is mapped as the largest module outside them, as "auto"
    # does, so accelerate moves each module's inputs to its GPU. Mapped tensor
    # by tensor, the modules got no such hook: "tensors on two devices, cuda:3
    # and cuda:0" (probe 2698993).
    roots = {m.group(1) for n in names if (m := _LAYER.match(n))}
    holders = {".".join(r.split(".")[:k]) for r in roots for k in range(1, r.count(".") + 1)}
    device_map: dict[str, int] = {}
    for name in names:
        if keep_off and name.endswith(keep_off):
            continue
        m = _LAYER.match(name)
        if m:
            gpu = min(n_gpus - 1, int(m.group(2)) * n_gpus // n_layers)
            device_map[name if m.group(1) in split else m.group(1)] = gpu
            continue
        parts = name.split(".")
        key = next((".".join(parts[:k]) for k in range(1, len(parts) + 1)
                    if ".".join(parts[:k]) not in holders), name)
        device_map[key] = n_gpus - 1 if key.startswith("lm_head") else 0
    return device_map


def device_map_for(path: str, n_gpus: int, model_cls) -> dict[str, int] | str:
    """``"auto"``, or for a model with parameters kept off the GPUs, `spread_device_map`."""
    try:
        from accelerate import init_empty_weights
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(path)
        with init_empty_weights():
            meta = model_cls.from_config(config)
        keep_off = getattr(meta, "_no_placement_params", None)
        if not keep_off:
            return "auto"
        names = [n for n, _ in meta.named_parameters()] + [n for n, _ in meta.named_buffers()]
        return spread_device_map(names, n_gpus, keep_off)
    except Exception:          # noqa: BLE001 - an unknown model loads as before
        return "auto"
