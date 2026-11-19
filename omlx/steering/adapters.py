# SPDX-License-Identifier: Apache-2.0
"""Per-architecture hook targets (plan §6). Main layer list only, never MTP."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional


class SteeringUnsupported(RuntimeError):
    """Architecture has no adapter; never fall back silently."""


@dataclass(frozen=True)
class Target:
    kind: str  # "attn" | "ffn"
    layer_idx: int
    module: Any
    output_index: Optional[int] = None
    residual_included: bool = False


@dataclass
class SteeringAdapter:
    name: str
    version: int
    components: frozenset
    find_targets: Callable[[Any], list]
    bypass_handlers: tuple = ()
    mode: str = "block"

    def on_scales_changed(self, model: Any) -> None:
        pass


def model_type_of(model: Any) -> str:
    for obj in (model, getattr(model, "language_model", None)):
        if obj is None:
            continue
        for src in (getattr(obj, "args", None), getattr(obj, "config", None), obj):
            mt = getattr(src, "model_type", None)
            if isinstance(mt, str) and mt:
                return mt
    return ""


def find_main_layers(model: Any) -> list:
    """Main decoder layers only (never ``model.mtp`` / ``language_model.mtp``)."""
    for path in (
        ("model", "layers"),
        ("language_model", "model", "layers"),
        ("_vlm_model", "language_model", "model", "layers"),
        ("layers",),
    ):
        obj = model
        for p in path:
            obj = getattr(obj, p, None)
            if obj is None:
                break
        else:
            return list(obj)
    raise SteeringUnsupported("cannot locate decoder layers")


def _attr(layer: Any, *names: str) -> Any:
    for n in names:
        m = getattr(layer, n, None)
        if m is not None and callable(m):
            return m
    return None


def _generic_targets(model: Any, attn_names, ffn_names) -> list:
    out = []
    for i, layer in enumerate(find_main_layers(model)):
        a = _attr(layer, *attn_names)
        f = _attr(layer, *ffn_names)
        if a is None or f is None:
            raise SteeringUnsupported(f"layer {i}: missing attention or ffn module")
        out.append(Target("attn", i, a))
        out.append(Target("ffn", i, f))
    return out


_COMPONENTS = frozenset({"attn", "ffn"})

ADAPTERS: dict[str, SteeringAdapter] = {}


def _register(adapter: SteeringAdapter, *model_types: str) -> None:
    for mt in model_types:
        ADAPTERS[mt] = adapter


_register(
    SteeringAdapter(
        "generic", 1, _COMPONENTS,
        lambda m: _generic_targets(m, ("self_attn",), ("mlp",)),
    ),
    "llama", "qwen2", "qwen3", "qwen3_moe", "mistral", "mistral3", "ministral",
)
# Hybrid GDN (linear_attn) / full attention layers, Qwen3.6-35B-A3B included.
_register(
    SteeringAdapter(
        "qwen3_next", 1, _COMPONENTS,
        lambda m: _generic_targets(m, ("linear_attn", "self_attn"), ("mlp",)),
    ),
    "qwen3_next", "qwen3_5", "qwen3_5_moe", "qwen3_5_text", "qwen3_5_moe_text",
)
# Qwen3.8 Flash Next: hyper-connections with deferred residual writes. The
# branch modules (self_attn / linear_attn / mlp) are still called normally —
# hc_fused fuses the combine that runs AFTER the branch output — so the same
# targets work. The hook point is the branch before the HC combine (DS4's
# Qwen attention hook), a different vector semantics than a plain residual
# add, hence the distinct pack mode: qwen3_next packs are refused here.
_register(
    SteeringAdapter(
        "qwen4_exp", 1, _COMPONENTS,
        lambda m: _generic_targets(m, ("linear_attn", "self_attn"), ("mlp",)),
        mode="branch",
    ),
    "qwen4_exp",
)


def get_adapter(model: Any) -> SteeringAdapter:
    mt = model_type_of(model)
    ad = ADAPTERS.get(mt)
    if ad is None:
        raise SteeringUnsupported(f"steering: unsupported model_type {mt!r}")
    return ad
