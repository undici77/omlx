# SPDX-License-Identifier: Apache-2.0
"""install_steering: tag instances, hook classes, attach state."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

from .adapters import SteeringUnsupportedError, find_main_layers, get_adapter
from .hooks import SteerTag, hook_class, hook_function, set_tag
from .pack import SteeringPackError, resolve_pack_ref
from .state import DEFAULT_K, DEFAULT_SCOPE, SteeringState

logger = logging.getLogger(__name__)

_STATE_ATTR = "_omlx_steering"


def get_state(model: Any) -> SteeringState | None:
    """Find the state on a model or its VLM / language-model wrappers."""
    for obj in (
        model,
        getattr(model, "_vlm_model", None),
        getattr(model, "language_model", None),
    ):
        if obj is None:
            continue
        st = getattr(obj, "__dict__", {}).get(_STATE_ATTR)
        if st is not None:
            return st
    return None


def steering_active(model: Any) -> bool:
    st = get_state(model)
    return st is not None and st.active()


def _pick(setting: Any, manifest_val: Any, default: Any) -> Any:
    if setting is not None:
        return setting
    return manifest_val if manifest_val is not None else default


def install_steering(
    model: Any,
    settings: Any,
    *,
    model_path: str | None = None,
    base_path: Path | None = None,
) -> SteeringState | None:
    """Install steering on a loaded model. Returns None when no pack is set.

    Must run after every other class patch so our ``__call__`` is outermost.
    """
    ref = getattr(settings, "steering_pack", None) if settings is not None else None
    if not ref:
        return None  # R1: nothing touched

    adapter = get_adapter(model)
    layers = find_main_layers(model)
    n_layers = len(layers)

    pack = resolve_pack_ref(ref, base_path=base_path, shape=None)
    if pack.n_layers != n_layers:
        raise SteeringPackError(
            f"steering pack {pack.name}: {pack.n_layers} layers, model has {n_layers}"
        )
    hook = pack.manifest.get("hook") or {}
    if hook.get("adapter") and hook["adapter"] != adapter.name:
        raise SteeringPackError(
            f"steering pack built for adapter {hook['adapter']!r}, model uses {adapter.name!r}"
        )
    if hook.get("mode", adapter.mode) != adapter.mode:
        raise SteeringPackError("steering pack hook mode mismatch")

    usage = pack.usage
    rec = usage.get("recommended") or {}
    comp = pack.component
    # R2: hook only the components the pack was built for.
    components = frozenset({comp}) & adapter.components
    if not components:
        raise SteeringUnsupportedError(f"adapter {adapter.name} lacks component {comp!r}")

    g = lambda n: getattr(settings, n, None)  # noqa: E731
    ffn = _pick(g("steering_ffn_scale"), rec.get("ffn"), 1.0 if comp == "ffn" else 0.0)
    attn = _pick(g("steering_attn_scale"), rec.get("attn"), 1.0 if comp == "attn" else 0.0)

    steer = pack.steer
    mu_good = np.asarray(steer["mu_good"], np.float32) if steer.get("mu_good") else None
    mu_bad = np.asarray(steer["mu_bad"], np.float32) if steer.get("mu_bad") else None
    mask = steer.get("layers")
    state = SteeringState(
        pack_sha=pack.sha256,
        bank=mx.array(pack.vectors),
        n_layers=n_layers,
        components=components,
        attn_scale=attn,
        ffn_scale=ffn,
        scope=_pick(g("steering_scope"), usage.get("scope"), DEFAULT_SCOPE),
        decode_tokens=_pick(g("steering_decode_tokens"), usage.get("decode_tokens"), DEFAULT_K),
        method=_pick(g("steering_method"), usage.get("method"), "project"),
        norm_preserve=bool(_pick(g("steering_norm_preserve"), usage.get("norm_preserve"), False)),
        mu_good=mu_good,
        mu_bad=mu_bad,
        layer_mask=frozenset(mask) if mask else None,
    )
    import os

    state.debug = os.environ.get("OMLX_STEERING_DEBUG") == "1"

    # MLX streams are thread-local. Install runs on a different thread than the
    # engine's scheduler, so any lazy op built here (a bank slice) would stay
    # bound to this thread's stream and fail later with "no Stream(gpu, N) in
    # current thread". Materialize every array the hooks will read.
    rows = [state.bank[i] for i in range(n_layers)]
    mx.eval(state.bank, rows)

    hooked = set()
    for t in adapter.find_targets(model):
        if t.kind not in components:
            continue
        if state.layer_mask is not None and t.layer_idx not in state.layer_mask:
            continue
        target = None
        if state.method == "retarget" and mu_good is not None and mu_bad is not None:
            mu = mu_bad if state.retarget_to == "bad" else mu_good
            target = mx.array(float(mu[t.layer_idx]), dtype=mx.float32)
            mx.eval(target)
        tag = SteerTag(
            state, t.kind, t.layer_idx, rows[t.layer_idx], target,
            output_index=t.output_index, residual_included=t.residual_included,
        )
        set_tag(t.module, tag)
        cls = type(t.module)
        if cls not in hooked:
            hook_class(cls)
            hooked.add(cls)

    # Verifier handlers: fused paths that bypass the module call entirely.
    _install_verify_injections(components)

    object.__setattr__(model, _STATE_ATTR, state)
    logger.info(
        "steering: pack=%s adapter=%s layers=%d attn=%s ffn=%s scope=%s",
        pack.name, adapter.name, n_layers, attn, ffn, state.scope,
    )
    return state


def _install_verify_injections(components: frozenset) -> None:
    """mlx-vlm's exact verifier computes attention, GDN and FFN
    without calling the submodules, so spec-verify decode would silently
    bypass the hooks. Wrap the verifier methods — after oMLX's own overrides,
    which replace these same methods — so each returned branch output is
    steered through the submodule's tag; untagged modules pass through.
    oMLX's patched ``_layer`` consumes the returned values, so this injects
    at the explicit intermediate tensor instead of gating verify off.
    """
    try:
        from mlx_vlm.models.qwen3_5.speculative_verifier import (
            Qwen3_5BatchInvariantForward,
        )
    except ImportError:
        return  # mlx-vlm not installed (text-only deployment): never used
    if "ffn" in components:
        # (self, feed_forward, x) — the MoE / dense MLP block itself.
        hook_function(Qwen3_5BatchInvariantForward, "_feed_forward", module_arg=1)
    if "attn" in components:
        # (self, self_attn, normed, mask, cache, ...) — full-attention layers.
        hook_function(Qwen3_5BatchInvariantForward, "_attention", module_arg=1)
        # (self, layer_or_linear_attn, ...) — the stock verifier passes the
        # decoder layer, oMLX's armed verify passes layer.linear_attn.
        hook_function(
            Qwen3_5BatchInvariantForward,
            "_gated_delta",
            module_arg=1,
            resolve=lambda mod: getattr(mod, "linear_attn", mod),
        )


def mark_prefill(model: Any) -> None:
    """Scheduler entrypoint signal: the next forward is prefill."""
    st = get_state(model)
    if st is not None:
        st.begin_prefill()


def mark_decode(model: Any, uids: Any) -> None:
    """Scheduler entrypoint signal: the next forward is a decode step."""
    st = get_state(model)
    if st is not None:
        st.begin_decode_uids(list(uids))
