# SPDX-License-Identifier: Apache-2.0
"""Capture-time direction building on the live engine (plan §13).

The vector must come from the same hook, same patches and same chat template
as inference (C3), so the running engine captures activations through its own
steering tags instead of a re-implemented load path. Capture arms
``state.capture``; the hooks then record the last prompt token's branch
output per layer instead of projecting (unsteered activations, §13.2).
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import mlx.core as mx
import numpy as np


class CaptureBuffer:
    """Per-layer last-token branch outputs of one forward.

    The hooks call ``record`` once per steered module call; capture runs one
    full-prompt forward without chunking, so each layer records exactly once.
    Vectors stay lazy until the driver evals them on the capturing thread.
    """

    def __init__(self, kind: str):
        self.kind = kind
        self.vectors: dict[int, Any] = {}

    def record(self, layer_idx: int, y: Any) -> None:
        # Last token of the branch output: [B, L, E], [L, E] or [B, 1, E].
        rows = y.reshape(-1, y.shape[-1])
        self.vectors[layer_idx] = rows[rows.shape[0] - 1]


def capture_one(
    fwd: Any,
    ids: list[int],
    state: Any,
    kind: str,
    cache_factory: Callable[[], Any],
) -> dict[int, np.ndarray]:
    """One full-prompt forward with capture armed; ``{layer: (E,) float64}``.

    Runs on the engine's MLX executor (stream-bound arrays), one prompt per
    task so serving traffic interleaves (§13.1). Capture mode makes the hooks
    record instead of project, so the activations are the unsteered ones even
    when the scales are non-zero.
    """
    buf = CaptureBuffer(kind)
    state.capture = buf
    try:
        out = fwd(mx.array([ids]), cache=cache_factory())
        logits = out.logits if hasattr(out, "logits") else out
        # Cast to f32 before numpy: activations are usually bfloat16, which
        # has no PEP 3118 buffer format and cannot cross into numpy directly.
        items = sorted(buf.vectors.items())
        cast = [v.astype(mx.float32) for _, v in items]
        mx.eval(logits, *cast)
    finally:
        state.capture = None
    if len(buf.vectors) != state.n_layers:
        raise RuntimeError(
            f"capture recorded {len(buf.vectors)}/{state.n_layers} layers "
            f"for component {kind!r} — silent bypass during capture"
        )
    return {i: np.asarray(v, np.float64) for (i, _), v in zip(items, cast)}


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v


def _layer_direction(
    good: np.ndarray, bad: np.ndarray, *, pair_normalize: bool, orthogonalize: bool
) -> np.ndarray:
    """One layer's direction from (n, E) capture stacks (§13.3)."""
    if pair_normalize:
        diffs = good - bad
        d = _unit(diffs.sum(axis=0) / max(len(diffs), 1))
    else:
        d = _unit(good.mean(axis=0) - bad.mean(axis=0))
    if orthogonalize:
        b = _unit(bad.mean(axis=0))
        d = _unit(d - float(d @ b) * b)
    return d


def build_direction(
    good: list[dict[int, np.ndarray]],
    bad: list[dict[int, np.ndarray]],
    *,
    pair_normalize: bool = False,
    orthogonalize: bool = True,
    seed: int = 0,
) -> dict[str, Any]:
    """DS4's build_direction maths (plan §13.3), float64 throughout.

    Pairs ``good[i]`` with ``bad[i]`` over ``n = min(len(good), len(bad))``
    prompts. Returns unit-norm ``vectors`` (L, E) f32, the projection means
    ``mu_good`` / ``mu_bad`` (L,) f32 for retarget mode (§2e Q3), and the
    split-half stability cosine per layer (§2e Q2; ``None`` entries when the
    pairs are too few to split).
    """
    if not good or not bad:
        raise ValueError("capture needs at least one good and one bad prompt")
    n = min(len(good), len(bad))
    good, bad = good[:n], bad[:n]
    layers = sorted(good[0])
    for caps in (*good, *bad):
        if sorted(caps) != layers:
            raise ValueError("captures disagree on the recorded layer set")
    if n < 2:
        raise ValueError("capture needs at least 2 pairs for split-half stability")

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    half_a, half_b = perm[: n // 2], perm[n // 2 : n // 2 * 2]

    vectors = np.zeros((len(layers), good[0][layers[0]].shape[0]), np.float64)
    mu_good = np.zeros(len(layers), np.float64)
    mu_bad = np.zeros(len(layers), np.float64)
    split_half_cos: list[Optional[float]] = []
    for row, layer in enumerate(layers):
        g = np.stack([c[layer] for c in good])
        b = np.stack([c[layer] for c in bad])
        d = _layer_direction(
            g, b, pair_normalize=pair_normalize, orthogonalize=orthogonalize
        )
        vectors[row] = d
        mu_good[row] = float(np.mean(g @ d))
        mu_bad[row] = float(np.mean(b @ d))
        if len(half_a) >= 1 and len(half_b) >= 1:
            da = _layer_direction(
                g[half_a], b[half_a],
                pair_normalize=pair_normalize, orthogonalize=orthogonalize,
            )
            db = _layer_direction(
                g[half_b], b[half_b],
                pair_normalize=pair_normalize, orthogonalize=orthogonalize,
            )
            denom = float(np.linalg.norm(da) * np.linalg.norm(db))
            split_half_cos.append(float(da @ db) / denom if denom > 1e-12 else None)
        else:
            split_half_cos.append(None)

    return {
        "vectors": vectors.astype(np.float32),
        "mu_good": mu_good.astype(np.float32),
        "mu_bad": mu_bad.astype(np.float32),
        "split_half_cos": split_half_cos,
        "direction_method": "pair_normalize" if pair_normalize else "mean_diff",
        "n_pairs": n,
    }
