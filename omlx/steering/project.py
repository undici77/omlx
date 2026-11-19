# SPDX-License-Identifier: Apache-2.0
"""Projection ops (plan §5.1). Compiled MLX ops, no custom kernel in v1.

Every scale is an *input array*, never captured, so changing it never
recompiles. All math runs in fp32 and is cast back to the input dtype.
"""

from __future__ import annotations

from functools import partial

import mlx.core as mx


def _compile(fn):
    try:
        return partial(mx.compile, shapeless=True)(fn)
    except Exception:  # mock / very old mlx: fall back to eager
        return fn


def _project(y, v, s):
    """DS4: y - s * v * (v . y). s: () or (B, 1)."""
    yf = y.astype(mx.float32)
    c = (yf @ v) * s
    return (yf - c[..., None] * v).astype(y.dtype)


def _project_np(y, v, s):
    """project + norm preservation: ||y'||^2 = ||y||^2 + (s^2 - 2s)(v.y)^2."""
    yf = y.astype(mx.float32)
    d = yf @ v
    out = yf - (d * s)[..., None] * v
    n0 = mx.sum(yf * yf, axis=-1)
    n1 = mx.maximum(n0 + (s * s - 2.0 * s) * d * d, 1e-12)
    return (out * mx.sqrt(n0 / n1)[..., None]).astype(y.dtype)


def _retarget(y, v, lam, t):
    """y + lam * v * (t - v . y). t: () per-layer target projection mean."""
    yf = y.astype(mx.float32)
    d = yf @ v
    return (yf + (lam * (t - d))[..., None] * v).astype(y.dtype)


def _retarget_np(y, v, lam, t):
    yf = y.astype(mx.float32)
    d = yf @ v
    d2 = d + lam * (t - d)
    out = yf + (d2 - d)[..., None] * v
    n0 = mx.sum(yf * yf, axis=-1)
    # v is unit: replacing component d by d2 changes ||y||^2 by d2^2 - d^2.
    n1 = mx.maximum(n0 + d2 * d2 - d * d, 1e-12)
    return (out * mx.sqrt(n0 / n1)[..., None]).astype(y.dtype)


_project_c = _compile(_project)
_project_np_c = _compile(_project_np)
_retarget_c = _compile(_retarget)
_retarget_np_c = _compile(_retarget_np)


def apply(y, v, s, *, method="project", norm_preserve=False, target=None):
    """Single entry point so a Metal kernel (backlog O1) can replace it."""
    if method == "retarget":
        t = target if target is not None else mx.array(0.0, dtype=mx.float32)
        return (_retarget_np_c if norm_preserve else _retarget_c)(y, v, s, t)
    return (_project_np_c if norm_preserve else _project_c)(y, v, s)
