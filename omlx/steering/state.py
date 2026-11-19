# SPDX-License-Identifier: Apache-2.0
"""Live steering state attached to a loaded model (plan §5.2, §2c, §10)."""

from __future__ import annotations

import hashlib
from typing import Any, Optional, Sequence

import mlx.core as mx
import numpy as np

SCOPES = ("all", "prefill", "prefill+k")
METHODS = ("project", "retarget")
DEFAULT_SCOPE = "prefill+k"
DEFAULT_K = 32


class SteeringState:
    """One active (pack, attn scale, ffn scale, scope) per loaded engine (C8).

    Mutated only on the engine's MLX thread (scheduler drain), never mid-forward.
    """

    def __init__(
        self,
        *,
        pack_sha: str,
        bank: Any,
        n_layers: int,
        components: frozenset[str],
        attn_scale: float = 0.0,
        ffn_scale: float = 0.0,
        scope: str = DEFAULT_SCOPE,
        decode_tokens: int = DEFAULT_K,
        method: str = "project",
        norm_preserve: bool = False,
        mu_good: Optional[np.ndarray] = None,
        mu_bad: Optional[np.ndarray] = None,
        layer_mask: Optional[frozenset[int]] = None,
        retarget_to: str = "bad",
    ):
        self.pack_sha = pack_sha
        self.bank = bank  # mx.array [L, E] f32
        self.n_layers = n_layers
        self.components = components
        self.mu_good = mu_good
        self.mu_bad = mu_bad
        self.layer_mask = layer_mask
        self.retarget_to = retarget_to
        self.epoch = 0
        self.hits: dict[tuple[str, int], int] = {}
        self.debug = False
        self.capture: Any = None
        # Phase signal, set by scheduler entrypoints (never inferred from L).
        self.phase = "prefill"
        # Decode-step gate for prefill / prefill+k (R5: hook returns at once).
        self.step_active = True
        self.row_mask: Any = None  # (B, 1) f32, or None = all rows steered
        self._uid_steps: dict[Any, int] = {}
        self.set(
            attn=attn_scale,
            ffn=ffn_scale,
            scope=scope,
            decode_tokens=decode_tokens,
            method=method,
            norm_preserve=norm_preserve,
            bump=False,
        )

    # -- configuration ------------------------------------------------------
    def set(
        self,
        *,
        attn: Optional[float] = None,
        ffn: Optional[float] = None,
        scope: Optional[str] = None,
        decode_tokens: Optional[int] = None,
        method: Optional[str] = None,
        norm_preserve: Optional[bool] = None,
        bump: bool = True,
    ) -> None:
        if attn is not None:
            self.attn_scale = float(attn)
            self.attn_s = mx.array(float(attn), dtype=mx.float32)
        if ffn is not None:
            self.ffn_scale = float(ffn)
            self.ffn_s = mx.array(float(ffn), dtype=mx.float32)
        if scope is not None:
            if scope not in SCOPES:
                raise ValueError(f"steering scope must be one of {SCOPES}")
            self.scope = scope
        if decode_tokens is not None:
            self.decode_tokens = max(1, int(decode_tokens))
        if method is not None:
            if method not in METHODS:
                raise ValueError(f"steering method must be one of {METHODS}")
            self.method = method
        if norm_preserve is not None:
            self.norm_preserve = bool(norm_preserve)
        if bump:
            self.epoch += 1

    def scale(self, kind: str) -> float:
        return self.attn_scale if kind == "attn" else self.ffn_scale

    def active(self) -> bool:
        """Any non-zero scale on an installed component (C7)."""
        return (self.attn_scale != 0.0 and "attn" in self.components) or (
            self.ffn_scale != 0.0 and "ffn" in self.components
        )

    def salt(self) -> str:
        """Cache namespace salt (plan §11); '' when steering is off."""
        if not self.active():
            return ""
        k = self.decode_tokens if self.scope == "prefill+k" else ""
        layers = (
            ",".join(map(str, sorted(self.layer_mask))) if self.layer_mask else "*"
        )
        raw = (
            f"{self.pack_sha[:16]}:a{self.attn_scale!r}:f{self.ffn_scale!r}:"
            f"{self.scope}:{k}:{self.method}:{int(self.norm_preserve)}:{layers}"
        )
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    # -- scope / phase ------------------------------------------------------
    def decode_row_limit(self) -> Optional[int]:
        """Rows with ``n_generated < limit`` are steered in a decode step.

        The first decode step still processes the last *prompt* token (n=0),
        which counts as prefill; generated token j is the input of step n=j.
        """
        if self.scope == "all":
            return None
        if self.scope == "prefill":
            return 1
        return self.decode_tokens + 1

    def begin_decode_step(self, generated: Sequence[int]) -> None:
        """Called by the scheduler before each decode forward (per-row K)."""
        self.phase = "decode"
        limit = self.decode_row_limit()
        if limit is None:
            self.step_active, self.row_mask = True, None
            return
        flags = [1.0 if n < limit else 0.0 for n in generated]
        if not any(flags):
            self.step_active, self.row_mask = False, None
        elif all(flags):
            self.step_active, self.row_mask = True, None
        else:
            self.step_active = True
            self.row_mask = mx.array(np.asarray(flags, np.float32).reshape(-1, 1))

    def begin_decode_uids(self, uids: Sequence[Any]) -> None:
        """Per-uid forward-step counter -> per-row K (aligned to batch rows).

        Counter is keyed by uid and rebuilt from the live rows each step, so
        batch insert / filter / reorder can never misalign it (pitfall 16).
        Under MTP one step may commit several tokens; K counts steps.
        """
        prev = self._uid_steps
        counts = [prev.get(u, 0) for u in uids]
        self._uid_steps = {u: c + 1 for u, c in zip(uids, counts)}
        self.begin_decode_step(counts)

    def begin_prefill(self) -> None:
        self.phase = "prefill"
        self.step_active, self.row_mask = True, None

    def scale_for(self, kind: str) -> Any:
        """Scale array for this forward, or None when the hook must no-op."""
        if self.scale(kind) == 0.0 or kind not in self.components:
            return None
        s = self.attn_s if kind == "attn" else self.ffn_s
        if self.phase == "prefill":
            return s
        if not self.step_active:
            return None
        return s if self.row_mask is None else s * self.row_mask

    def hit(self, kind: str, layer: int) -> None:
        k = (kind, layer)
        self.hits[k] = self.hits.get(k, 0) + 1
