# SPDX-License-Identifier: Apache-2.0
"""Class-level output hook + per-instance tag."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from . import project

_MARK = "_omlx_steer_hooked"
_TAG_ATTR = "_omlx_steer"


class SteerTag:
    """Plain object attached with ``object.__setattr__`` (never in the tree)."""

    __slots__ = (
        "state", "kind", "layer_idx", "v", "target",
        "output_index", "residual_included",
    )

    def __init__(
        self,
        state,
        kind: str,
        layer_idx: int,
        v,
        target=None,
        *,
        output_index: int | None = None,
        residual_included: bool = False,
    ):
        self.state = state
        self.kind = kind
        self.layer_idx = layer_idx
        self.v = v
        self.target = target
        self.output_index = output_index
        self.residual_included = residual_included

    def apply(self, out: Any, kwargs: dict) -> Any:
        st = self.state
        if st.capture is not None:
            return self._capture(out, kwargs)
        s = st.scale_for(self.kind)
        if s is None:
            return out
        if st.debug:
            st.hit(self.kind, self.layer_idx)
        return self._steer(out, kwargs, s)

    def steer_tensor(self, y: Any, s: Any, residual: Any = None) -> Any:
        if self.residual_included and residual is not None:
            delta = project.apply(
                y - residual, self.v, s, method=self.state.method,
                norm_preserve=self.state.norm_preserve, target=self.target,
            )
            return residual + delta
        return project.apply(
            y, self.v, s, method=self.state.method,
            norm_preserve=self.state.norm_preserve, target=self.target,
        )

    def _steer(self, out, kwargs, s):
        res = kwargs.get("residual") if self.residual_included else None
        if self.output_index is None:
            return self.steer_tensor(out, s, res)
        lst = list(out)
        lst[self.output_index] = self.steer_tensor(lst[self.output_index], s, res)
        return type(out)(lst) if isinstance(out, tuple) else lst

    def _capture(self, out, kwargs):
        cap = self.state.capture
        if cap.kind == self.kind:
            y = out if self.output_index is None else out[self.output_index]
            if self.residual_included and kwargs.get("residual") is not None:
                y = y - kwargs["residual"]
            cap.record(self.layer_idx, y)
        return out


def get_tag(module: Any) -> SteerTag | None:
    return getattr(module, "__dict__", {}).get(_TAG_ATTR)


def set_tag(module: Any, tag: SteerTag | None) -> None:
    # nn.Module.__setattr__ would register arrays/dicts as tree children.
    object.__setattr__(module, _TAG_ATTR, tag)


def hook_class(cls: type) -> bool:
    """Wrap ``cls.__call__`` once. Idempotent; returns True if newly hooked."""
    orig = cls.__call__
    if getattr(orig, _MARK, False):
        return False

    def __call__(self, *args, **kwargs):  # noqa: N807 - installed as a class method
        out = orig(self, *args, **kwargs)
        tag = self.__dict__.get(_TAG_ATTR)
        return out if tag is None else tag.apply(out, kwargs)

    setattr(__call__, _MARK, True)
    __call__.__wrapped__ = orig
    cls.__call__ = __call__
    return True


def steer_tensor(module: Any, y: Any, residual: Any = None) -> Any:
    """For fused paths that bypass the module call."""
    tag = get_tag(module)
    if tag is None:
        return y
    st = tag.state
    if st.capture is not None:
        # Capture runs unsteered: record instead of project.
        if st.capture.kind == tag.kind:
            cap_y = y
            if tag.residual_included and residual is not None:
                cap_y = y - residual
            st.capture.record(tag.layer_idx, cap_y)
        return y
    s = st.scale_for(tag.kind)
    if s is None:
        return y
    if st.debug:
        st.hit(tag.kind, tag.layer_idx)
    return tag.steer_tensor(y, s, residual)


def hook_function(
    owner: Any, name: str, *, module_arg: int = 0, resolve: Callable | None = None
) -> bool:
    """Wrap a module-level function / method that takes the submodule as an arg.

    ``resolve`` maps the argument actually passed to the hooked submodule —
    e.g. the stock verifier passes the decoder layer where the fused path
    passes ``layer.linear_attn`` directly.
    """
    orig = getattr(owner, name)
    if getattr(orig, _MARK, False):
        return False

    def wrapper(*args, **kwargs):
        out = orig(*args, **kwargs)
        mod = args[module_arg]
        return steer_tensor(resolve(mod) if resolve is not None else mod, out)

    setattr(wrapper, _MARK, True)
    wrapper.__wrapped__ = orig
    setattr(owner, name, wrapper)
    return True
