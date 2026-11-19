# SPDX-License-Identifier: Apache-2.0
"""Directional steering numerics + coverage on a tiny real model (macOS only)."""

import sys

import numpy as np
import pytest

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(sys.platform != "darwin", reason="needs real MLX"),
]

import mlx.core as mx  # noqa: E402

from omlx.steering import install_steering  # noqa: E402
from omlx.steering import project  # noqa: E402
from omlx.steering.install import get_state  # noqa: E402
from omlx.steering.pack import SteeringPack, save_pack  # noqa: E402


def _y(dtype=mx.float32):
    return mx.random.normal((2, 3, 8)).astype(dtype)


def _v():
    v = mx.random.normal((8,))
    return v / mx.sqrt(mx.sum(v * v))


@pytest.mark.parametrize("dtype", [mx.float32, mx.float16, mx.bfloat16])
def test_project_identities(dtype):
    y, v = _y(dtype), _v()
    s0 = project.apply(y, v, mx.array(0.0))
    assert mx.array_equal(s0, y)  # s=0 exact no-op
    s1 = project.apply(y, v, mx.array(1.0))
    assert s1.dtype == y.dtype
    resid = (s1.astype(mx.float32) @ v)
    assert float(mx.max(mx.abs(resid))) < (5e-2 if dtype != mx.float32 else 1e-5)
    s2 = project.apply(y, v, mx.array(-1.0)).astype(mx.float32) @ v
    assert np.allclose(np.array(s2), 2 * np.array(y.astype(mx.float32) @ v), atol=5e-2)


def test_per_row_scale_and_norm_preserve():
    y, v = _y(), _v()
    s = mx.array([[1.0], [0.0]])  # (B,1): row 1 untouched
    out = project.apply(y, v, s)
    assert mx.array_equal(out[1], y[1]) and not mx.array_equal(out[0], y[0])
    n = project.apply(y, v, mx.array(1.0), norm_preserve=True)
    assert np.allclose(
        np.array(mx.sum(n * n, -1)), np.array(mx.sum(y * y, -1)), rtol=1e-4
    )


def test_retarget_moves_toward_target():
    y, v = _y(), _v()
    t = mx.array(0.7)
    out = project.apply(y, v, mx.array(1.0), method="retarget", target=t)
    assert np.allclose(np.array(out @ v), 0.7, atol=1e-4)  # lambda=1: ablate to t
    half = project.apply(y, v, mx.array(0.5), method="retarget", target=t)
    d0, d1 = np.array(y @ v), np.array(half @ v)
    assert np.allclose(d1, d0 + 0.5 * (0.7 - d0), atol=1e-4)


def _tiny_llama():
    from mlx_lm.models.llama import Model, ModelArgs

    m = Model(ModelArgs(
        model_type="llama", hidden_size=32, num_hidden_layers=3,
        intermediate_size=64, num_attention_heads=4, num_key_value_heads=2,
        rms_norm_eps=1e-5, vocab_size=128,
    ))
    mx.eval(m.parameters())
    return m


def _settings(pack, **kw):
    from omlx.model_settings import ModelSettings

    return ModelSettings(steering_pack=str(pack), **kw)


def _pack(tmp_path, comp="ffn"):
    v = np.random.default_rng(1).standard_normal((3, 32)).astype(np.float32)
    save_pack(SteeringPack("t", v, {"hook": {"component": comp, "mode": "block"}}),
              tmp_path / "t")
    return tmp_path / "t"


def test_tiny_llama_effect_identity_params_and_coverage(tmp_path, monkeypatch):
    monkeypatch.setenv("OMLX_STEERING_DEBUG", "1")
    m = _tiny_llama()
    ids = mx.array([[1, 5, 9, 12]])
    base = m(ids)
    names_before = sorted(k for k, _ in __import__("mlx.utils").utils.tree_flatten(m.parameters()))

    st = install_steering(m, _settings(_pack(tmp_path), steering_ffn_scale=0.0))
    assert st is get_state(m)
    names_after = sorted(k for k, _ in __import__("mlx.utils").utils.tree_flatten(m.parameters()))
    assert names_before == names_after  # tag is not in the parameter tree
    assert mx.array_equal(m(ids), base)  # scale 0: bit-identical

    st.set(ffn=1.0, scope="all")
    st.begin_prefill()
    steered = m(ids)
    assert not np.allclose(np.array(steered), np.array(base))
    assert {i for (k, i) in st.hits if k == "ffn"} == {0, 1, 2}  # every layer hit

    # prefill scope: decode steps after the prompt token are not steered.
    st.set(scope="prefill")
    st.hits.clear()
    st.begin_decode_step([5])
    m(ids[:, :1])
    assert not st.hits
    st.begin_decode_step([0])  # last prompt token counts as prefill
    m(ids[:, :1])
    assert st.hits


def test_cache_still_works_with_steering(tmp_path):
    from mlx_lm.models.cache import make_prompt_cache

    m = _tiny_llama()
    install_steering(m, _settings(_pack(tmp_path), steering_ffn_scale=1.0))
    out = m(mx.array([[1, 2, 3]]), cache=make_prompt_cache(m))
    mx.eval(out)
    assert out.shape[-1] == 128
