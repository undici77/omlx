# SPDX-License-Identifier: Apache-2.0
"""Directional steering: logic-only tests (pure Python; numerics are Mac-only)."""

import json

import numpy as np
import pytest

import mlx.core as mx

from omlx.cache.paged_cache import PagedCacheManager, compute_block_hash
from omlx.model_settings import ModelSettings, validate_steering
from omlx.steering import SteeringState, install_steering, load_pack, save_pack
from omlx.steering.hooks import (
    _MARK,
    SteerTag,
    get_tag,
    hook_class,
    hook_function,
    set_tag,
)
from omlx.steering.install import get_state, mark_decode, mark_prefill
from omlx.steering.pack import SteeringPack, SteeringPackError, resolve_pack_ref


def _vecs(L=2, E=4):
    v = np.zeros((L, E), np.float32)
    v[:, 0] = 1.0
    return v


def _state(**kw):
    base = dict(
        pack_sha="ab" * 32, bank=mx.array(_vecs()), n_layers=2,
        components=frozenset({"ffn"}), ffn_scale=1.0, scope="prefill+k",
        decode_tokens=2,
    )
    base.update(kw)
    return SteeringState(**base)


# ---- pack ------------------------------------------------------------------
def test_pack_roundtrip_and_tamper(tmp_path):
    save_pack(SteeringPack("p", _vecs(), {"hook": {"component": "ffn"}}), tmp_path / "p")
    p = load_pack(tmp_path / "p")
    assert p.n_layers == 2 and p.hidden_size == 4 and p.component == "ffn"
    f = tmp_path / "p" / "direction.f32"
    data = bytearray(f.read_bytes())
    data[0] ^= 0xFF
    f.write_bytes(bytes(data))
    with pytest.raises(SteeringPackError, match="sha256"):
        load_pack(tmp_path / "p")


def test_pack_size_and_format_refused(tmp_path):
    save_pack(SteeringPack("p", _vecs(), {}), tmp_path / "p")
    (tmp_path / "p" / "direction.f32").write_bytes(b"\0" * 8)
    with pytest.raises(SteeringPackError, match="size"):
        load_pack(tmp_path / "p")
    m = json.loads((tmp_path / "p" / "manifest.json").read_text())
    m["format"] = "bogus"
    (tmp_path / "p" / "manifest.json").write_text(json.dumps(m))
    with pytest.raises(SteeringPackError, match="format"):
        load_pack(tmp_path / "p")


def test_resolve_named_dir_and_bare_f32(tmp_path):
    save_pack(SteeringPack("named", _vecs(), {}), tmp_path / "steering" / "named")
    assert resolve_pack_ref("named", base_path=tmp_path).name == "named"
    raw = tmp_path / "x.f32"
    _vecs().tofile(raw)
    assert resolve_pack_ref(str(raw), base_path=tmp_path, shape=(2, 4)).component == "ffn"
    with pytest.raises(SteeringPackError):
        resolve_pack_ref("nope", base_path=tmp_path)


# ---- hooks -----------------------------------------------------------------
class _Mod:
    def __call__(self, x, **kw):
        return x


def test_hook_class_idempotent_and_untagged_passthrough():
    class M(_Mod):
        pass

    assert hook_class(M) is True
    assert hook_class(M) is False
    assert M()(5) == 5  # untagged: untouched


def test_tag_stored_outside_tree():
    class M(_Mod):
        pass

    m = M()
    set_tag(m, SteerTag(_state(), "ffn", 0, None))
    assert get_tag(m) is not None
    assert "_omlx_steer" in m.__dict__


def test_tag_tuple_output_index_and_scale_zero():
    st = _state(ffn_scale=0.0)
    tag = SteerTag(st, "ffn", 0, st.bank[0], output_index=0)
    out = ("a", "b")
    assert tag.apply(out, {}) is out  # scale 0 -> exact no-op, same object


# ---- state / scopes ---------------------------------------------------------
def test_salt_empty_when_off_and_changes_with_scope():
    assert _state(ffn_scale=0.0).salt() == ""
    a = _state(scope="all").salt()
    b = _state(scope="prefill").salt()
    c = _state(scope="prefill+k", decode_tokens=2).salt()
    d = _state(scope="prefill+k", decode_tokens=3).salt()
    assert len({a, b, c, d}) == 4 and "" not in {a, b, c, d}
    assert a == _state(scope="all").salt()  # deterministic


def test_scope_prefill_phase_always_active_decode_gated():
    st = _state(scope="prefill")
    st.begin_prefill()
    assert st.scale_for("ffn") is not None
    st.begin_decode_step([5, 7])  # past the last prompt token
    assert st.scale_for("ffn") is None and st.step_active is False
    st.begin_decode_step([0, 7])  # row 0 processes the last prompt token
    assert st.scale_for("ffn") is not None


def test_scope_prefill_k_per_row_mask():
    st = _state(scope="prefill+k", decode_tokens=2)
    st.begin_decode_step([0, 2, 3])  # limit = K+1 = 3
    assert st.step_active and st.row_mask is not None
    assert np.array(st.row_mask).reshape(-1).tolist() == [1.0, 1.0, 0.0]
    st.begin_decode_step([9, 9])
    assert st.step_active is False
    st.begin_decode_step([0, 1])
    assert st.step_active and st.row_mask is None


def test_scope_all_and_inactive_component():
    st = _state(scope="all")
    st.begin_decode_step([100])
    assert st.scale_for("ffn") is not None
    assert st.scale_for("attn") is None  # attn not installed (R2)


def test_uid_counter_realigns_after_filter():
    st = _state(scope="prefill+k", decode_tokens=1)  # limit 2
    st.begin_decode_uids(["a", "b"])
    st.begin_decode_uids(["a", "b"])
    st.begin_decode_uids(["b", "c"])  # a finished, c is new
    assert st._uid_steps == {"b": 3, "c": 1}
    assert np.array(st.row_mask).reshape(-1).tolist() == [0.0, 1.0]


def test_mark_helpers_noop_without_state():
    class Plain:
        pass

    mark_prefill(Plain())
    mark_decode(Plain(), ["u"])


# ---- install ---------------------------------------------------------------
def test_install_without_pack_touches_nothing():
    class Model:
        pass

    m = Model()
    assert install_steering(m, ModelSettings()) is None
    assert get_state(m) is None


# ---- §7 verify-path injections ---------------------------------------------
class _FakeAttn:
    def __call__(self, x, **kw):
        return x


class _FakeMLP:
    def compute(self, x):
        return x * 2.0

    def __call__(self, x, **kw):
        return self.compute(x)


class _FakeLayer:
    def __init__(self, full_attn: bool):
        if full_attn:
            self.self_attn = _FakeAttn()
        else:
            self.linear_attn = _FakeAttn()
        self.mlp = _FakeMLP()


class _FakeLM:
    def __init__(self, n: int):
        from types import SimpleNamespace

        self.model = type("Inner", (), {})()
        self.model.layers = [_FakeLayer(i % 2 == 0) for i in range(n)]
        self.args = SimpleNamespace(model_type="qwen3_next")


class _FakeModel:
    def __init__(self, n: int = 2):
        self.language_model = _FakeLM(n)


class _FakeVerifier:
    """Stand-in for mlx_vlm's Qwen3_5BatchInvariantForward.

    Like the real verifier, it inlines the block's math (``compute``) instead
    of calling the hooked ``__call__``.
    """

    def _feed_forward(self, feed_forward, x):
        return feed_forward.compute(x)

    def _attention(self, attn, x, *args):
        return attn(x)

    def _gated_delta(self, layer, x, *args):
        return layer.linear_attn(x)


def _fake_vlm_verifier_module(monkeypatch, verifier_cls) -> None:
    import sys
    import types

    mod = types.ModuleType("mlx_vlm.models.qwen3_5.speculative_verifier")
    mod.Qwen3_5BatchInvariantForward = verifier_cls
    monkeypatch.setitem(
        sys.modules, "mlx_vlm.models.qwen3_5.speculative_verifier", mod
    )


def _installed_fake_model(monkeypatch, tmp_path):
    monkeypatch.setenv("OMLX_STEERING_DEBUG", "1")
    _fake_vlm_verifier_module(monkeypatch, _FakeVerifier)
    save_pack(
        SteeringPack("t", _vecs(), {"hook": {"component": "ffn"}}),
        tmp_path / "steering" / "t",
    )
    model = _FakeModel(2)
    st = install_steering(
        model, ModelSettings(steering_pack="t"), base_path=tmp_path
    )
    return model, st


def test_install_wraps_verifier_feed_forward(monkeypatch, tmp_path):
    model, st = _installed_fake_model(monkeypatch, tmp_path)
    assert st is not None and st.n_layers == 2 and get_state(model) is st

    x = mx.array(np.array([[1.0, 2.0, 3.0, 4.0]], np.float32))
    mlp = model.language_model.model.layers[0].mlp
    st.hits.clear()
    out = np.array(_FakeVerifier()._feed_forward(mlp, x))
    # v = [1,0,0,0], s = 1: removes the first component of mlp(x) = 2x.
    assert out.tolist() == [[0.0, 4.0, 6.0, 8.0]]
    assert st.hits.get(("ffn", 0)) == 1

    # Untagged module: passthrough, no hit.
    st.hits.clear()
    plain = _FakeMLP()
    out = np.array(_FakeVerifier()._feed_forward(plain, x))
    assert out.tolist() == [[2.0, 4.0, 6.0, 8.0]]
    assert st.hits == {}


def test_install_leaves_attn_verifier_unwrapped_for_ffn_pack(monkeypatch, tmp_path):
    model, st = _installed_fake_model(monkeypatch, tmp_path)
    # R2: an ffn pack must not hook the attention verify paths.
    assert getattr(_FakeVerifier._attention, _MARK, False) is False
    assert getattr(_FakeVerifier._gated_delta, _MARK, False) is False
    assert getattr(_FakeVerifier._feed_forward, _MARK, False) is True


def test_reinstall_does_not_double_wrap(monkeypatch, tmp_path):
    model, _ = _installed_fake_model(monkeypatch, tmp_path)
    install_steering(model, ModelSettings(steering_pack="t"), base_path=tmp_path)
    st = get_state(model)
    x = mx.array(np.array([[1.0, 2.0, 3.0, 4.0]], np.float32))
    mlp = model.language_model.model.layers[1].mlp
    st.hits.clear()
    out = np.array(_FakeVerifier()._feed_forward(mlp, x))
    assert out.tolist() == [[0.0, 4.0, 6.0, 8.0]]  # steered exactly once
    assert st.hits.get(("ffn", 1)) == 1


def test_hook_function_resolve_maps_layer_to_module():
    from types import SimpleNamespace

    st = _state(
        components=frozenset({"attn"}), attn_scale=1.0, ffn_scale=0.0, scope="all"
    )
    st.debug = True
    layer = SimpleNamespace(linear_attn=_FakeAttn())
    set_tag(layer.linear_attn, SteerTag(st, "attn", 0, st.bank[0]))

    class Owner:
        def gdn(self, layer, x):
            return x

    assert hook_function(
        Owner, "gdn", module_arg=1, resolve=lambda m: getattr(m, "linear_attn", m)
    ) is True
    assert hook_function(Owner, "gdn", module_arg=1) is False  # idempotent

    x = mx.array(np.array([[1.0, 2.0, 3.0, 4.0]], np.float32))
    st.hits.clear()
    out = np.array(Owner().gdn(layer, x))
    assert out.tolist() == [[0.0, 2.0, 3.0, 4.0]]
    assert st.hits.get(("attn", 0)) == 1


# ---- qwen4_exp (Qwen3.8 Flash Next) ----------------------------------------
def test_qwen4_exp_adapter_finds_branch_targets():
    from omlx.steering.adapters import get_adapter

    model = _FakeModel(2)
    model.language_model.args.model_type = "qwen4_exp"
    ad = get_adapter(model)
    assert ad.name == "qwen4_exp" and ad.mode == "branch"
    targets = ad.find_targets(model)
    assert len(targets) == 4 and {t.kind for t in targets} == {"attn", "ffn"}


def test_install_refuses_block_mode_pack_on_branch_adapter(monkeypatch, tmp_path):
    _fake_vlm_verifier_module(monkeypatch, _FakeVerifier)
    save_pack(
        SteeringPack("t", _vecs(), {"hook": {"component": "ffn", "mode": "block"}}),
        tmp_path / "steering" / "t",
    )
    model = _FakeModel(2)
    model.language_model.args.model_type = "qwen4_exp"
    with pytest.raises(SteeringPackError, match="mode"):
        install_steering(
            model, ModelSettings(steering_pack="t"), base_path=tmp_path
        )


def test_install_accepts_modeless_pack_on_branch_adapter(monkeypatch, tmp_path):
    _fake_vlm_verifier_module(monkeypatch, _FakeVerifier)
    save_pack(
        SteeringPack("t", _vecs(), {"hook": {"component": "ffn"}}),
        tmp_path / "steering" / "t",
    )
    model = _FakeModel(2)
    model.language_model.args.model_type = "qwen4_exp"
    st = install_steering(
        model, ModelSettings(steering_pack="t"), base_path=tmp_path
    )
    assert st is not None and st.n_layers == 2


# ---- capture / build_direction (plan §13) ----------------------------------
def test_capture_buffer_records_last_token():
    from omlx.steering.capture import CaptureBuffer

    buf = CaptureBuffer("ffn")
    buf.record(0, mx.array(np.array([[1.0, 2.0], [3.0, 4.0]], np.float32)))
    assert np.array(buf.vectors[0]).tolist() == [3.0, 4.0]
    buf.record(1, mx.array(np.array([[[5.0, 6.0]]], np.float32)))
    assert np.array(buf.vectors[1]).tolist() == [5.0, 6.0]


def test_build_direction_recovers_axis_after_orthogonalization():
    from omlx.steering.capture import build_direction

    rng = np.random.default_rng(7)
    axis = np.zeros(8)
    axis[0] = 1.0
    bias = np.zeros(8)
    bias[1] = 1.0  # control mean: shared by good and bad
    good, bad = [], []
    for _ in range(6):
        good.append({0: rng.standard_normal(8) * 0.5 + 10.0 * axis + 5.0 * bias})
        bad.append({0: rng.standard_normal(8) * 0.5 + 5.0 * bias})
    out = build_direction(good, bad, orthogonalize=True, seed=1)
    v = out["vectors"][0].astype(np.float64)
    assert abs(float(v @ axis)) > 0.95  # aligned with the real difference...
    assert abs(float(v @ bias)) < 0.2  # ...with the control mean removed
    assert out["mu_good"][0] > out["mu_bad"][0]
    assert abs(float(np.linalg.norm(v)) - 1.0) < 1e-5
    # Above the plan's 0.8 layer-drop bar for this noise level.
    assert out["split_half_cos"][0] is not None and out["split_half_cos"][0] > 0.8
    assert out["direction_method"] == "mean_diff" and out["n_pairs"] == 6


def test_build_direction_pair_normalize_and_validation():
    from omlx.steering.capture import build_direction

    good = [{0: np.array([2.0, 0.0])}, {0: np.array([4.0, 0.0])}]
    bad = [{0: np.array([1.0, 0.0])}, {0: np.array([2.0, 0.0])}]
    # The diffs here lie along the control mean itself, so orthogonalization
    # would correctly annihilate them — test the pairing path without it.
    out = build_direction(good, bad, pair_normalize=True, orthogonalize=False)
    assert np.array(out["vectors"][0]).tolist() == [1.0, 0.0]
    assert out["direction_method"] == "pair_normalize"
    with pytest.raises(ValueError):
        build_direction(good[:1], bad[:1])  # too few pairs for split-half
    with pytest.raises(ValueError):
        build_direction([{1: np.zeros(2)}] * 2, bad)  # layer sets disagree


def test_steer_tensor_captures_instead_of_projecting():
    from omlx.steering.capture import CaptureBuffer
    from omlx.steering.hooks import steer_tensor

    st = _state(scope="all")
    st.debug = True
    buf = CaptureBuffer("ffn")
    st.capture = buf
    mod = _FakeMLP()
    set_tag(mod, SteerTag(st, "ffn", 1, st.bank[0]))
    x = mx.array(np.array([[1.0, 2.0, 3.0, 4.0]], np.float32))
    out = steer_tensor(mod, mod.compute(x))
    # Unsteered (capture runs at scale 0), and the branch output is recorded.
    assert np.array(out).tolist() == [[2.0, 4.0, 6.0, 8.0]]
    assert np.array(buf.vectors[1]).tolist() == [2.0, 4.0, 6.0, 8.0]
    st.capture = None
    assert st.hits == {}  # no hits counted while capturing


# ---- settings --------------------------------------------------------------
def test_settings_validation():
    ModelSettings(steering_pack="p", steering_ffn_scale=-1.0, steering_scope="all")
    for bad in (
        dict(steering_pack="p", steering_ffn_scale=float("nan")),
        dict(steering_pack="p", steering_ffn_scale=1000),
        dict(steering_pack="p", steering_scope="nope"),
        dict(steering_pack="p", steering_decode_tokens=0),
        dict(steering_ffn_scale=1.0),  # scale without pack
    ):
        with pytest.raises(ValueError):
            validate_steering(bad)


# ---- cache salt ------------------------------------------------------------
def test_hash_namespace_salt_leaves_model_name():
    m = PagedCacheManager(block_size=4, max_blocks=8, model_name="m")
    base = m.hash_namespace
    assert base == "m"
    m.set_hash_salt("abc")
    assert m.model_name == "m" and m.hash_namespace != base
    h0 = compute_block_hash(b"", [1, 2, 3, 4], model_name=base)
    h1 = compute_block_hash(b"", [1, 2, 3, 4], model_name=m.hash_namespace)
    assert h0 != h1
    m.set_hash_salt("")
    assert compute_block_hash(b"", [1, 2, 3, 4], model_name=m.hash_namespace) == h0


# ---- admin live push -------------------------------------------------------
def test_pool_live_push_and_scheduler_drain():
    from types import SimpleNamespace

    from omlx.engine_pool import EnginePool
    from omlx.scheduler import Scheduler

    got = {}
    sched = SimpleNamespace(request_steering_update=lambda **k: got.update(k))
    core = SimpleNamespace(scheduler=sched, _wake_engine_loop=lambda: got.update(woke=1))
    pool = EnginePool.__new__(EnginePool)
    pool._entries = {"m": SimpleNamespace(engine=core)}
    assert pool.apply_steering_scales("m", ffn=-1.0, scope="all") is True
    assert got["ffn"] == -1.0 and got["scope"] == "all" and got["woke"] == 1
    pool._entries = {"m": SimpleNamespace(engine=None)}
    assert pool.apply_steering_scales("m", ffn=1.0) is False

    # Scheduler drain: state updated, salt pushed, running requests flagged.
    st = _state(ffn_scale=0.0)
    salts, req = [], SimpleNamespace(skip_cache_store=False)
    s = Scheduler.__new__(Scheduler)
    s.model = SimpleNamespace()
    object.__setattr__(s.model, "_omlx_steering", st)
    s.paged_cache_manager = SimpleNamespace(set_hash_salt=salts.append)
    s.running = {"r": req}
    s.prefilling = []
    s._pending_steering = None
    s._process_pending_steering_update()  # nothing pending: no-op
    assert salts == []
    s.request_steering_update(ffn=-0.5, scope="all")
    s._process_pending_steering_update()
    assert st.ffn_scale == -0.5 and st.scope == "all"
    assert salts == [st.salt()] and salts[0] != ""
    assert req.skip_cache_store is True and s._pending_steering is None
