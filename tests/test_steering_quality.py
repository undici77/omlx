# SPDX-License-Identifier: Apache-2.0
"""Fast tests for the steering quality gate."""

import json

import numpy as np
import pytest

from omlx.steering.quality import (
    GATES,
    evaluate_gates,
    extract_json,
    kl_forward_pair,
    load_neutral_corpus,
    load_prompt_file,
    repetition_rate,
    structure_valid,
    write_quality_results,
)


def test_repetition_rate_bounds():
    assert repetition_rate("") == 0.0
    assert repetition_rate("too few words") == 0.0


def test_repetition_rate_detects_loops():
    looped = "the cat sat on the mat " * 12
    assert repetition_rate(looped) > 0.5


def test_repetition_rate_low_for_prose():
    prose = (
        "The river bends around the old mill and passes willow, reed and "
        "stone before it reaches the sea where gulls wait for the tide."
    )
    assert repetition_rate(prose) < 0.1


def test_extract_json_plain_and_embedded():
    assert json.loads(extract_json('{"a": 1}')) == {"a": 1}
    assert json.loads(extract_json('Here you go: {"a": 1} hope that helps.')) == {
        "a": 1
    }
    assert json.loads(extract_json("items: [1, 2, 3]")) == [1, 2, 3]


def test_extract_json_respects_braces_in_strings():
    text = '{"code": "}{"}'
    assert json.loads(extract_json(text)) == {"code": "}{"}


def test_structure_valid_rejects_broken_json():
    assert structure_valid('{"a": 1}') is True
    assert structure_valid('Sure! {"a": 1}') is True
    assert structure_valid('{"a": 1') is False
    assert structure_valid("no json here") is False
    assert structure_valid("[1, 2,") is False


def test_neutral_corpus_ships_200_paragraphs():
    corpus = load_neutral_corpus()
    assert len(corpus) >= 200
    for para in corpus:
        assert len(para.split()) >= 25


def test_quality_prompt_sets_shipped():
    gen = load_prompt_file("gen_prompts.txt")
    struct = load_prompt_file("structure_prompts.txt")
    assert len(gen) >= 20
    assert len(struct) >= 10


def _entry(**overrides):
    entry = {
        "scope": "prefill+k",
        "scale": -1.0,
        "kl": 0.01,
        "nll_delta_pct": 0.5,
        "repeat_delta_pt": 0.2,
        "structure_delta_pt": 0.0,
    }
    entry.update(overrides)
    return entry


def test_gates_pass_on_healthy_entry():
    passed, failed = evaluate_gates(_entry(), full=False)
    assert passed and not failed


def test_gates_fail_on_kl_and_structure():
    passed, failed = evaluate_gates(
        _entry(kl=GATES["kl"] + 0.01, structure_delta_pt=-5.0), full=False
    )
    assert not passed
    assert any("kl" in f for f in failed)
    assert any("structure" in f for f in failed)


def test_gates_full_mode_checks_benchmarks():
    passed, failed = evaluate_gates(
        _entry(mmlu_delta_pt=-2.0, gsm8k_delta_pt=-0.5), full=True
    )
    assert not passed
    assert any("mmlu" in f for f in failed)


def test_write_quality_results_updates_manifest(tmp_path):
    from omlx.steering.pack import SteeringPack, load_pack, save_pack

    vectors = np.ones((2, 4), dtype=np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    manifest = {
        "name": "q",
        "model": {"model_type": "x", "hidden_size": 4, "n_layers": 2},
        "hook": {"component": "ffn", "mode": "block"},
        "usage": {"scope": "prefill+k"},
    }
    dest = save_pack(SteeringPack("q", vectors, manifest), tmp_path / "q")
    entries = [_entry()]
    write_quality_results(dest, entries, {"scope": "prefill+k", "scale": -1.0})
    reloaded = load_pack(dest)
    assert reloaded.manifest["quality"] == entries
    # install_steering reads rec["ffn"]/rec["attn"] — the recommended scale
    # must land as a dict on the pack's component, never a bare float.
    assert reloaded.manifest["usage"]["recommended"] == {"ffn": -1.0}
    assert reloaded.manifest["usage"]["recommended_scope"] == "prefill+k"


class _FakeState:
    """Steering-state stand-in: only the scale attribute matters here."""

    def __init__(self):
        self.ffn_scale = 0.0
        self.attn_scale = 0.0


class _ModelOutput:
    """mlx_lm LanguageModelOutput stand-in (logits attribute, no indexing)."""

    def __init__(self, logits):
        self.logits = logits


def _fake_fwd(logits, wrapped=False):
    def fwd(arr, cache=None):
        return _ModelOutput(logits) if wrapped else logits

    return fwd


def _fake_fwd_pair(first, second, wrapped=False):
    """Fake model whose first forward returns ``first``, the second ``second``."""
    calls = {"n": 0}

    def fwd(arr, cache=None):
        calls["n"] += 1
        out = first if calls["n"] == 1 else second
        return _ModelOutput(out) if wrapped else out

    return fwd


def _fake_cache():
    return None


@pytest.mark.slow
def test_kl_forward_pair_zero_for_identical_logits():
    import mlx.core as mx

    logits = mx.array(np.random.RandomState(0).randn(1, 6, 16).astype(np.float32))
    st = _FakeState()
    st.ffn_scale = 1.0  # the job sets the working scale before the pair
    ids = [1, 2, 3, 4, 5, 6]
    kl, nll_ref, nll_steer = kl_forward_pair(
        _fake_fwd(logits), ids, st, _fake_cache, "ffn", 1.0
    )
    assert kl == pytest.approx(0.0, abs=1e-6)
    assert nll_ref == pytest.approx(nll_steer, rel=1e-6)
    assert nll_ref > 0.0
    assert st.ffn_scale == 1.0  # the pair restores the pre-call scale


@pytest.mark.slow
def test_kl_forward_pair_positive_for_shifted_logits():
    import mlx.core as mx

    rng = np.random.RandomState(1)
    base = mx.array(rng.randn(1, 6, 16).astype(np.float32))
    shifted = mx.array((rng.randn(1, 6, 16) + 0.5).astype(np.float32))
    st = _FakeState()
    kl, _, _ = kl_forward_pair(
        _fake_fwd_pair(base, shifted),
        [1, 2, 3, 4, 5, 6],
        st,
        _fake_cache,
        "ffn",
        1.0,
    )
    assert kl > 0.0


@pytest.mark.slow
def test_kl_forward_pair_accepts_language_model_output():
    """qwen4_exp-style models return a LanguageModelOutput, not bare logits."""
    import mlx.core as mx

    logits = mx.array(np.random.RandomState(2).randn(1, 6, 16).astype(np.float32))
    st = _FakeState()
    kl, nll_ref, nll_steer = kl_forward_pair(
        _fake_fwd(logits, wrapped=True),
        [1, 2, 3, 4, 5, 6],
        st,
        _fake_cache,
        "ffn",
        1.0,
    )
    assert kl == pytest.approx(0.0, abs=1e-6)
    assert nll_ref == pytest.approx(nll_steer, rel=1e-6)
