# SPDX-License-Identifier: Apache-2.0
"""Steering quality gate Q7 (plan §2e): KL/NLL, degeneration, structure, effect.

The job runs on a loaded engine with the pack installed, one (scope, scale)
at a time on an idle server, the same way `steer sweep` drives the live
state. A scale-0 pass provides the reference; every later point is compared
against it and gated. Results land in the pack manifest as `quality[]` and
the passing point with the largest effect becomes `usage.recommended`.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "data"
NEUTRAL_DIR = DATA_DIR / "neutral"
QUALITY_DIR = DATA_DIR / "quality"

# Provisional gates (plan §2e table); recalibrate after the first runs.
GATES = {
    "kl": 0.05,  # nats/token
    "nll_rel": 0.02,  # relative NLL increase
    "structure": 0.0,  # no drop in valid-JSON rate (points)
    "repetition": 1.0,  # repeated-4gram rise, points
    "mmlu": 1.0,  # accuracy drop, points
    "gsm8k": 1.5,
    "humaneval": 2.0,
    "truthfulqa": 1.0,
}

# Full-mode benchmarks: name -> (module, class, gate key).
FULL_BENCHMARKS = ("mmlu", "gsm8k", "humaneval", "truthfulqa")


# -- corpus and prompt loaders ----------------------------------------------


def load_paragraphs(directory: Path, limit: int | None = None) -> list[str]:
    """Blank-line separated paragraphs; '#' lines are comments (DS4 rule)."""
    paras: list[str] = []
    for path in sorted(directory.glob("*.txt")):
        block: list[str] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("#"):
                continue
            if not line:
                if block:
                    paras.append(" ".join(block))
                    block = []
            else:
                block.append(line)
        if block:
            paras.append(" ".join(block))
    if limit is not None:
        paras = paras[:limit]
    return paras


def load_neutral_corpus(limit: int | None = None) -> list[str]:
    return load_paragraphs(NEUTRAL_DIR, limit)


def load_prompt_file(name: str, limit: int | None = None) -> list[str]:
    """One prompt per non-empty line; '#' lines are comments (DS4 rule)."""
    prompts = []
    for line in (QUALITY_DIR / name).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            prompts.append(line)
    if limit is not None:
        prompts = prompts[:limit]
    return prompts


# -- text metrics -------------------------------------------------------------


def repetition_rate(text: str, n: int = 4) -> float:
    """Fraction of word n-grams that occur more than once (plan §2e)."""
    words = re.findall(r"\w+", text.lower())
    if len(words) <= n:
        return 0.0
    grams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
    return (len(grams) - len(set(grams))) / len(grams)


def extract_json(text: str) -> str | None:
    """First balanced JSON object or array in the reply, strings respected."""
    starts = sorted(i for i in (text.find("{"), text.find("[")) if i != -1)
    for start in starts:
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch in "{[":
                depth += 1
            elif ch in "}]":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
    return None


def structure_valid(text: str) -> bool:
    blob = extract_json(text)
    if blob is None:
        return False
    try:
        json.loads(blob)
    except ValueError:
        return False
    return True


# -- gates ---------------------------------------------------------------------


def evaluate_gates(entry: dict[str, Any], full: bool) -> tuple[bool, list[str]]:
    """Apply the §2e gates to one quality entry; returns (passed, reasons)."""
    failed: list[str] = []
    if entry["kl"] > GATES["kl"]:
        failed.append(f"kl {entry['kl']:.3f} > {GATES['kl']}")
    if entry["nll_delta_pct"] > GATES["nll_rel"] * 100:
        failed.append(
            f"nll +{entry['nll_delta_pct']:.1f}% > +{GATES['nll_rel'] * 100:.0f}%"
        )
    if entry["repeat_delta_pt"] > GATES["repetition"]:
        failed.append(
            f"repetition +{entry['repeat_delta_pt']:.1f}pt "
            f"> +{GATES['repetition']:.0f}pt"
        )
    if entry["structure_delta_pt"] < GATES["structure"]:
        failed.append(
            f"structure {entry['structure_delta_pt']:+.1f}pt < "
            f"{GATES['structure']:+.0f}pt"
        )
    if full:
        for key in ("mmlu", "gsm8k", "humaneval", "truthfulqa"):
            delta = entry.get(f"{key}_delta_pt")
            if delta is not None and delta < -GATES[key]:
                failed.append(f"{key} {delta:+.1f}pt < -{GATES[key]:.1f}pt")
    return (not failed), failed


# -- teacher-forced KL/NLL pair -------------------------------------------------


def _log_softmax(x: Any, axis: int = -1) -> Any:
    """Numerically stable log-softmax; mlx.core has no mx.log_softmax."""
    import mlx.core as mx

    m = mx.max(x, axis=axis, keepdims=True)
    shifted = x - m
    return shifted - mx.log(mx.exp(shifted).sum(axis=axis, keepdims=True))


def kl_forward_pair(
    fwd: Any,
    ids: list[int],
    st: Any,
    make_cache: Any,
    component: str,
    scale: float,
    chunk: int = 64,
) -> tuple[float, float, float]:
    """Two teacher-forced forwards (scale 0 vs scale) over one text.

    Runs inside the MLX executor. The component scale is toggled directly on
    the steering state and left at ``scale``; each forward gets a fresh cache
    so no KV is reused across the pair. Returns (kl nats/token, nll_ref,
    nll_steer), averaged over predicted positions.
    """
    import mlx.core as mx

    attr = f"{component}_scale"
    saved = getattr(st, attr)
    arr = mx.array(np.asarray(ids, dtype=np.int32))[None]

    def _logits(out: Any) -> Any:
        # mlx_lm models return either bare logits or a LanguageModelOutput.
        return out.logits if hasattr(out, "logits") else out

    try:
        setattr(st, attr, 0.0)
        logits_ref = _logits(fwd(arr, cache=make_cache()))
        setattr(st, attr, scale)
        logits_steer = _logits(fwd(arr, cache=make_cache()))
    finally:
        setattr(st, attr, saved)
    targets = arr[:, 1:]  # next-token ids for positions 0..L-2
    n_pos = targets.shape[1]
    kl_sum = mx.array(0.0)
    nll_ref_sum = mx.array(0.0)
    nll_steer_sum = mx.array(0.0)
    for t0 in range(0, n_pos, chunk):
        t1 = min(t0 + chunk, n_pos)
        lr = _log_softmax(logits_ref[:, t0:t1, :])
        ls = _log_softmax(logits_steer[:, t0:t1, :])
        pr = mx.exp(lr)
        kl_sum = kl_sum + (pr * (lr - ls)).sum()
        idx = targets[:, t0:t1, None]
        nll_ref_sum = nll_ref_sum - mx.take_along_axis(lr, idx, axis=-1).sum()
        nll_steer_sum = nll_steer_sum - mx.take_along_axis(ls, idx, axis=-1).sum()
    kl = float(np.asarray(kl_sum / n_pos))
    nll_ref = float(np.asarray(nll_ref_sum / n_pos))
    nll_steer = float(np.asarray(nll_steer_sum / n_pos))
    return kl, nll_ref, nll_steer


# -- job orchestration ----------------------------------------------------------


@dataclass
class VerifyRequest:
    scopes: list[str] = field(default_factory=lambda: ["prefill+k"])
    scales: list[float] = field(default_factory=lambda: [-0.5, -1.0])
    quick: bool = False
    thinking: bool = False
    samples: int = 300
    neutral_limit: int | None = None
    gen_limit: int | None = None


def _mean_words(texts: list[str]) -> float:
    if not texts:
        return 0.0
    return sum(len(re.findall(r"\w+", t)) for t in texts) / len(texts)


async def _generate_texts(engine: Any, prompts: list[str], tokens: int) -> list[str]:
    texts = []
    for prompt in prompts:
        out = await engine.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=tokens,
        )
        texts.append(out.text or "")
    return texts


def _load_benchmark(name: str) -> Any:
    """Lazy, explicit imports — no dynamic import of caller-supplied names."""
    if name == "mmlu":
        from ..eval.mmlu import MMLUBenchmark

        return MMLUBenchmark()
    if name == "gsm8k":
        from ..eval.gsm8k import GSM8KBenchmark

        return GSM8KBenchmark()
    if name == "humaneval":
        from ..eval.humaneval import HumanEvalBenchmark

        return HumanEvalBenchmark()
    if name == "truthfulqa":
        from ..eval.truthfulqa import TruthfulQABenchmark

        return TruthfulQABenchmark()
    raise ValueError(f"unknown benchmark {name!r}")


async def _benchmark_accuracy(
    engine: Any, benchmark_name: str, samples: int, thinking: bool
) -> float | None:
    """Accuracy of one bundled benchmark, or None when data is unavailable."""
    try:
        benchmark = _load_benchmark(benchmark_name)
        items = await benchmark.load_dataset(samples)
        if not items:
            return None
        result = await benchmark.run(engine, items, enable_thinking=thinking)
        return result.accuracy * 100.0
    except Exception as exc:  # dataset missing or engine refused: skip, don't fail
        logger.warning("quality gate benchmark %s unavailable: %s", benchmark_name, exc)
        return None


async def run_verify_job(
    job: dict[str, Any],
    engine: Any,
    model: Any,
    st: Any,
    request: VerifyRequest,
    pack_dir: Path | None,
) -> None:
    """Drive the (scope, scale) grid and fill ``job`` with quality entries."""
    import asyncio

    from ..engine_core import get_mlx_executor

    loop = asyncio.get_running_loop()
    executor = get_mlx_executor()
    fwd = model if not hasattr(model, "language_model") else model.language_model
    from mlx_lm.models.cache import make_prompt_cache

    component = "ffn" if "ffn" in st.components else "attn"
    scale_attr = f"{component}_scale"
    orig_scale = getattr(st, scale_attr)
    orig_scope = st.scope
    max_tokens = 180
    neutral = load_neutral_corpus(
        request.neutral_limit
        if request.neutral_limit is not None
        else (60 if request.quick else None)
    )
    gen_prompts = load_prompt_file(
        "gen_prompts.txt",
        (
            request.gen_limit
            if request.gen_limit is not None
            else (8 if request.quick else None)
        ),
    )
    struct_prompts = load_prompt_file(
        "structure_prompts.txt",
        (
            request.gen_limit
            if request.gen_limit is not None
            else (8 if request.quick else None)
        ),
    )

    async def measure_point(scope: str, scale: float) -> dict[str, Any]:
        st.set(scope=scope, **{component: scale}, bump=True)
        # KL/NLL: two teacher-forced forwards per paragraph (capture path).
        kls, nll_refs, nll_sts = [], [], []
        for text in neutral:
            ids = engine.tokenizer.encode(text)[:256]
            if len(ids) < 8:
                continue
            kl, nll_ref, nll_st = await loop.run_in_executor(
                executor,
                lambda ids=ids: kl_forward_pair(
                    fwd,
                    ids,
                    st,
                    lambda: make_prompt_cache(fwd),
                    component,
                    scale,
                ),
            )
            kls.append(kl)
            nll_refs.append(nll_ref)
            nll_sts.append(nll_st)
        gen_texts = await _generate_texts(engine, gen_prompts, max_tokens)
        struct_texts = await _generate_texts(engine, struct_prompts, max_tokens)
        entry: dict[str, Any] = {
            "scope": scope,
            "scale": scale,
            "kl": float(np.mean(kls)) if kls else 0.0,
            "nll_ref": float(np.mean(nll_refs)) if nll_refs else 0.0,
            "nll": float(np.mean(nll_sts)) if nll_sts else 0.0,
            "repeat": (
                float(np.mean([repetition_rate(t) for t in gen_texts]))
                if gen_texts
                else 0.0
            ),
            "mean_words": _mean_words(gen_texts),
            "structure": (
                float(np.mean([structure_valid(t) for t in struct_texts]))
                if struct_texts
                else 1.0
            ),
        }
        if not request.quick:
            for key in FULL_BENCHMARKS:
                acc = await _benchmark_accuracy(
                    engine,
                    key,
                    request.samples,
                    request.thinking,
                )
                if acc is not None:
                    entry[key] = acc
        return entry

    try:
        # The scale-0 reference runs once; the grid covers the non-zero points.
        points = [(request.scopes[0], 0.0)] + [
            (scope, scale)
            for scope in request.scopes
            for scale in request.scales
            if scale != 0.0
        ]
        job["total"] = len(points)
        entries: list[dict[str, Any]] = []
        baseline: dict[str, Any] | None = None
        for i, (scope, scale) in enumerate(points):
            entry = await measure_point(scope, scale)
            if scale == 0.0:
                baseline = entry
                entry["role"] = "reference"
            else:
                base = baseline  # None only if the grid was misconfigured
                nll_base = base["nll_ref"] if base else entry["nll_ref"]
                entry["nll_delta_pct"] = (
                    (entry["nll"] - nll_base) / nll_base * 100.0 if nll_base else 0.0
                )
                entry["repeat_delta_pt"] = (
                    (entry["repeat"] - base["repeat"]) * 100.0 if base else 0.0
                )
                entry["structure_delta_pt"] = (
                    (entry["structure"] - base["structure"]) * 100.0 if base else 0.0
                )
                base_words = base["mean_words"] if base else entry["mean_words"]
                entry["effect_pct"] = (
                    (entry["mean_words"] - base_words) / base_words * 100.0
                    if base_words
                    else 0.0
                )
                if not request.quick:
                    for key in FULL_BENCHMARKS:
                        if key in entry and base and key in base:
                            entry[f"{key}_delta_pt"] = entry[key] - base[key]
                passed, failed = evaluate_gates(entry, full=not request.quick)
                entry["passed"] = passed
                if failed:
                    entry["failed"] = failed
            entries.append(entry)
            job["done"] = i + 1
            job["quality"] = entries

        # Recommended: the passing point with the largest |effect| (§2f).
        passing = [e for e in entries if e.get("passed")]
        recommended = None
        if passing:
            recommended = max(passing, key=lambda e: abs(e.get("effect_pct", 0.0)))
        job["recommended"] = (
            {"scope": recommended["scope"], "scale": recommended["scale"]}
            if recommended
            else None
        )
        if pack_dir is not None:
            write_quality_results(pack_dir, entries, job["recommended"])
    except Exception as exc:
        logger.exception("steering verify job failed")
        job["status"] = "error"
        job["error"] = str(exc)
    finally:
        st.set(scope=orig_scope, **{component: orig_scale}, bump=True)
        if job.get("status") != "error":
            job["status"] = "done"


def write_quality_results(
    pack_dir: Path,
    entries: list[dict[str, Any]],
    recommended: dict[str, Any] | None,
) -> None:
    """Persist quality[] and usage.recommended into the pack manifest."""
    from .pack import SteeringPack, load_pack, save_pack

    pack = load_pack(pack_dir)
    manifest = dict(pack.manifest)
    manifest["quality"] = entries
    if recommended is not None:
        usage = dict(manifest.get("usage") or {})
        # install_steering reads rec["ffn"]/rec["attn"] for the default
        # scales at load — a bare float here would crash every later load.
        usage["recommended"] = {pack.component: recommended["scale"]}
        usage["recommended_scope"] = recommended["scope"]
        manifest["usage"] = usage
    save_pack(SteeringPack(pack.name, pack.vectors, manifest), pack_dir)
