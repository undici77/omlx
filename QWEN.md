# oMLX Project Context

oMLX is a production-ready LLM inference server optimized for Apple Silicon (M1/M2/M3/M4) Macs. It provides high-performance serving with features like continuous batching, tiered KV caching, and multi-model support.

---

## ⚡ Canonical Workflow (Start Here — Always)

The single authoritative procedure for any task (bug fix, merge, audit, verification). **Run the steps in order; stop at the first failing gate and fix it before continuing.**

| # | Gate | Command | Pass criterion | If it fails |
|---|------|---------|----------------|-------------|
| 1 | Conflict markers | `git diff --name-only --diff-filter=U; grep -rlE "^(<<<<<<<\|>>>>>>>) " omlx/ tests/ scripts/ apps/ *.md` | both empty | Resolve (see *Merge Conflict Resolution*). Never commit markers. |
| 2 | Compliance | `PYTHONPATH=. python scripts/compliance_check.py` | `ALL CHECKS PASSED` | Fix the reported default/header/README item. |
| 3 | **Security & privacy scan** | `python scripts/security_scan.py` | `SECURITY SCAN PASSED` | See *Security scan failed — what to do* below. |
| 4 | Fast tests | `PYTHONPATH=. pytest -m "not slow" --tb=short -q` | 0 failed, 0 collection errors | Triage table below. |
| 5 | Built package *(only when you build/release)* | `python scripts/security_scan.py --dist dist/*.whl` (or unpacked `.app`) | `SECURITY SCAN PASSED` | Same as gate 3. |
| 6 | Re-run gates 2 + 3 | as above | pass | Something you changed in step 4 broke a gate. |
| 7 | Commit | `git add -A && git restore --staged "oMLX.*.app"` then `git commit` | — | The `.app` bundle is a build artifact: never commit it. |

```bash
source .venv/bin/activate                  # venv is always .venv/
pip install -e ".[dev,mcp,grammar]"        # NEVER add the [audio] extra on Linux (conflicts with mlx-lm)
export PYTHONPATH=.                        # makes `import omlx` hit the repo → installs the MLX mock first
```

> On Linux there is **no MLX**: the NumPy mock (`omlx/utils/mlx_mock.py`) stands in. Tests verify logic only; hardware numerics are `@pytest.mark.slow` (Apple Silicon only).

### Security scan failed — what to do

`scripts/security_scan.py` fails on anything **new** versus the reviewed `scripts/security_baseline.json`. Read each `[FAIL]` line and decide:

| Finding | Question to answer | Action |
|---------|-------------------|--------|
| `[new-network-host] x.com` | Is this a documented feature (HF download, web search, update check, omlx.ai upload)? Is it opt-in / user-initiated? | Legit → `--update-baseline` + justify in PR. Unknown/unexplained → **remove the code**, treat as suspicious. |
| `[telemetry:*]` | Does any data leave the device without explicit user opt-in? | Almost always **reject**. Only baseline if it is a false positive (docs, dataset text, minified lib). |
| `[exfil:*]` (ssh/browser/keychain/webhook/IP) | Is it the feature's stated purpose (cluster SSH keys, HF-token keychain item)? | Legit → baseline + explain. Anything else → **reject and escalate**. |
| `[danger:dyn-exec / pickle / os-system / shell-true / pipe-to-shell]` | Can attacker-controlled data (network, model files, prompts) reach it? | Reachable → rewrite (no eval/pickle on untrusted input; arg lists, no shell). Not reachable → baseline + justify. |
| `[obfuscation:*]` | Why is code/base64 unreadable? | Test fixtures only → baseline. Otherwise **reject**. |
| `[unexpected-binary/executable]` | Who built it, from which reviewed source? | Rebuild from `csrc/` in CI, then `--update-baseline --dist` and note the SHA-256. |
| `merge-conflict markers in …` | — | Resolve the conflict; never baseline. |
| `deps:url-or-git-dependency` | Is the git SHA pinned and the repo trusted? | Pinned + trusted → baseline. Unpinned → pin it. |

Rules: **never run `--update-baseline` just to turn the scan green.** Review `git diff scripts/security_baseline.json` line by line and explain every addition in the PR description. Full policy: `SECURITY_AUDIT.md` §5.6.

### Triage test failures (gate 4)

```bash
PYTHONPATH=. pytest -m "not slow" --tb=short -q 2>&1 | tee /tmp/test_output.txt
sed 's/\x1b\[[0-9;]*m//g' /tmp/test_output.txt | grep "^FAILED\|^ERROR" | sed 's/::.*//' | sort | uniq -c | sort -rn
```

| Symptom | Root cause | Fix (never extend the mock for MLX *internals*) |
|---------|-----------|-----|
| **Collection error** (`NotImplementedError` / `ModuleNotFoundError: mlx_audio` at import) | Module-scope MLX/vendor code or missing `[audio]` extra | Add the file to `_MLX_IMPORT_SKIP` in `tests/conftest.py` with a one-line reason. |
| Test fails with `NotImplementedError: mlx.foo()` | Simple op (reshape, mean, passthrough…) | Add `foo` to `omlx/utils/mlx_mock.py` (see *Adding a missing MLX function*). |
| …same, but needs real MLX internals (RotatingKVCache, `mx.key`, `as_strided`, `grid_sample`, quantize shapes, `TokenizerWrapper.tool_parser`) | Mock cannot reproduce C++ behaviour | Add the file to `_MLX_SKIP_REASONS` in `tests/conftest.py`. |
| Needs real Harmony token IDs | Mock can't emit them | Nothing — conftest auto-skips "harmony" tests. |
| Needs a model/vocab file not in repo | Offline CI | `pytest.skip()` in the fixture, or `@pytest.mark.slow`. |
| Numerical precision assertion | NumPy ≠ MLX numerics | `@pytest.mark.slow`. |
| Fails only in this container (process groups, ports) | Environment | Mark/skip with reason; do not "fix" the code. |
| Anything else | Real bug | Fix the logic. |

Repeat until `PYTHONPATH=. pytest -m "not slow" --tb=no -q | tail -3` shows `N passed, M skipped, K deselected` with **zero failed and zero errors**. Do not hide a failure by deleting a test.

> **Known state (2026-10-15, Linux, no MLX):** ~135 mock-limitation failures remain untriaged (`tests/integration/test_e2e_streaming.py`, `test_k2_*`, `test_mlx0324_compat.py`, `test_sampling.py`, `test_output_parser.py`, audio tests, `test_splash_engine.py`, …). Treat them as "triage per table above", not as regressions, unless your change touched that area.

> **⚠️ Build artifact warning:** `oMLX.0.3.x.app/` is a macOS `.app` bundle that may appear as hundreds of staged files. Always run `git restore --staged "oMLX.*.app"` before committing.

---

## Project Overview

- **Purpose:** Efficient LLM and VLM inference on macOS, compatible with OpenAI and Anthropic APIs.
- **Core Stack:** Python 3.10+, MLX, FastAPI, uvicorn, Swift/SwiftUI (for macOS menubar app).
- **Venv location:** `.venv/` (always `source .venv/bin/activate` first).
- **Key Features:**
    - **Continuous Batching:** High-throughput request processing.
    - **Tiered KV Cache:** Hot (RAM) and Cold (SSD) tiers for persistent context.
    - **Multi-Model Serving:** LRU-based memory management, model pinning, and TTL-based unloading.
    - **Diverse Model Support:** LLM, VLM (Vision), Embedding, Reranker, and Audio (STT/TTS).
    - **Admin Dashboard:** Web UI for monitoring, model management, and built-in chat.
    - **Integrations:** MCP (Model Context Protocol) support and one-click integration for tools like Claude Code.

## Architecture

1. **API Layer** (`omlx/server.py`, `omlx/api/`): FastAPI — OpenAI and Anthropic compatible endpoints.
2. **Engine Pool** (`omlx/engine_pool.py`): Model lifecycle — load/unload/LRU/memory limits.
3. **Inference Engines** (`omlx/engine/`): `BatchedEngine` (LLM), `VLMBatchedEngine`, `EmbeddingEngine`, `RerankerEngine`, `STTEngine`, `TTSEngine`, `STSEngine`.
4. **Scheduler** (`omlx/scheduler.py`): FCFS scheduling integrated with `mlx-lm` BatchGenerator.
5. **Cache Stack** (`omlx/cache/`): `PagedCacheManager` (hot, RAM) + `PagedSSDCacheManager` (cold, SSD).
6. **Admin UI** (`omlx/admin/`): Vendored frontend for offline management.

## Key Source Files

| File | Purpose |
|------|---------|
| `omlx/cli.py` | CLI entry point (`omlx serve ...`) |
| `omlx/server.py` | FastAPI server setup and routes |
| `omlx/engine_pool.py` | Model lifecycle management |
| `omlx/scheduler.py` | Request scheduling (FCFS + chunked prefill) |
| `omlx/settings.py` | Global/per-model settings — **security-sensitive defaults** |
| `omlx/utils/mlx_mock.py` | **The MLX mock — single source of truth for Linux/CI testing** |
| `scripts/compliance_check.py` | Compliance gate: security defaults, README privacy, license headers |
| `scripts/security_scan.py` + `security_baseline.json` | Merge-time malware/telemetry/exfiltration scan (source + built packages) |
| `tests/` | ~4600+ fast tests, ~10 slow (Apple Silicon only) |
| `packaging/` | macOS `.app` and `.dmg` build scripts |

## Development Conventions

- **Python:** 3.10+ (3.11+ recommended for macOS app).
- **License header:** Every `.py` file must begin with exactly: `# SPDX-License-Identifier: Apache-2.0`
- **Formatting:** `black` (line-length 88) + `ruff`.
- **Type safety:** type hints throughout; verify with `mypy`.
- **Test naming:** `tests/test_<module_name>.py`.
- **PR gate:** `PYTHONPATH=. pytest -m "not slow"` must pass with zero failures.

## Running the Server (macOS only)

```bash
omlx serve --model-dir ~/models [--port 8000] [--max-model-memory 32GB] \
           [--paged-ssd-cache-dir ~/.omlx/cache] [--mcp-config mcp.json]
# check-updates and check-statuskit default to disabled — do not enable without user consent
```

---

## MLX Mock Architecture

### Why it exists

`mlx` is an Apple-Silicon-only C++ framework unavailable on Linux/CI. The mock (`omlx/utils/mlx_mock.py`) is a NumPy-backed `importlib` hook that intercepts every `import mlx.*` and returns a structurally correct fake module. Tests verify logic and data flow; numerical precision is left to `@pytest.mark.slow` tests on real hardware.

### File roles — strict rule

| File | Role | May be edited? |
|------|------|---------------|
| `omlx/utils/mlx_mock.py` | **The only real mock.** All logic lives here. | ✅ Yes |
| `tests/mlx_mock.py` | **5-line compatibility shim only.** Re-exports from the utils mock. | ❌ Never |

`tests/mlx_mock.py` must always contain exactly:
```python
from omlx.utils.mlx_mock import *  # noqa: F401, F403
from omlx.utils.mlx_mock import MockMLXLoader, MockMLXFinder, install_mock, _map_dtype
```
If a merge ever reverts it to a 400+ line duplicate, restore the shim immediately. Two diverging mock files caused silent test failures before the consolidation commit `fe6066a` (May 2026).

### How the mock is loaded

```
PYTHONPATH=. pytest
  └─ tests/conftest.py
       └─ import omlx                        ← triggers omlx/__init__.py
            └─ platform.system() != "Darwin" ← guard: mock only on non-macOS
                 └─ install_mock()
                      └─ sys.meta_path.insert(0, MockMLXFinder())
                           └─ all subsequent `import mlx.*` → mock
```

**The mock is active for the entire test process.** It cannot activate on macOS (production). `PYTHONPATH=.` is mandatory so `import omlx` resolves to the local repo, not an installed package.

### Adding a missing MLX function

**Only for simple ops** (passthrough, NumPy delegation, shape reductions). If the test needs real MLX internals, skip it via `tests/conftest.py` instead (see triage table). When a test fails with `NotImplementedError: mlx.foo() is not implemented in the MLX mock`, add `foo` to the `if self.__name__ == "mlx.core":` block inside `MockModule.__getattr__` in `omlx/utils/mlx_mock.py`:

```python
# No-op / passthrough
if name == "contiguous": return lambda a, **k: a
if name == "compile":    return lambda f, *a, **k: f   # identity — handles @mx.compile

# NumPy delegation
if name == "power":
    return lambda a, b: loader.array(
        np.power(loader.array(a)._data, loader.array(b)._data if hasattr(b, "_data") else b))

# Shape-reducing
if name in ("mean", "max", "min", "sum"):
    return lambda a, axis=None, keepdims=False: loader.array(
        getattr(np, name)(loader.array(a)._data, axis=axis, keepdims=keepdims))
```

> **⚠️ Anti-pattern — double-lambda bug:** `_default_func` is called when the attribute is **invoked**
> (e.g., `make_logits_processors(...)`). Return the **value**, not a lambda wrapping the value:
> ```python
> # WRONG — returns a lambda, not the list:
> if _n == "make_logits_processors": return lambda *a, **k: []
> # CORRECT — returns the list directly:
> if _n == "make_logits_processors": return []
> ```

**Fail-loud policy:** `_default_func` must always raise `NotImplementedError` for unknown names. Never make it return a default value silently — that masks bugs.

### Test marking rules

| Situation | Action | Reason |
|-----------|--------|--------|
| Test requires a real model on disk | `@pytest.mark.slow` | Cannot run in CI |
| Test asserts hardware-dependent numerical precision | `@pytest.mark.slow` | NumPy != MLX numerics |
| Test needs a vocab/external file not in repo | `pytest.skip()` in fixture | Offline CI must not fail |
| Test uses `mlx` for structural logic only | No special mark needed | Mock handles it |
| Test needs real Harmony token IDs (`load_harmony_gpt_oss_encoding`, `HarmonyEncoding.encode()`) | **Skip automatically via conftest** — do NOT add manual skip markers to test files | Mock cannot produce exact token IDs; conftest.py `pytest_collection_modifyitems` auto-skips all such tests |

> **⚠️ New rule — skip Harmony tests, don't maintain the mock:**
> The MLX mock cannot produce the exact token IDs that the real `openai_harmony`
> library generates. Tests that depend on real Harmony token IDs would fail spuriously.
> **Do NOT add mock implementations for `HarmonyEncoding`, `StreamableParser`, etc.**
> Instead, `tests/conftest.py` automatically skips any test file or method whose
> name/path contains "harmony" when the mock is active. This keeps test files
> unchanged — zero modifications needed.

`pytest.ini` enforces `addopts = -m "not slow and not integration"` — the default CI run skips both marks automatically.

---

## Merge Conflict Resolution

> **Find conflicts, scoped to project files only:**
> ```bash
> git diff --name-only --diff-filter=U                      # files git still sees as unmerged
> grep -rlE "^(<<<<<<<|>>>>>>>) " omlx/ tests/ scripts/ apps/ *.md   # leftover markers
> # WRONG: `grep -rl "<<<<<<" .` scans packaging/_export/ (vendored stdlib → false positives)
> ```
> `security_scan.py` also fails on any leftover marker. After resolving, `git add` the file so it leaves the *unmerged* state.

Priority order when resolving conflicts:

1. **`omlx/_version.py`** — Keep the Apache-2.0 header AND the higher version number. Both must survive.
   ```python
   # SPDX-License-Identifier: Apache-2.0
   __version__ = "0.3.12"   # always keep the higher of the two conflicting versions
   ```

2. **`omlx/settings.py`** — Run `PYTHONPATH=. python scripts/compliance_check.py` after resolving to confirm security defaults were not accidentally changed.

3. **`omlx/utils/mlx_mock.py`** — Single source of truth. Never let a merge revert it to an older state.

4. **`tests/mlx_mock.py`** — Must remain the 5-line shim shown above.

5. **`tests/conftest.py`** — Must `import omlx` before any `import mlx.*`.

6. **`scripts/security_baseline.json`** — Never hand-merge. Take either side, then run `python scripts/security_scan.py`; if it fails on legitimate additions from both branches, review them and run `--update-baseline`.

7. **General rule:** When both branches add different features to the same file, keep both. Only discard a change if it is a direct logical contradiction.

---

## macOS App Build (xcodebuild) — Log Locations

The macOS Swift app is built by `build_tahoe.sh` (or `apps/omlx-mac/Scripts/build.sh`). Logs are written to:

| Log | Path | Purpose |
|-----|------|---------|
| **xcodebuild** | `apps/omlx-mac/build/xcodebuild.log` | Full xcodebuild output — search here for actual compiler **errors** (`grep -n "error:"`) vs warnings |
| **SPM resolve** | `apps/omlx-mac/build/spm-resolve.log` | Swift Package Manager dependency resolution output |

### How to diagnose a Swift build failure

```bash
# 1. Find actual compiler errors (not warnings, not the "Error Domain" simulator noise)
grep -n "error:" apps/omlx-mac/build/xcodebuild.log | grep -v "Error Domain"

# 2. See the last ~50 lines of the build (the failure summary + the swift-frontend command)
tail -50 apps/omlx-mac/build/xcodebuild.log

# 3. Search for a specific file's errors
grep -A2 "SomeFile.swift:.*error:" apps/omlx-mac/build/xcodebuild.log
```

**Note:** The xcodebuild log is overwritten on each build run. If the build fails, the log still contains the full output — read it before the next build clears it.
