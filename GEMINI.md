# oMLX Project Context

oMLX is a production-ready LLM inference server optimized for Apple Silicon (M1/M2/M3/M4) Macs. It provides high-performance serving with features like continuous batching, tiered KV caching, and multi-model support.

---

## ⚡ Canonical Workflow (Start Here — Always)

This is the single authoritative procedure for any task: fixing bugs, auditing, or verifying the repo. **Follow these steps in order every time.**

```bash
# ── Step 1: Activate the venv (it is always at .venv/) ──────────────────────
source .venv/bin/activate

# ── Step 2: Install dependencies — NO audio extra on Linux/CI ───────────────
# The [audio] extra (mlx-audio) conflicts with mlx-lm > 0.31.2 on non-macOS.
# NEVER run: pip install -e ".[dev,mcp,audio,grammar]"  ← breaks on Linux
pip install -e ".[dev,mcp,grammar]"

# ── Step 3: Compliance gate — run FIRST, takes ~1 second ────────────────────
# Checks: settings.py security defaults, README.md privacy sections,
# Apache-2.0 license headers on every .py file.
PYTHONPATH=. python scripts/compliance_check.py
# Expected: "ALL CHECKS PASSED. Project is compliant with Security Audit v1.0."
# If it fails: fix reported issues BEFORE doing anything else.

# ── Step 4: Run the fast test suite ─────────────────────────────────────────
# PYTHONPATH=. is required so that `import omlx` resolves to the local repo,
# which triggers the MLX mock installation before any mlx.* import runs.
# pytest.ini already sets: addopts = -m "not slow and not integration"
# so the -m flag below is redundant but is kept for clarity.
PYTHONPATH=. pytest -m "not slow" --tb=short -q 2>&1 | tee /tmp/test_output.txt

# ── Step 5: Triage failures (if any) ────────────────────────────────────────
grep "^FAILED\|^ERROR" /tmp/test_output.txt | sed 's/ - .*//' | sort | uniq -c | sort -rn
# Root-cause decision table:
#   "NotImplementedError: mlx.foo()"   → add foo to omlx/utils/mlx_mock.py
#   "vocab file / HarmonyGptOss"       → add pytest.skip() in test fixture
#   numerical precision assertion      → mark test @pytest.mark.slow
#   anything else                      → fix the logic bug

# ── Step 6: Iterate until zero failures ─────────────────────────────────────
PYTHONPATH=. pytest -m "not slow" --tb=no -q 2>&1 | tail -3
# Target line: "N passed, M skipped, K deselected"  (zero "failed")

# ── Step 7: Compliance gate again ───────────────────────────────────────────
PYTHONPATH=. python scripts/compliance_check.py

# ── Step 8: Commit ───────────────────────────────────────────────────────────
git add -A
git restore --staged "oMLX.*.app"   # ← exclude macOS build artifact (hundreds of files)
git commit -m "fix: ..."
```

> **⚠️ Build artifact warning:** `oMLX.0.3.x.app/` is a macOS `.app` bundle that may appear as
> hundreds of staged files. Always run `git restore --staged "oMLX.*.app"` before committing.

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
| `scripts/compliance_check.py` | Automated security/license compliance gate |
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

When a test fails with `NotImplementedError: mlx.foo() is not implemented in the MLX mock`, add `foo` to the `if self.__name__ == "mlx.core":` block inside `MockModule.__getattr__` in `omlx/utils/mlx_mock.py`:

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

`pytest.ini` enforces `addopts = -m "not slow and not integration"` — the default CI run skips both marks automatically.

---

## Merge Conflict Resolution

> **Scope conflict scans to project files only:**
> ```bash
> # CORRECT — only scans omlx/ and tests/
> grep -rl "<<<<<<" omlx/ tests/
> # WRONG — scans packaging/_export/ (thousands of vendored stdlib files → false positives)
> grep -rl "<<<<<<" .
> ```

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

6. **General rule:** When both branches add different features to the same file, keep both. Only discard a change if it is a direct logical contradiction.
