# oMLX Project Context

oMLX is a production-ready LLM inference server optimized for Apple Silicon (M1/M2/M3/M4) Macs. It provides high-performance serving with features like continuous batching, tiered KV caching, and multi-model support.

## Project Overview

- **Purpose:** Efficient LLM and VLM inference on macOS, compatible with OpenAI and Anthropic APIs.
- **Core Stack:** Python 3.10+, MLX, FastAPI, uvicorn, PyObjC (for menubar app).
- **Key Features:**
    - **Continuous Batching:** High-throughput request processing.
    - **Tiered KV Cache:** Hot (RAM) and Cold (SSD) tiers for persistent context.
    - **Multi-Model Serving:** LRU-based memory management, model pinning, and TTL-based unloading.
    - **Diverse Model Support:** LLM, VLM (Vision), Embedding, Reranker, and Audio (STT/TTS).
    - **Admin Dashboard:** Web UI for monitoring, model management, and built-in chat.
    - **Integrations:** MCP (Model Context Protocol) support and one-click integration for tools like Claude Code.

## Architecture

The system is organized into several key layers:

1.  **API Layer (`omlx/server.py`, `omlx/api/`):** FastAPI implementation of OpenAI and Anthropic compatible endpoints.
2.  **Engine Pool (`omlx/engine_pool.py`):** Manages multiple model engines, handles loading/unloading, and enforces memory limits.
3.  **Inference Engines (`omlx/engine/`):**
    - `BatchedEngine`: Text LLM inference with continuous batching.
    - `VLMBatchedEngine`: Vision-Language Model support.
    - `EmbeddingEngine` / `RerankerEngine`: Specialized task engines.
    - `STTEngine` / `TTSEngine` / `STSEngine`: Audio processing.
4.  **Scheduler (`omlx/scheduler.py`):** FCFS request scheduling integrated with `mlx-lm` BatchGenerator.
5.  **Cache Stack (`omlx/cache/`):**
    - `PagedCacheManager`: Block-based KV cache with prefix sharing.
    - `PagedSSDCacheManager`: Cold storage tier for KV blocks.
6.  **Admin UI (`omlx/admin/`):** Vendored frontend for offline management.

## Building and Running

### Installation

```bash
# Install core dependencies
pip install -e .

# Install with optional components
pip install -e ".[dev,mcp,audio,grammar]"
```

### Running the Server

```bash
# Start server with models directory
omlx serve --model-dir ~/models

# Common flags
--port 8000
--max-model-memory 32GB
--paged-ssd-cache-dir ~/.omlx/cache
--mcp-config mcp.json
--check-updates         # Check for oMLX updates (default: disabled)
--check-statuskit      # Check menubar icon visibility (Tahoe, default: disabled)
```

### Testing

```bash
# Run all fast tests
pytest -m "not slow"

# Run a specific test
pytest tests/test_config.py

# Run slow tests (requires models)
pytest -m slow
```

## Development Conventions

- **Python Version:** 3.10+ (3.11+ recommended for macOS app).
- **License Header:** Every source file must start with `# SPDX-License-Identifier: Apache-2.0`.
- **Formatting:** Adhere to `black` (line-length 88) and `ruff` standards.
- **Type Safety:** Use type hints and run `mypy` for verification.
- **Testing:**
    - Test files should follow the pattern `tests/test_<module_name>.py`.
    - Use `@pytest.mark.slow` for tests requiring model loading.
    - Use `@pytest.mark.integration` for tests requiring a running server.
- **Documentation:** Maintain `README.md` and document new features in the `docs/` directory.
- **Pull Requests:** Ensure all tests pass (`pytest -m "not slow"`) and formatting is correct before submission.

## Known Platform Limitations

- **Non-macOS Environments:** Since oMLX is optimized for Apple Silicon, `mlx` is unavailable on Linux/Windows. The MLX mock (`omlx/utils/mlx_mock.py`) provides a NumPy-backed stub that is automatically installed on non-macOS. See the **MLX Mock Architecture** section below for full details.
- **Dependency Conflicts:** The `[audio]` extra (specifically `mlx-audio`) may have version conflicts with `mlx-lm > 0.31.2`. On non-macOS environments or when using newer `mlx-lm` versions, always install **without** the audio extra:
  ```bash
  pip install -e ".[dev,mcp,grammar]"
  ```
  Never add `audio` to the install command in CI or on Linux — it will break the install.

---

## AI Agent Optimization & Audit Guide

To minimize token usage and turns during audits or verification, use these optimized workflows:

### 1. Environment Setup (Linux/CI)
When running in a non-macOS environment, always install without the `audio` extra and always set `PYTHONPATH=.` so that `import omlx` resolves to the local repo (which triggers mock installation before `mlx` is imported):
```bash
# Install — no audio extra
pip install -e ".[dev,mcp,grammar]"

# Run all fast tests
PYTHONPATH=. pytest -m "not slow"
```
`pytest.ini` already contains `addopts = -m "not slow and not integration"` so the `-m "not slow"` flag is redundant but harmless. Never run the full suite without `-m "not slow"` unless you have real Apple Silicon hardware.

### 2. Programmatic Compliance Audit
**Always run this first** before any manual inspection. It checks `settings.py` defaults, `README.md` privacy sections, and Apache-2.0 license headers on all 300+ Python files in one pass:
```bash
python scripts/compliance_check.py
```
Expected output: `ALL CHECKS PASSED. Project is compliant with Security Audit v1.0.`

If it fails, fix the reported issues **before** touching anything else. License headers are mandatory on every `.py` file — the format is exactly `# SPDX-License-Identifier: Apache-2.0` as the first line.

### 3. MLX Mock Architecture — Single Source of Truth

#### The Problem This Solves
`mlx` is an Apple-Silicon-only C++ framework. On Linux/CI it cannot be installed. All tests must still run. The solution is a pure-Python/NumPy import-hook mock that intercepts `import mlx.core` and returns a fake module backed by NumPy arrays.

#### File Locations and Roles
| File | Role | Edit? |
|---|---|---|
| `omlx/utils/mlx_mock.py` | **The only real mock file.** Contains all mock logic. | ✅ Yes |
| `tests/mlx_mock.py` | **Compatibility shim only.** 5 lines. Re-exports everything from the utils mock. | ❌ Never edit directly |

**Critical rule: never edit `tests/mlx_mock.py` directly.** It is a shim:
```python
from omlx.utils.mlx_mock import *  # noqa: F401, F403
from omlx.utils.mlx_mock import MockMLXLoader, MockMLXFinder, install_mock, _map_dtype
```
All changes go to `omlx/utils/mlx_mock.py`. The shim exists only as a compatibility alias in case anything ever imports it directly.

**History note:** Before May 2026, there were two nearly-identical 430-line files. They diverged silently over time, causing hard-to-diagnose test failures. The consolidation commit (`fe6066a`) eliminated this. Do not re-introduce the two-file pattern.

#### How the Mock Is Loaded (The Loading Chain)
```
PYTHONPATH=. pytest
  └─ tests/conftest.py
       └─ import omlx                        ← triggers __init__.py
            └─ omlx/__init__.py (line 21-36) ← detects platform != Darwin
                 └─ loads omlx/utils/mlx_mock.py via importlib.util
                      └─ calls install_mock()
                           └─ inserts MockMLXFinder into sys.meta_path[0]
                                └─ all subsequent `import mlx.*` → mock
```
After `import omlx`, any `import mlx.core as mx` in any module returns the mock. This happens **before** any test module is loaded because `conftest.py` runs first.

#### How the Mock Works Internally
`MockMLXLoader` is a Python import-hook (`importlib.abc.Loader`). It intercepts module creation for `mlx`, `mlx.core`, `mlx.core.random`, `mlx.core.linalg`, `mlx.core.distributed`, `mlx.nn`, `mlx.utils`, and `mlx_lm`.

The `array` class inside the mock wraps a `numpy.ndarray` (`self._data`). All arithmetic operations (`+`, `-`, `*`, `/`, `@`, `**`, bitwise ops) delegate to NumPy. dtype constants (`mx.float32`, `mx.bfloat16`, etc.) are Python strings. `bfloat16` is mapped to `float16` internally since NumPy has no bfloat16.

Named functions (like `mx.zeros`, `mx.concatenate`, `mx.pad`, etc.) are explicitly registered as lambdas in `MockModule.__getattr__`. **Unknown function names** now raise:
```
NotImplementedError: mlx.<name>() is not implemented in the MLX mock.
Add it to omlx/utils/mlx_mock.py.
```

#### How to Add a Missing MLX Function
When a test fails with `NotImplementedError: mlx.foo() is not implemented...`, open `omlx/utils/mlx_mock.py` and find the block of `if name == ...` handlers inside `MockModule.__getattr__` (around line 290–345). Add a line using the existing patterns:

```python
# Simple passthrough (no-op)
if name == "contiguous": return lambda a, **k: a

# NumPy delegation (element-wise)
if name == "power": return lambda a, b: loader.array(
    np.power(loader.array(a)._data, loader.array(b)._data if hasattr(b, "_data") else b))

# Shape-reducing operation
if name in ("mean", "max", "min", "sum"):
    return lambda a, axis=None, keepdims=False: loader.array(
        getattr(np, name)(loader.array(a)._data, axis=axis, keepdims=keepdims))
```

**Never use `_default_func` intentionally.** It exists only as a last-resort backstop that raises `NotImplementedError`. If `_default_func` triggers during a test, it means a new MLX function was used in production code without a mock entry — fix it.

#### Tests That Cannot Be Mocked: `@pytest.mark.slow`
Some tests require the **actual numerical output** of MLX operations, which NumPy cannot faithfully replicate (different precision, different rounding, different FP8/bfloat16 semantics). These tests are marked `@pytest.mark.slow` and skipped in CI:

```python
@pytest.mark.slow
def test_embed_produces_normalized_vectors(self):
    """Requires real MLX model forward pass for numerical correctness."""
```

**Rule:** Mark a test `@pytest.mark.slow` when it:
1. Requires loading a real model from disk, **OR**
2. Asserts numerical precision that depends on real MLX hardware (e.g., turboquant MSE thresholds, FP8 dequant exact values).

Do **not** mark a test slow just because it uses `mlx` — the mock handles structural correctness tests fine.

#### Tests That Require External Data: `pytest.skip`
Tests that require downloading a vocabulary file, external model, or any network resource use `pytest.importorskip` or inline `pytest.skip`:

```python
try:
    encoding = load_harmony_encoding("HarmonyGptOss")
except Exception:
    pytest.skip("HarmonyGptOss vocabulary file unavailable in this environment")
```

This pattern is used in `tests/test_harmony.py`, `tests/test_harmony_parser.py`, and `tests/test_output_parser.py`.

---

### 4. Merge Conflict Resolution Strategy

When resolving merge conflicts in this repo, follow this priority order:

1. **`omlx/_version.py`** — Always keep the Apache-2.0 license header AND the higher version number. The license header was added by one branch; the version bump by another. Both must survive.
   ```python
   # SPDX-License-Identifier: Apache-2.0
   __version__ = "0.3.9"   # keep the higher of the two conflicting versions
   ```

2. **`omlx/utils/mlx_mock.py`** — The utils mock is the single source of truth. Never let a merge revert it to an older state. After any merge, immediately run `PYTHONPATH=. pytest -m "not slow"` to verify.

3. **`tests/mlx_mock.py`** — This file must remain a shim. If a merge overwrites it with the old 430-line duplicate, replace its content with the 5-line re-export shown above.

4. **General conflict rule:** When HEAD and origin/main add different features to the same file, keep both. Only discard a change if it is a direct logical contradiction.

---

### 5. Full "Fix, Compile, Pass Tests, Comply" Workflow

This is the complete ordered procedure for bringing the repo to a clean state after a merge or after receiving a broken branch:

```bash
# Step 0: Install (no audio on non-macOS)
pip install -e ".[dev,mcp,grammar]"

# Step 1: Compliance check first — fast, catches license/settings issues
python scripts/compliance_check.py

# Step 2: Run all fast tests and capture failures
PYTHONPATH=. pytest -m "not slow" --tb=short -q 2>&1 | grep "FAILED\|ERROR"

# Step 3: For each failure, determine root cause:
#   - NotImplementedError: mlx.foo() → add foo to omlx/utils/mlx_mock.py
#   - Harmony/vocab not found → add pytest.skip() to fixture
#   - Numerical precision → mark @pytest.mark.slow
#   - Logic bug → fix the bug

# Step 4: Re-run until zero failures
PYTHONPATH=. pytest -m "not slow" --tb=no 2>&1 | grep "FAILED" | wc -l
# Must be 0

# Step 5: Compliance check again (confirms no license headers were removed)
python scripts/compliance_check.py

# Step 6: Commit
git add -A
# IMPORTANT: exclude build artifacts like oMLX.*.app/ from staging
git restore --staged "oMLX.*.app"
git commit -m "fix: ..."
```

**Important:** The `.app` bundle (`oMLX.0.3.x.app/`) is a macOS build artifact with hundreds of files. Always unstage it before committing: `git restore --staged "oMLX.*.app"`. It should be added to `.gitignore` if not already.

---

### 6. Key Source Files Reference

- `omlx/cli.py`: CLI entry point (`omlx serve ...`).
- `omlx/server.py`: FastAPI server setup and routes.
- `omlx/engine_pool.py`: Model lifecycle management (load/unload/LRU).
- `omlx/scheduler.py`: Request scheduling logic (FCFS + chunked prefill).
- `omlx/settings.py`: Global and per-model settings — **security-sensitive defaults live here**.
- `omlx/utils/mlx_mock.py`: **The MLX mock — single source of truth for non-macOS testing.**
- `scripts/compliance_check.py`: Automated security/license compliance verifier.
- `packaging/`: Scripts and config for the macOS `.app` and `.dmg` builds.
- `tests/`: Comprehensive test suite (~2950+ fast tests, ~10 slow tests).
