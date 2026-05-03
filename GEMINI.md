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

## Key File Locations

- `omlx/cli.py`: CLI entry point.
- `omlx/server.py`: FastAPI server setup and routes.
- `omlx/engine_pool.py`: Model lifecycle management.
- `omlx/scheduler.py`: Request scheduling logic.
- `omlx/settings.py`: Global and per-model settings management.
- `packaging/`: Scripts and config for the macOS `.app` and `.dmg` builds.
- `tests/`: Comprehensive test suite covering units and integrations.
