<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/images/icon-rounded-dark.svg" width="140">
    <source media="(prefers-color-scheme: light)" srcset="docs/images/icon-rounded-light.svg" width="140">
    <img alt="oMLX" src="docs/images/icon-rounded-light.svg" width="140">
  </picture>
</p>

<h1 align="center">oMLX</h1>
<p align="center"><b>LLM inference, optimized for your Mac</b><br>Continuous batching and tiered KV caching, managed directly from your menu bar.</p>

<p align="center">
<a href="https://www.buymeacoffee.com/jundot"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me A Coffee" height="40"></a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/license-Apache%202.0-blue" alt="License">
  <img src="https://img.shields.io/badge/python-3.10+-green" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/platform-Apple%20Silicon-black?logo=apple" alt="Apple Silicon">
</p>

<p align="center">
  <a href="mailto:junkim.dot@gmail.com">junkim.dot@gmail.com</a> · <a href="https://omlx.ai/me">https://omlx.ai/me</a>
</p>

<p align="center">
  <a href="#install">Install</a> ·
  <a href="#quickstart">Quickstart</a> ·
  <a href="#features">Features</a> ·
  <a href="#models">Models</a> ·
  <a href="#cli-configuration">CLI Configuration</a> ·
  <a href="https://omlx.ai/benchmarks">Benchmarks</a> ·
  <a href="https://omlx.ai">oMLX.ai</a>
</p>

<p align="center">
  <b>English</b> ·
  <a href="README.zh.md">中文</a> ·
  <a href="README.ko.md">한국어</a> ·
  <a href="README.ja.md">日本語</a>
</p>

---

## 🛡️ Privacy & Secure macOS Build (Isolated Build)

This version of oMLX is built with a **Privacy-First** philosophy. We believe that a local LLM server should be exactly that: local and private. To ensure the highest standards of security for the developer community and all users, we have implemented several critical safeguards against supply-chain attacks and unauthorized data flow.

### Privacy First
- **Zero Phoning Home:** All auto-update features are **disabled by default**. We believe you should have total control over when and how your software changes.
- **Diagnostic Transparency:** System-level diagnostic checks (like the StatusKit check for hidden menubar icons on macOS Tahoe) are **disabled by default**. oMLX will not probe or attempt to fix system settings without your explicit opt-in.
- **No Analytics:** No telemetry, no tracking, and no external pings. Your data never leaves your machine.
- **Respect for the User:** My intentions are to provide a tool that respects your digital sovereignty. This version is a commitment to the community that your privacy is not a "feature"—it is the foundation.

### No "Pre-chewed" Binaries
- **Double Check for the Community:** We do not provide pre-compiled DMG or binary files in this repository. This is not a "trust issue" vs the community, but rather a deliberate "double check" to ensure we respect and protect every user.
- **Verified Privacy:** oMLX is built from the ground up to respect your privacy. By requiring a build from source, we provide a transparent way for you to verify that the software behaves exactly as documented—your data never leaves your machine.
- **User Control:** You have complete control over the build chain. You know exactly what code is being compiled and executed on your machine, eliminating the risk of compromised upstream binaries and encouraging the use of our secure, audited build scripts.

### Security Features
- **Isolated Build (`build_tahoe.sh`):** Includes a dedicated build script for **macOS Tahoe (26.x)** that creates a temporary `.build_venv` virtual environment. This prevents host Python pollution and ensures a deterministic build.
- **Supply Chain Protection:** Automatically runs `pip-audit` to scan all dependencies for known vulnerabilities. This protects against compromised Python packages (PyPI) before they ever reach your binary.
- **Reproducible & Secure:** Uses `venvstacks` with `exclude-newer` date-locking to freeze the dependency tree at a known-good state, shielding the project from malicious upstream updates.

### Build Instructions
1. **Requirements:** macOS 15.0+ (Sequoia/Tahoe), Apple Silicon (M1+), and Python 3.11 (recommended).
2. **Run the Secure Build:**
   ```bash
   ./build_tahoe.sh
   ```
3. **Output:** The final production-ready DMG will be located in `packaging/dist/`.

---

<p align="center">
  <img src="docs/images/omlx_dashboard.png" alt="oMLX Admin Dashboard" width="800">
</p>

> *Every LLM server I tried made me choose between convenience and control. I wanted to pin everyday models in memory, auto-swap heavier ones on demand, set context limits - and manage it all from a menu bar.*
>
> *oMLX persists KV cache across a hot in-memory tier and cold SSD tier - even when context changes mid-conversation, all past context stays cached and reusable across requests, making local LLMs practical for real coding work with tools like Claude Code. That's why I built it.*

## Install

### macOS App (Build from Source)

For your security and total control over the build chain, **no pre-built binaries or DMG files are provided**. This ensures that you are running code you have audited and built yourself on your own hardware, protecting you from potential "pre-chewed" binary compromises.

To create the macOS app:
1.  Clone this repository.
2.  Run the secure build script:
    ```bash
    ./build_tahoe.sh
    ```
3.  The final production-ready DMG will be located in `packaging/dist/`. Drag the generated `oMLX.app` to your Applications folder.

Requires macOS 15.0+ (Sequoia), Python 3.11+ (recommended), and Apple Silicon (M1/M2/M3/M4).

## Quickstart

### macOS App

Launch oMLX from your Applications folder. The Welcome screen guides you through three steps - model directory, server start, and first model download. That's it. To connect OpenClaw, OpenCode, or Codex, see [Integrations](#integrations).

<p align="center">
  <img src="docs/images/Screenshot 2026-02-10 at 00.36.32.png" alt="oMLX Welcome Screen" width="360">
  <img src="docs/images/Screenshot 2026-02-10 at 00.34.30.png" alt="oMLX Menubar" width="240">
</p>

### CLI

If you want to use the CLI, the `omlx` command is available inside the application bundle or can be run from the repository after building.

```bash
# Example if running from source directory after build_tahoe.sh
./.build_venv/bin/omlx serve --model-dir ~/models
```

## Features

Supports text LLMs, vision-language models (VLM), OCR models, embeddings, and rerankers on Apple Silicon.

### Admin Dashboard

Web UI at `/admin` for real-time monitoring, model management, chat, benchmark, and per-model settings. Supports English, Korean, Japanese, Chinese, and Russian. All CDN dependencies are vendored for fully offline operation.

<p align="center">
  <img src="docs/images/Screenshot 2026-02-10 at 00.45.34.png" alt="oMLX Admin Dashboard" width="720">
</p>

### Vision-Language Models

Run VLMs with the same continuous batching and tiered KV cache stack as text LLMs. Supports multi-image chat, base64/URL/file image inputs, and tool calling with vision context. OCR models (DeepSeek-OCR, DOTS-OCR, GLM-OCR) are auto-detected with optimized prompts.

### Tiered KV Cache (Hot + Cold)

Block-based KV cache management inspired by vLLM, with prefix sharing and Copy-on-Write. The cache operates across two tiers:

- **Hot tier (RAM)**: Frequently accessed blocks stay in memory for fast access.
- **Cold tier (SSD)**: When the hot cache fills up, blocks are offloaded to SSD in safetensors format. On the next request with a matching prefix, they're restored from disk instead of recomputed from scratch - even after a server restart.

<p align="center">
  <img src="docs/images/omlx_hot_cold_cache.png" alt="oMLX Hot & Cold Cache" width="720">
</p>

### Continuous Batching

Handles concurrent requests through mlx-lm's BatchGenerator. Max concurrent requests is configurable via CLI or admin panel.

### Claude Code Optimization

Context scaling support for running smaller context models with Claude Code. Scales reported token counts so that auto-compact triggers at the right timing, and SSE keep-alive prevents read timeouts during long prefill.

### Multi-Model Serving

Load LLMs, VLMs, embedding models, and rerankers within the same server. Models are managed through a combination of automatic and manual controls:

- **LRU eviction**: Least-recently-used models are evicted automatically when memory runs low.
- **Manual load/unload**: Interactive status badges in the admin panel let you load or unload models on demand.
- **Model pinning**: Pin frequently used models to keep them always loaded.
- **Per-model TTL**: Set an idle timeout per model to auto-unload after a period of inactivity.
- **Process memory enforcement**: Total memory limit (default: system RAM - 8GB) prevents system-wide OOM.

### Per-Model Settings

Configure sampling parameters, chat template kwargs, TTL, model alias, model type override, and more per model directly from the admin panel. Changes apply immediately without server restart.

- **Model alias**: set a custom API-visible name. `/v1/models` returns the alias, and requests accept both the alias and directory name.
- **Model type override**: manually set a model as LLM or VLM regardless of auto-detection.

<p align="center">
  <img src="docs/images/omlx_ChatTemplateKwargs.png" alt="oMLX Chat Template Kwargs" width="480">
</p>

### Built-in Chat

Chat directly with any loaded model from the admin panel. Supports conversation history, model switching, dark mode, reasoning model output, and image upload for VLM/OCR models.

<p align="center">
  <img src="docs/images/ScreenShot_2026-03-14_104350_610.png" alt="oMLX Chat" width="720">
</p>


### Model Downloader

Search and download MLX models from HuggingFace directly in the admin dashboard. Browse model cards, check file sizes, and download with one click.

<p align="center">
  <img src="docs/images/downloader_omlx.png" alt="oMLX Model Downloader" width="720">
</p>

### Integrations

Set up OpenClaw, OpenCode, Codex, and Pi directly from the admin dashboard with a single click. No manual config editing required.

<p align="center">
  <img src="docs/images/omlx_integrations.png" alt="oMLX Integrations" width="720">
</p>

### Performance Benchmark

One-click benchmarking from the admin panel. Measures prefill (PP) and text generation (TG) tokens per second, with partial prefix cache hit testing for realistic performance numbers.

<p align="center">
  <img src="docs/images/benchmark_omlx.png" alt="oMLX Benchmark Tool" width="720">
</p>

### macOS Menubar App

Native PyObjC menubar app (not Electron). Start, stop, and monitor the server without opening a terminal. Includes persistent serving stats (survives restarts), auto-restart on crash, and a privacy-respecting **optional** in-app update check (disabled by default).

<p align="center">
  <img src="docs/images/Screenshot 2026-02-10 at 00.51.54.png" alt="oMLX Menubar Stats" width="400">
</p>

### API Compatibility

Drop-in replacement for OpenAI and Anthropic APIs. Supports streaming usage stats (`stream_options.include_usage`), Anthropic adaptive thinking, and vision inputs (base64, URL).

| Endpoint | Description |
|----------|-------------|
| `POST /v1/chat/completions` | Chat completions (streaming) |
| `POST /v1/completions` | Text completions (streaming) |
| `POST /v1/messages` | Anthropic Messages API |
| `POST /v1/embeddings` | Text embeddings |
| `POST /v1/rerank` | Document reranking |
| `GET /v1/models` | List available models |

### Tool Calling & Structured Output

Supports all function calling formats available in mlx-lm, JSON schema validation, and MCP tool integration. Tool calling requires the model's chat template to support the `tools` parameter. The following model families are auto-detected via mlx-lm's built-in tool parsers:

| Model Family | Format |
|---|---|
| Llama, Qwen, DeepSeek, etc. | JSON `<tool_call>` |
| Qwen3.5 Series | XML `<function=...>` |
| Gemma | `<start_function_call>` |
| GLM (4.7, 5) | `<arg_key>/<arg_value>` XML |
| MiniMax | Namespaced `<minimax:tool_call>` |
| Mistral | `[TOOL_CALLS]` |
| Kimi K2 | `<\|tool_calls_section_begin\|>` |
| Longcat | `<longcat_tool_call>` |

Models not listed above may still work if their chat template accepts `tools` and their output uses a recognized `<tool_call>` XML format. For tool-enabled streaming, assistant text is emitted incrementally while known tool-call control markup is suppressed from visible content; structured tool calls are emitted after parsing the completed turn.

## Models

Point `--model-dir` at a directory containing MLX-format model subdirectories. Two-level organization folders (e.g., `mlx-community/model-name/`) are also supported.

```
~/models/
├── Step-3.5-Flash-8bit/
├── Qwen3-Coder-Next-8bit/
├── gpt-oss-120b-MXFP4-Q8/
├── Qwen3.5-122B-A10B-4bit/
└── bge-m3/
```

Models are auto-detected by type. You can also download models directly from the admin dashboard.

| Type | Models |
|------|--------|
| LLM | Any model supported by [mlx-lm](https://github.com/ml-explore/mlx-lm) |
| VLM | Qwen3.5 Series, GLM-4V, Pixtral, and other [mlx-vlm](https://github.com/Blaizzy/mlx-vlm) models |
| OCR | DeepSeek-OCR, DOTS-OCR, GLM-OCR |
| Embedding | BERT, BGE-M3, ModernBERT |
| Reranker | ModernBERT, XLM-RoBERTa |

## CLI Configuration

```bash
# Memory limit for loaded models
omlx serve --model-dir ~/models --max-model-memory 32GB

# Process-level memory limit (default: auto = RAM - 8GB)
omlx serve --model-dir ~/models --max-process-memory 80%

# Enable SSD cache for KV blocks
omlx serve --model-dir ~/models --paged-ssd-cache-dir ~/.omlx/cache

# Set in-memory hot cache size
omlx serve --model-dir ~/models --hot-cache-max-size 20%

# Adjust max concurrent requests (default: 8)
omlx serve --model-dir ~/models --max-concurrent-requests 16

# With MCP tools
omlx serve --model-dir ~/models --mcp-config mcp.json

# HuggingFace mirror endpoint (for restricted regions)
omlx serve --model-dir ~/models --hf-endpoint https://hf-mirror.com

# API key authentication
omlx serve --model-dir ~/models --api-key your-secret-key
# Localhost-only: skip verification via admin panel global settings

# Behavioral & Privacy Checks (Disabled by default)
omlx serve --check-updates
omlx serve --check-statuskit
```

All settings can also be configured from the web admin panel at `/admin`. Settings are persisted to `~/.omlx/settings.json`, and CLI flags take precedence.

<details>
<summary>Architecture</summary>

```
FastAPI Server (OpenAI / Anthropic API)
    │
    ├── EnginePool (multi-model, LRU eviction, TTL, manual load/unload)
    │   ├── BatchedEngine (LLMs, continuous batching)
    │   ├── VLMEngine (vision-language models)
    │   ├── EmbeddingEngine
    │   └── RerankerEngine
    │
    ├── ProcessMemoryEnforcer (total memory limit, TTL checks)
    │
    ├── Scheduler (FCFS, configurable concurrency)
    │   └── mlx-lm BatchGenerator
    │
    └── Cache Stack
        ├── PagedCacheManager (GPU, block-based, CoW, prefix sharing)
        ├── Hot Cache (in-memory tier, write-back)
        └── PagedSSDCacheManager (SSD cold tier, safetensors format)
```

</details>

## Development

### CLI Server

```bash
git clone https://github.com/jundot/omlx.git
cd omlx
pip install -e ".[dev]"
pytest -m "not slow"
```

### macOS App

Requires Python 3.11+ and [venvstacks](https://venvstacks.lmstudio.ai) (`pip install venvstacks`).

```bash
cd packaging

# Full build (venvstacks + app bundle + DMG)
python build.py

# Skip venvstacks (code changes only)
python build.py --skip-venv

# DMG only
python build.py --dmg-only
```

See [packaging/README.md](packaging/README.md) for details on the app bundle structure and layer configuration.

## Contributing

Contributions are welcome! See [Contributing Guide](docs/CONTRIBUTING.md) for details.

- Bug fixes and improvements
- Performance optimizations
- Documentation improvements

## License

[Apache 2.0](LICENSE)

## Acknowledgments

- [MLX](https://github.com/ml-explore/mlx) and [mlx-lm](https://github.com/ml-explore/mlx-lm) by Apple
- [mlx-vlm](https://github.com/Blaizzy/mlx-vlm) - Vision-language model inference on Apple Silicon
- [venvstacks](https://venvstacks.lmstudio.ai) - Portable Python environment layering for the macOS app bundle
- [mlx-embeddings](https://github.com/Blaizzy/mlx-embeddings) - Embedding model support for Apple Silicon
- [dflash-mlx](https://github.com/bstnxbt/dflash-mlx) - Block diffusion speculative decoding on Apple Silicon
