#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
CLI for oMLX.

Commands:
    omlx serve --model-dir /path/to/models    Start multi-model server

Usage:
    # Multi-model serving
    omlx serve --model-dir /path/to/models

    # With pinned models
    omlx serve --model-dir /path/to/models --pin llama-3b,qwen-7b
"""

import argparse
import errno
import faulthandler
import math
import socket
import sys

from ._version import __version__


def _positive_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be a finite number greater than 0")
    return parsed


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be an integer greater than 0")
    return parsed


def _has_cli_overrides(args) -> bool:
    """Check if CLI args contain non-default values that should be saved.

    All argparse defaults are None, so `is not None` means the user
    explicitly passed the flag on the command line.
    """
    persisted_fields = (
        "model_dir",
        "port",
        "host",
        "log_level",
        "sse_keepalive_mode",
        "max_audio_upload_size",
        "max_image_upload_size",
        "max_image_side_length",
        "max_concurrent_requests",
        "embedding_batch_size",
        "memory_guard",
        "memory_guard_gb",
        "paged_ssd_cache_dir",
        "paged_ssd_cache_max_size",
        "hot_cache_max_size",
        "hot_cache_write_through",
        "initial_cache_blocks",
        "mcp_config",
        "hf_endpoint",
        "hf_cache_enabled",
        "ms_endpoint",
        "http_proxy",
        "https_proxy",
        "no_proxy",
        "ca_bundle",
    )
    if any(getattr(args, field, None) is not None for field in persisted_fields):
        return True

    # --no-cache is the only persistable boolean flag with a False default.
    return bool(getattr(args, "no_cache", False))


def _migrate_saved_network_auth(settings, args) -> None:
    import json
    import os
    import tempfile
    from pathlib import Path

    from .utils.network import is_valid_bind_host, network_auth_error

    if getattr(args, "host", None) is not None or os.environ.get("OMLX_HOST"):
        return
    path = settings.base_path / "settings.json"
    if not path.exists():
        return
    host = settings.server.host
    if not isinstance(host, str) or not all(
        is_valid_bind_host(part.strip()) for part in host.split(",")
    ):
        return
    if not network_auth_error(
        host, settings.auth.api_key, settings.auth.skip_api_key_verification
    ):
        return

    settings.server.host = "127.0.0.1"
    if settings.validate():
        settings.server.host = host
        return

    message = (
        f"The saved server address ({host}) was changed to 127.0.0.1 because "
        "API key authentication is required for access from other devices. "
        "The server is now limited to this Mac. Your other settings and models "
        "have been preserved. To allow access from other devices, set an API "
        "key and enable authentication in Settings, then change the server address."
    )
    notice_path = os.environ.get("OMLX_STARTUP_NOTICE_PATH")
    if not notice_path:
        warning = message.replace("was changed", "will be changed").replace(
            "is now limited", "will be limited"
        )
        print(f"Warning: {warning}", flush=True)
        try:
            input("Press Enter to continue, or Ctrl+C to cancel. ")
        except (EOFError, KeyboardInterrupt):
            print("\nStartup canceled. Settings have not been changed.", flush=True)
            raise SystemExit(1) from None

    # Preserve unknown settings and avoid persisting environment overrides.
    data = json.loads(path.read_text(encoding="utf-8"))
    data.setdefault("server", {})["host"] = settings.server.host
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False
        ) as output:
            temporary = Path(output.name)
            json.dump(data, output, indent=2)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)

    if notice_path:
        print(f"Warning: {message}", flush=True)

        notice = Path(notice_path)
        temporary = notice.with_suffix(".tmp")
        try:
            temporary.write_text(message, encoding="utf-8")
            os.replace(temporary, notice)
        finally:
            temporary.unlink(missing_ok=True)


def _is_local_address(host: str) -> bool:
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    with socket.socket(family) as probe:
        try:
            probe.bind((host, 0))
        except OSError as exc:
            return exc.errno != errno.EADDRNOTAVAIL
    return True


def serve_command(args):
    """Start the OpenAI-compatible multi-model server."""
    import logging
    import os
    import uvicorn

    from ._version import __version__
    from . import process_title
    from .integrations.base import bracket_host
    from .settings import burst_decode_env, init_settings
    from .logging_config import (
        AdminStatsAccessFilter,
        configure_file_logging,
        console_log_level,
        is_console_handler,
    )

    process_title.set_process_title()

    try:
        from ._build_info import build_number
    except ImportError:
        build_number = None

    # Print version banner
    print("\033[33moMLX - LLM inference, optimized for your Mac\033[0m")
    print("\033[33m├─ https://github.com/jundot/omlx\033[0m")
    if build_number:
        print(f"\033[33m├─ Version: {__version__}\033[0m")
        print(f"\033[33m└─ Build: {build_number}\033[0m")
    else:
        print(f"\033[33m└─ Version: {__version__}\033[0m")
    print()

    # Initialize global settings first (to get log_level from file if not specified)
    settings = init_settings(base_path=args.base_path, cli_args=args)

    # The native ANE compile-cache gate reads this env var once, at the first
    # compile, so it must be exported before any engine loads. setdefault
    # keeps an explicit env override authoritative.
    if settings.cache.ane_compile_cache:
        os.environ.setdefault("OMLX_QWEN35_ANE_COMPILE_CACHE", "1")

    # Register TRACE level (5) — includes full message content
    TRACE = 5
    logging.addLevelName(TRACE, "TRACE")

    # Configure logging (use settings value which has proper priority)
    level_name = settings.server.log_level.upper()
    log_level = (
        TRACE if level_name == "TRACE" else getattr(logging, level_name, logging.INFO)
    )
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    for handler in logging.getLogger().handlers:
        if is_console_handler(handler):
            handler.setLevel(console_log_level(log_level))
    # Set omlx loggers
    for name in [
        "omlx",
        "omlx.scheduler",
        "omlx.paged_ssd_cache",
        "omlx.memory_monitor",
        "omlx.paged_cache",
        "omlx.prefix_cache",
        "omlx.engine_pool",
        "omlx.model_discovery",
    ]:
        logging.getLogger(name).setLevel(log_level)

    # Suppress repetitive admin stats access logs
    logging.getLogger("uvicorn.access").addFilter(AdminStatsAccessFilter())

    # Suppress noisy third-party loggers unless trace level
    if log_level > TRACE:
        logging.getLogger("httpcore").setLevel(logging.INFO)
        logging.getLogger("httpx").setLevel(logging.INFO)

    # Ensure required directories exist
    settings.ensure_directories()

    # Apply HuggingFace endpoint if configured
    if settings.huggingface.endpoint:
        os.environ["HF_ENDPOINT"] = settings.huggingface.endpoint

    # Apply ModelScope endpoint if configured
    if settings.modelscope.endpoint:
        os.environ["MODELSCOPE_DOMAIN"] = settings.modelscope.endpoint

    # Apply proxy/TLS settings if configured
    if settings.network.http_proxy:
        os.environ["HTTP_PROXY"] = settings.network.http_proxy
        os.environ["http_proxy"] = settings.network.http_proxy
    if settings.network.https_proxy:
        os.environ["HTTPS_PROXY"] = settings.network.https_proxy
        os.environ["https_proxy"] = settings.network.https_proxy
    if settings.network.no_proxy:
        os.environ["NO_PROXY"] = settings.network.no_proxy
        os.environ["no_proxy"] = settings.network.no_proxy
    if settings.network.ca_bundle:
        os.environ["REQUESTS_CA_BUNDLE"] = settings.network.ca_bundle
        os.environ["SSL_CERT_FILE"] = settings.network.ca_bundle

    # Seed Burst Decode env vars so EngineConfig picks up the saved mode at
    # engine construction (no restart needed when the mode changes later).
    for _key, _value in burst_decode_env(settings.server.burst_decode_mode).items():
        os.environ[_key] = _value

    # Validate before persisting CLI overrides, so invalid flags never poison
    # settings.json.
    try:
        _migrate_saved_network_auth(settings, args)
    except (OSError, ValueError) as error:
        print(f"Configuration error: {error}")
        sys.exit(1)
    errors = settings.validate()
    if errors:
        for error in errors:
            print(f"Configuration error: {error}")
        sys.exit(1)

    # Save CLI args to settings.json if non-default values provided
    if _has_cli_overrides(args):
        try:
            settings.save_cli_overrides(args)
            print("Saved CLI arguments to settings.json")
        except Exception as e:
            print(f"Warning: Failed to save settings: {e}")

    # Configure file logging (writes to {base_path}/logs/server.log)
    log_dir = settings.logging.get_log_dir(settings.base_path)
    configure_file_logging(
        log_dir=log_dir,
        level=settings.server.log_level,
        include_request_id=True,
        retention_days=settings.logging.retention_days,
    )
    print(f"Log directory: {log_dir}")

    # Enable native crash diagnostics (SIGABRT, SIGSEGV, SIGFPE, SIGBUS).
    # On Metal/MLX crashes (#511, #520), this dumps all Python thread
    # tracebacks to the server log before the process terminates.
    crash_log_path = log_dir / "crash.log"
    _crash_file = open(crash_log_path, "a")
    faulthandler.enable(file=_crash_file, all_threads=True)

    # Bind the socket before importing/initializing the server. Uvicorn's
    # normal startup runs ASGI lifespan before binding host/port, which means
    # pinned models can be preloaded before a port conflict is detected.
    bind_hosts = [h.strip() for h in settings.server.host.split(",") if h.strip()]
    # A secondary address such as a VPN IP can be missing until its link is up.
    for h in bind_hosts[1:]:
        if not _is_local_address(h):
            print(f"Warning: skipping {h}, it is not assigned to any local interface")
            bind_hosts.remove(h)
    for h in bind_hosts:
        print(f"Binding server at http://{bracket_host(h)}:{settings.server.port}")
    # uvicorn does not support "trace" — map to "debug" for its internal logging
    uvicorn_level = (
        "debug" if settings.server.log_level == "trace" else settings.server.log_level
    )
    # Only show access logs at trace level
    show_access_log = settings.server.log_level == "trace"
    uvicorn_config = uvicorn.Config(
        "omlx.server:app",
        host=bind_hosts[0],
        port=settings.server.port,
        log_level=uvicorn_level,
        access_log=show_access_log,
    )
    # Bind a socket per host so an occupied port fails fast before model preload.
    # uvicorn.Server.run(sockets=[...]) accepts a list and listens on all of them.
    serve_sockets = [uvicorn_config.bind_socket()]
    for h in bind_hosts[1:]:
        extra_cfg = uvicorn.Config(
            "omlx.server:app",
            host=h,
            port=settings.server.port,
            log_level=uvicorn_level,
            access_log=show_access_log,
        )
        serve_sockets.append(extra_cfg.bind_socket())

    # Read by omlx.server at import time.
    if getattr(args, "headless", False):
        os.environ["OMLX_HEADLESS"] = "1"

    try:
        # Import server and config after the port is known to be available.
        from .server import init_server
        from .config import parse_size

        model_dirs = settings.get_effective_model_dirs()
        print(f"Base path: {settings.base_path}")
        print(f"Model directories: {', '.join(str(d) for d in model_dirs)}")
        # State first: a bare tier line reads as "this is enforced" even when
        # the guard is off, and with it off the tier governs nothing.
        if settings.memory.prefill_memory_guard:
            print(f"Memory guard: on (tier: {settings.memory.memory_guard_tier})")
        else:
            print("Memory guard: off")

        # Store MCP config path for FastAPI startup
        # Priority: CLI arg > settings.json
        mcp_config = args.mcp_config or settings.mcp.config_path
        if mcp_config:
            print(f"MCP config: {mcp_config}")
            os.environ["OMLX_MCP_CONFIG"] = mcp_config

        # Build scheduler config for BatchedEngine. Settings already include
        # the CLI args and leave the SSD cache dir unset when it is off.
        scheduler_config = settings.to_scheduler_config()
        if args.no_cache:
            scheduler_config.paged_ssd_cache_dir = None
        paged_ssd_cache_dir = scheduler_config.paged_ssd_cache_dir
        if paged_ssd_cache_dir:
            cache_max_size_bytes = scheduler_config.paged_ssd_cache_max_size
        else:
            scheduler_config.paged_ssd_cache_max_size = 0
            cache_max_size_bytes = 0

        # Hot cache: CLI arg > settings
        if paged_ssd_cache_dir:
            if args.hot_cache_max_size:
                hot_cache_max_bytes = parse_size(args.hot_cache_max_size)
            else:
                hot_cache_max_bytes = settings.cache.get_hot_cache_max_size_bytes()
            scheduler_config.hot_cache_max_size = hot_cache_max_bytes
        else:
            scheduler_config.hot_cache_max_size = 0

        # Write-through: explicit CLI flag > settings file (already mapped by
        # settings.to_scheduler_config()).
        if getattr(args, "hot_cache_write_through", None) is not None:
            scheduler_config.hot_cache_write_through = bool(
                args.hot_cache_write_through
            )

        if args.no_cache:
            print(
                "Mode: Multi-model serving (no oMLX cache, mlx-lm BatchGenerator only)"
            )
        elif paged_ssd_cache_dir:
            print("Mode: Multi-model serving (continuous batching + paged SSD cache)")
            # Format cache size for display
            cache_max_size_display = f"{cache_max_size_bytes / (1024**3):.1f}GB"
            if scheduler_config.paged_ssd_cache_auto_size:
                cache_max_size_display = f"auto, current limit {cache_max_size_display}"
            print(
                f"paged SSD cache: {paged_ssd_cache_dir} (max: {cache_max_size_display})"
            )
            if scheduler_config.hot_cache_max_size > 0:
                hot_display = f"{scheduler_config.hot_cache_max_size / (1024**3):.1f}GB"
                print(f"Hot cache: {hot_display} (in-memory)")
        else:
            print("Mode: Multi-model serving (continuous batching, no cache)")
            if settings.cache.is_ssd_cache_dir_unavailable():
                print(
                    "paged SSD cache: off until restart, directory unavailable: "
                    f"{settings.cache.unavailable_ssd_cache_dir}"
                )

        # Set MLX buffer cache limit high to prevent the allocator from
        # immediately releasing Metal buffers when the cache is full.
        # Without this, allocator::free() can call buf->release() while the
        # GPU is still using the buffer, causing kernel panics on M4.
        # With a large cache limit, freed buffers always stay in the pool
        # and are only released via mx.clear_cache() (which we protect
        # with mx.synchronize()). See issue #300.
        import mlx.core as mx

        total_mem = mx.device_info().get("memory_size", 0)
        if total_mem > 0:
            mx.set_cache_limit(total_mem)

        # Initialize server
        # Note: pinned_models and default_model are managed via admin page (model_settings.json)
        # Sampling parameters (max_tokens, temperature, etc.) are per-model settings
        init_server(
            model_dirs=[str(d) for d in model_dirs],
            scheduler_config=scheduler_config,
            api_key=settings.auth.api_key,
            global_settings=settings,
        )

        for h in bind_hosts:
            print(f"Starting server at http://{bracket_host(h)}:{settings.server.port}")
        try:
            uvicorn.Server(uvicorn_config).run(sockets=serve_sockets)
        except KeyboardInterrupt:
            pass
    finally:
        # Uvicorn closes sockets during normal shutdown; this covers failures
        # after bind succeeds but before the server takes ownership.
        for sock in serve_sockets:
            sock.close()


def launch_command(args, extra_args: list[str] | None = None):
    """Launch an external tool integrated with oMLX.

    extra_args are unknown CLI tokens forwarded to the underlying tool binary
    (e.g. ``-r`` / ``--resume <id>`` for Claude Code).
    """
    import requests

    from .integrations import IntegrationContext, get_integration, list_integrations
    from .integrations.base import bracket_host
    from .settings import GlobalSettings

    def _optional_str(value) -> str | None:
        return value if isinstance(value, str) and value else None

    tool_name = args.tool

    if tool_name == "list":
        print("Available integrations:")
        for integ in list_integrations():
            installed = "installed" if integ.is_installed() else "not installed"
            print(f"  {integ.name:12s} {integ.display_name} ({installed})")
        return

    integration = get_integration(tool_name)
    if integration is None:
        print(f"Unknown integration: {tool_name}")
        print("Available: " + ", ".join(i.name for i in list_integrations()))
        sys.exit(1)

    # Resolve host/port: CLI args > env vars > settings.json > defaults
    settings = GlobalSettings.load()
    host = args.host or settings.server.host
    port = args.port or settings.server.port

    # host may be a comma-separated list of bind addresses; pick the first one
    # for connecting. Wildcard addresses (0.0.0.0, ::) are valid bind targets
    # but not connectable — fall back to localhost in that case.
    first_bind = [h.strip() for h in host.split(",") if h.strip()][0] if host else ""
    connect_host = (
        first_bind if first_bind not in ("", "0.0.0.0", "::") else "127.0.0.1"
    )
    base_url = f"http://{bracket_host(connect_host)}:{port}"

    # Claude Desktop resolves tier aliases server-side, so it needs no
    # launch-time model — but it does need the desktop switch. The toggle
    # lives in the dashboard; the CLI only points the user there.
    if tool_name == "claude_desktop" and not bool(
        getattr(getattr(settings, "claude_code", None), "desktop_enabled", False)
    ):
        print("Claude Desktop mode is not enabled in oMLX settings.")
        print(
            "Enable it in the oMLX dashboard (Integrations → Claude Desktop) "
            "and re-run this command."
        )
        sys.exit(1)

    # Check if oMLX server is running
    try:
        resp = requests.get(f"{base_url}/health", timeout=3)
        resp.raise_for_status()
    except Exception:
        print(f"oMLX server is not running at {base_url}")
        print("Start the server first: omlx start")
        sys.exit(1)
    try:
        default_model = _optional_str(resp.json().get("default_model")) or ""
    except (ValueError, AttributeError):
        default_model = ""

    # Get API key: CLI args > settings.json > empty
    api_key = getattr(args, "api_key", None) or settings.auth.api_key or ""

    claude_settings = getattr(settings, "claude_code", None)
    cli_opus_model = _optional_str(getattr(args, "opus_model", None))
    cli_sonnet_model = _optional_str(getattr(args, "sonnet_model", None))
    cli_haiku_model = _optional_str(getattr(args, "haiku_model", None))
    settings_opus_model = _optional_str(getattr(claude_settings, "opus_model", None))
    settings_sonnet_model = _optional_str(
        getattr(claude_settings, "sonnet_model", None)
    )
    settings_haiku_model = _optional_str(getattr(claude_settings, "haiku_model", None))
    opus_model = cli_opus_model or settings_opus_model
    sonnet_model = cli_sonnet_model or settings_sonnet_model
    haiku_model = cli_haiku_model or settings_haiku_model

    # Build headers for authenticated requests
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    # Pre-fetch model status (context_window, max_tokens, model_type per model)
    models_status_map: dict[str, dict] = {}
    try:
        resp = requests.get(f"{base_url}/v1/models/status", headers=headers, timeout=5)
        if resp.ok:
            for m in resp.json().get("models", []):
                if m_id := m.get("id"):
                    models_status_map[m_id] = m
                if model_alias := m.get("model_alias"):
                    models_status_map[model_alias] = m
    except Exception:
        pass

    # Determine model. Explicit CLI tier flags bypass the picker; otherwise always
    # prompt interactively so the user's selection is honoured. Integrations
    # that resolve models server-side (requires_model_selection=False) skip
    # this block.
    model = args.model
    if not model and not integration.requires_model_selection:
        # The integration registers the server's whole model catalog, so
        # there is nothing to pick here; the optional per-tool default (or
        # --model) only seeds the tool's own default model.
        setting_tool = "codex" if tool_name == "codex_app" else tool_name
        model = (
            _optional_str(getattr(settings.integrations, f"{setting_tool}_model", None))
            or ""
        )
    if not model and (cli_opus_model or cli_sonnet_model or cli_haiku_model):
        model = cli_sonnet_model or cli_opus_model or cli_haiku_model or ""
    elif not model and integration.requires_model_selection:
        # Fetch available models from server
        try:
            resp = requests.get(f"{base_url}/v1/models", headers=headers, timeout=5)
            resp.raise_for_status()
            data = resp.json()
            models = [
                m["id"]
                for m in data.get("data", [])
                if m.get("model_type") in ("llm", "vlm", None)
            ]
        except Exception:
            models = []

        if not models:
            print("No models available. Load a model first.")
            sys.exit(1)

        if len(models) == 1:
            model = models[0]
            print(f"Using model: {model}")
        else:
            models_info_list = [
                {"id": m_id, **models_status_map.get(m_id, {})} for m_id in models
            ]
            model = integration.select_model(models_info_list, integration.display_name)

    # Check if tool is installed
    if not integration.is_installed():
        print(f"{integration.display_name} is not installed.")
        print(f"Install: {integration.install_hint}")
        sys.exit(1)

    # Tier precedence: explicit tier flag > saved claude_code tier setting >
    # the model picked (or auto-selected) above. The picker only chooses the
    # default model; tiers configured on the Claude Code settings page keep
    # their role, otherwise the three persisted selections would be silently
    # replaced by one model on every interactive launch (#3543). Roles without
    # a saved model fall back to the picked model in the integration.

    # Enforce Claude Code's model requirements after all interactive,
    # automatic, and explicit model paths have resolved. The picker also marks
    # disabled models, but this central check prevents --model and tier flags
    # from bypassing the same restriction.
    if tool_name == "claude":
        from .integrations.claude import claude_code_model_disabled_reason

        models_to_validate = [
            ("", model),
            ("Opus tier ", opus_model),
            ("Sonnet tier ", sonnet_model),
            ("Haiku tier ", haiku_model),
        ]
        validated_models: set[str] = set()
        for role, model_id in models_to_validate:
            if not model_id or model_id in validated_models:
                continue
            validated_models.add(model_id)
            disabled_reason = claude_code_model_disabled_reason(
                {"id": model_id, **models_status_map.get(model_id, {})}
            )
            if disabled_reason:
                print(
                    f"Cannot launch {integration.display_name} with "
                    f"{role}model '{model_id}'."
                )
                print(disabled_reason)
                print(
                    "Choose a model with at least 48K context or increase its "
                    "configured max_context_window."
                )
                sys.exit(1)

    # Resolve model limits from pre-fetched status
    model_info = models_status_map.get(model, {})
    context_window = model_info.get("max_context_window")
    if tool_name == "claude":
        # Claude's context overrides are process-wide, including tier switches
        # and subagents. Do not advertise more than any configured model allows.
        context_windows = [
            info["max_context_window"]
            for model_id in (model, opus_model, sonnet_model, haiku_model)
            if (info := models_status_map.get(model_id, {}))
            and isinstance(info.get("max_context_window"), int)
            and info["max_context_window"] > 0
        ]
        context_window = min(context_windows) if context_windows else None
    ctx = IntegrationContext(
        host=connect_host,
        port=port,
        api_key=api_key,
        model=model,
        default_model=default_model,
        opus_model=opus_model if tool_name == "claude" else None,
        sonnet_model=sonnet_model if tool_name == "claude" else None,
        haiku_model=haiku_model if tool_name == "claude" else None,
        context_window=context_window,
        max_tokens=model_info.get("max_tokens"),
        model_type=model_info.get("model_type"),
        reasoning=model_info.get("enable_thinking"),
        models_status_map=models_status_map,
        tools_profile=getattr(args, "tools_profile", "coding"),
        extra_args=tuple(extra_args or ()),
        cross_session=getattr(args, "cross_session", False),
    )

    # Launch
    if model:
        print(f"Launching {integration.display_name} with model {model}...")
    else:
        print(f"Launching {integration.display_name}...")
    integration.launch(ctx)


def _app_control_socket_path():
    from pathlib import Path

    return Path.home() / "Library" / "Application Support" / "oMLX" / "control.sock"


def _app_bundle_path():
    from pathlib import Path

    from .utils.install import get_app_bundle_cli_path

    cli_path = get_app_bundle_cli_path()
    try:
        return cli_path.parents[2]
    except IndexError:
        return Path("/Applications/oMLX.app")


def _open_macos_app() -> None:
    import subprocess

    app_path = _app_bundle_path()
    subprocess.run(
        ["/usr/bin/open", "-gj", str(app_path)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )


def _send_app_control(command: str, timeout: float = 2.0) -> dict:
    import json
    import socket

    sock_path = _app_control_socket_path()
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout)
        sock.connect(str(sock_path))
        sock.sendall(json.dumps({"command": command}).encode("utf-8") + b"\n")
        chunks: list[bytes] = []
        while True:
            chunk = sock.recv(4096)
            if not chunk:
                break
            chunks.append(chunk)
            if b"\n" in chunk:
                break
    raw = b"".join(chunks).split(b"\n", 1)[0]
    return json.loads(raw.decode("utf-8"))


def _send_app_control_with_launch(command: str, timeout: float) -> dict:
    import time

    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    _open_macos_app()
    while time.monotonic() < deadline:
        try:
            return _send_app_control(command)
        except OSError as exc:
            last_error = exc
            time.sleep(0.2)
    raise RuntimeError(f"Could not reach oMLX.app control socket: {last_error}")


def _wait_app_control_state(states: set[str], timeout: float) -> dict:
    import time

    deadline = time.monotonic() + timeout
    last: dict = {}
    while time.monotonic() < deadline:
        last = _send_app_control("status")
        if last.get("state") in states:
            return last
        time.sleep(0.5)
    return last


def _run_brew_services(command: str) -> int:
    import shutil
    import subprocess

    brew = shutil.which("brew")
    if not brew:
        print("Homebrew is not available on PATH.")
        return 1
    result = subprocess.run([brew, "services", command, "omlx"])
    return result.returncode


def lifecycle_command(args) -> int:
    """Run background lifecycle commands for the current installation."""
    from .utils.install import is_app_bundle, is_homebrew

    command = args.command
    timeout = getattr(args, "timeout", 60.0)
    no_wait = getattr(args, "no_wait", False)

    if is_app_bundle():
        try:
            if command == "stop":
                try:
                    response = _send_app_control(command)
                except OSError:
                    print("oMLX stopped")
                    return 0
            else:
                response = _send_app_control_with_launch(command, timeout=timeout)
            if not response.get("ok"):
                print(response.get("message") or f"oMLX {command} failed")
                return 1

            if command in {"start", "restart"} and not no_wait:
                response = _wait_app_control_state({"running", "unresponsive"}, timeout)
                if response.get("state") not in {"running", "unresponsive"}:
                    print(
                        f"oMLX server is {response.get('state', 'unknown')} "
                        f"after {int(timeout)}s."
                    )
                    return 1

            if command == "stop":
                print("oMLX stopped")
            elif command == "start":
                print(
                    f"oMLX server {response.get('state')} on port {response.get('port')}"
                )
            elif command == "restart":
                print(f"oMLX server restarted on port {response.get('port')}")
            return 0
        except Exception as exc:
            print(f"Failed to control oMLX.app: {exc}")
            return 1

    if is_homebrew():
        mapping = {"start": "start", "stop": "stop", "restart": "restart"}
        return _run_brew_services(mapping[command])

    if command == "start":
        print("Background start is available for the macOS app and Homebrew installs.")
        print("For this install, run foreground server mode with: omlx serve")
    else:
        print("Background stop/restart requires the macOS app or Homebrew service.")
    return 1


def diagnose_menubar() -> int:
    """Diagnose why the oMLX menubar icon might be missing.

    Reports macOS version, app install path, running menubar process, and the
    most recent visibility warning from the log. Prints manual recovery steps
    since Tahoe's ControlCenter doesn't expose a public API to re-enable a
    hidden status item.
    """
    import platform
    import subprocess
    from pathlib import Path

    print("oMLX menubar diagnostics")
    print("=" * 40)

    mac_ver = platform.mac_ver()[0] or "unknown"
    print(f"macOS:          {mac_ver}")
    print("Bundle ID:      app.omlx")

    app_path = Path("/Applications/oMLX.app")
    print(f"App installed:  {'yes' if app_path.exists() else 'NO (install DMG first)'}")

    try:
        res = subprocess.run(
            ["pgrep", "-af", "oMLX"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        running = bool(res.stdout.strip())
        print(f"Menubar app:    {'running' if running else 'NOT running'}")
        if running:
            first_line = res.stdout.strip().splitlines()[0]
            pid = first_line.split()[0] if first_line else "?"
            print(f"PID:            {pid}")
    except (subprocess.SubprocessError, FileNotFoundError) as e:
        print(f"Menubar app:    check failed ({e})")

    # `menubar.log` is the Swift app's own visibility-probe log — every line
    # in it is relevant. `server.log` is the Python child's stdout/stderr, so
    # only lines that mention the menubar are worth pulling out of it.
    log_dir = Path.home() / "Library" / "Application Support" / "oMLX" / "logs"
    log_candidates = [(log_dir / "menubar.log", False), (log_dir / "server.log", True)]
    print(f"Log dir:        {log_dir}")

    hits: list[tuple[str, str]] = []
    for path, needs_filter in log_candidates:
        if not path.exists():
            continue
        try:
            with open(path, "rb") as f:
                f.seek(0, 2)
                size = f.tell()
                f.seek(max(0, size - 131072))
                tail = f.read().decode("utf-8", errors="replace")
        except OSError as e:
            print(f"Could not read {path.name}: {e}")
            continue
        for ln in tail.splitlines():
            if not ln.strip():
                continue
            if needs_filter and not (
                "menubar visibility probe" in ln
                or "NSStatusItem" in ln
                or "ControlCenter" in ln
                or "Menu Bar" in ln
            ):
                continue
            hits.append((path.name, ln))

    if hits:
        print("\nRecent visibility log entries (last 10):")
        for src, ln in hits[-10:]:
            print(f"  [{src}] {ln}")
    else:
        print("\nNo visibility log entries found (app may not have probed yet).")

    print()
    print("If the icon is missing on macOS Tahoe (26.x):")
    print("  1. In the oMLX app: Settings > Appearance > Menu Bar Icon > Restore")
    print("  2. Or turn it back on in System Settings > Menu Bar")
    print(
        "     open 'x-apple.systempreferences:com.apple.ControlCenter-Settings.extension?MenuBar'"
    )
    print("  3. If oMLX isn't in the list, quit the app and relaunch oMLX.app")
    print()
    print("Note: Restore edits ControlCenter's own StatusKit approval, which")
    print("needs Full Disk Access. Without it, use the System Settings toggle.")
    print()
    print("If the log shows notch=true, the icon is behind the camera notch:")
    print("  the menu bar is full. Quit or turn off a few other menu bar icons,")
    print("  or use a menubar manager like Ice or Bartender.")
    return 0


def diagnose_command(args) -> int:
    """Dispatch 'omlx diagnose <target>' to the appropriate subcommand."""
    target = getattr(args, "target", None)
    if target == "menubar":
        return diagnose_menubar()
    print(f"Unknown diagnose target: {target}")
    print("Available: menubar")
    return 1


def cluster_command(args) -> int:
    """Run cluster diagnostics, collective checks, and shard planning."""
    import json

    action = getattr(args, "cluster_action", None)
    if action == "status":
        from .cluster.probe import collect_cluster_status, format_cluster_status

        try:
            status = collect_cluster_status(route_to=args.route_to)
        except ValueError as exc:
            print(f"Cluster status error: {exc}", file=sys.stderr)
            return 2
        if args.json:
            print(json.dumps(status.to_dict(), indent=2, sort_keys=True))
        else:
            print(format_cluster_status(status))
        return 0

    if action == "worker-smoke":
        from .cluster.supervisor import run_worker_smoke

        try:
            result = run_worker_smoke(timeout=args.timeout)
        except (OSError, RuntimeError, TimeoutError) as exc:
            print(f"Cluster worker smoke failed: {exc}", file=sys.stderr)
            return 1
        if args.json:
            print(json.dumps(result, indent=2, sort_keys=True))
        else:
            print("oMLX cluster worker smoke passed")
            print(f"Worker PID:  {result['worker_pid']}")
            print(f"Protocol:    {result['protocol_version']}")
            print(f"Round trip:  {result['elapsed_seconds']:.3f}s")
        return 0

    if action == "collective-smoke":
        from .cluster.collective import (
            CollectiveSmokeError,
            run_local_collective_smoke,
        )

        try:
            result = run_local_collective_smoke(timeout=args.timeout)
        except (CollectiveSmokeError, OSError, RuntimeError, ValueError) as exc:
            print(f"Cluster collective smoke failed: {exc}", file=sys.stderr)
            return 1
        if args.json:
            print(json.dumps(result, indent=2, sort_keys=True))
        else:
            print("oMLX local MLX collective smoke passed")
            print(f"Backend:     {result['backend']} (loopback only)")
            print(f"Ranks:       {result['rank_count']}")
            print(f"All-sum:     {result['expected_sum']}")
            print(f"MLX:         {result['mlx_version']}")
            print(f"Elapsed:     {result['elapsed_seconds']:.3f}s")
        return 0

    if action == "pipeline-smoke":
        from .cluster.collective import (
            CollectiveSmokeError,
            run_local_pipeline_smoke,
        )

        try:
            result = run_local_pipeline_smoke(timeout=args.timeout)
        except (CollectiveSmokeError, OSError, RuntimeError, ValueError) as exc:
            print(f"Cluster pipeline smoke failed: {exc}", file=sys.stderr)
            return 1
        if args.json:
            print(json.dumps(result, indent=2, sort_keys=True))
        else:
            print("oMLX unequal Nemotron-H pipeline smoke passed")
            print(f"Backend:     {result['backend']} (loopback only)")
            print(f"Ranks:       {result['rank_count']}")
            print(f"Checksum:    {result['ranks'][0]['checksum']}")
            print(f"Elapsed:     {result['elapsed_seconds']:.3f}s")
        return 0

    if action == "plan":
        import socket

        from .cluster.planner import (
            NodeBudget,
            PlanningError,
            format_shard_plan,
            locate_model_layout,
            plan_unequal_pipeline,
            synthetic_model_layout,
        )
        from .config import parse_size
        from .utils import hardware

        def parse_cluster_size(value: str) -> int:
            normalized = (
                value.strip()
                .upper()
                .replace("KIB", "KB")
                .replace("MIB", "MB")
                .replace("GIB", "GB")
                .replace("TIB", "TB")
            )
            size = parse_size(normalized)
            if size < 0:
                raise ValueError("sizes must be non-negative")
            return size

        try:
            reserve_bytes = parse_cluster_size(args.reserve)
            nodes = []
            for rank, definition in enumerate(args.node or []):
                node_id, separator, raw_size = definition.rpartition("=")
                if not separator or not node_id.strip() or not raw_size.strip():
                    raise ValueError("--node must use NAME=SIZE (for example studio=256GB)")
                nodes.append(
                    NodeBudget(
                        node_id=node_id.strip(),
                        capacity_bytes=parse_cluster_size(raw_size),
                        reserve_bytes=reserve_bytes,
                        rank=rank,
                    )
                )
            if not nodes:
                detected = hardware.detect_hardware()
                nodes.append(
                    NodeBudget(
                        node_id=socket.gethostname(),
                        capacity_bytes=detected.max_working_set_bytes,
                        reserve_bytes=reserve_bytes,
                        rank=0,
                    )
                )

            holder = None
            if args.model:
                # The Mac being planned for is often the one holding a single
                # stage, so ask each peer rather than assume this node can
                # read the whole model.
                holder = locate_model_layout(args.model, args.peer or [])
                model = holder.layout
            else:
                model = synthetic_model_layout(
                    total_weight_bytes=parse_cluster_size(args.model_size),
                    layer_count=args.layers,
                )
            plan = plan_unequal_pipeline(model, nodes)
        except (OSError, PlanningError, ValueError) as exc:
            print(f"Cluster planning failed: {exc}", file=sys.stderr)
            return 2

        if args.json:
            print(json.dumps(plan.to_dict(), indent=2, sort_keys=True))
        else:
            print(format_shard_plan(plan))
            if holder is not None and not holder.is_local:
                print(f"Measured:    {holder.node} (the node holding the model)")
        return 0

    print(
        "Unknown cluster action. Available: status, worker-smoke, "
        "collective-smoke, pipeline-smoke, plan",
        file=sys.stderr,
    )
    return 2


def steer_command(args) -> int:
    """Dispatch 'omlx steer <action>'."""
    action = getattr(args, "steer_action", None)
    if action == "build":
        return _steer_build(args)
    if action == "sweep":
        return _steer_sweep(args)
    if action == "verify":
        return _steer_verify(args)
    if action == "set":
        return _steer_set(args)
    if action == "show":
        return _steer_show(args)
    if action == "list":
        return _steer_list(args)
    if action == "export":
        return _steer_export(args)
    if action == "import":
        return _steer_import(args)
    print(
        "Unknown steer action. Available: build, sweep, set, show, "
        "list, import, export",
        file=sys.stderr,
    )
    return 2


def _steer_server(args) -> tuple[str, dict[str, str]]:
    """Resolve the server base URL and admin auth headers."""
    import requests

    from .integrations.base import bracket_host
    from .settings import GlobalSettings

    settings = GlobalSettings.load()
    host = args.host or settings.server.host
    port = args.port or settings.server.port
    # host may be a comma-separated list of bind addresses; pick the first
    # one for connecting and fall back to localhost for wildcard binds.
    first_bind = [h.strip() for h in host.split(",") if h.strip()][0] if host else ""
    connect_host = (
        first_bind if first_bind not in ("", "0.0.0.0", "::") else "127.0.0.1"
    )
    base_url = f"http://{bracket_host(connect_host)}:{port}"
    try:
        requests.get(f"{base_url}/health", timeout=3).raise_for_status()
    except Exception:
        print(f"oMLX server is not running at {base_url}")
        print("Start the server first: omlx start")
        sys.exit(1)
    # The admin API accepts only the main API key.
    api_key = getattr(args, "api_key", None) or settings.auth.api_key or ""
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    return base_url, headers


def _steer_model_url(base_url: str, model: str) -> str:
    from urllib.parse import quote

    return f"{base_url}/admin/api/models/{quote(str(model), safe='')}"


def _steer_fail(action: str, resp) -> int:
    try:
        detail = resp.json().get("detail", resp.text)
    except Exception:
        detail = resp.text
    print(
        f"steer {action} failed: HTTP {resp.status_code}: {detail}",
        file=sys.stderr,
    )
    return 1


def _steer_read_prompts(path: str) -> list[str]:
    """One prompt per non-empty line; '#' lines are comments (DS4 rule)."""
    prompts = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line and not line.startswith("#"):
                prompts.append(line)
    return prompts


# Bundled prompt sets (DS4, MIT): `--preset NAME` -> (good file, bad file).
_STEER_PRESETS = {"verbosity": ("succinct.txt", "verbose.txt")}


def _steer_data_dir():
    from pathlib import Path

    return Path(__file__).parent / "steering" / "data" / "ds4"


def _steer_build(args) -> int:
    import time

    import requests

    good_file, bad_file = args.good_file, args.bad_file
    if args.preset:
        good_name, bad_name = _STEER_PRESETS[args.preset]
        good_file = good_file or str(_steer_data_dir() / good_name)
        bad_file = bad_file or str(_steer_data_dir() / bad_name)
    if not good_file or not bad_file:
        print(
            "Give --good-file and --bad-file, or --preset verbosity",
            file=sys.stderr,
        )
        return 1
    good = _steer_read_prompts(good_file)
    bad = _steer_read_prompts(bad_file)
    if len(good) < 2 or len(bad) < 2:
        print("Capture needs at least 2 good and 2 bad prompts", file=sys.stderr)
        return 1
    base_url, headers = _steer_server(args)
    resp = requests.post(
        f"{_steer_model_url(base_url, args.model)}/steering/build",
        json={
            "name": args.name,
            "good": good,
            "bad": bad,
            "component": args.component,
            "system": args.system,
            "thinking": bool(args.think),
            "pair_normalize": bool(args.pair_normalize),
            "orthogonalize": not args.no_orthogonalize,
        },
        headers=headers,
        timeout=60,
    )
    if resp.status_code != 200:
        return _steer_fail("build", resp)
    job = resp.json()
    job_id = job["job_id"]
    total = job.get("total", 0)
    print(f"Capturing {total} prompts on {args.model} (job {job_id})")
    last_done = -1
    while True:
        time.sleep(1.0)
        poll = requests.get(
            f"{base_url}/admin/api/steering/jobs/{job_id}",
            headers=headers,
            timeout=10,
        )
        if poll.status_code != 200:
            return _steer_fail("build", poll)
        job = poll.json()
        if job.get("done") != last_done:
            last_done = job.get("done", 0)
            print(f"  {last_done}/{total}")
        if job.get("status") == "error":
            print(f"Build failed: {job.get('error')}", file=sys.stderr)
            return 1
        if job.get("status") == "done":
            break
    print(f"Pack written to {job.get('pack')}")
    manifest = job.get("manifest") or {}
    model_info = manifest.get("model") or {}
    hook = manifest.get("hook") or {}
    build = manifest.get("build") or {}
    print(
        f"component={hook.get('component')}/{hook.get('mode')} "
        f"shape={model_info.get('n_layers')}x{model_info.get('hidden_size')} "
        f"method={build.get('direction_method')}"
    )
    cos = sorted(c for c in (build.get("split_half_cos") or []) if c is not None)
    if cos:
        print(f"split-half cosine: min={cos[0]:.3f} median={cos[len(cos) // 2]:.3f}")
    mask = build.get("layer_mask")
    if mask:
        print(f"layer mask: {len(mask)}/{model_info.get('n_layers')} layers kept")
    return 0


def _steer_sweep(args) -> int:
    import time

    import requests

    scales = []
    for token in args.scales.split(","):
        token = token.strip()
        if token:
            scales.append(float(token))
    if not scales:
        print("No scales parsed from --scales", file=sys.stderr)
        return 1
    prompts_file = args.prompts or str(_steer_data_dir() / "eval_prompts.txt")
    prompts = _steer_read_prompts(prompts_file)
    if not prompts:
        print(f"No prompts found in {prompts_file}", file=sys.stderr)
        return 1
    base_url, headers = _steer_server(args)
    model_url = _steer_model_url(base_url, args.model)
    settings_url = f"{model_url}/settings"

    state = requests.get(f"{model_url}/steering", headers=headers, timeout=10)
    if state.status_code != 200:
        return _steer_fail("sweep", state)
    orig = state.json()
    orig_pack = orig.get("pack")
    orig_scales = {"ffn": orig.get("ffn_scale"), "attn": orig.get("attn_scale")}

    print(
        "WARNING: sweep changes the engine-wide steering scale; "
        "run it on an idle server."
    )
    resp = requests.put(
        settings_url,
        json={"steering_pack": args.pack},
        headers=headers,
        timeout=60,
    )
    if resp.status_code != 200:
        return _steer_fail("sweep", resp)
    if resp.json().get("requires_reload"):
        print("Pack change reloads the engine; the first request may be slow.")

    # The pack's own manifest decides which component gets swept; fall back
    # to the live state, then ffn (FFN first, attention at 0).
    component = None
    try:
        from .steering.pack import resolve_pack_ref

        component = resolve_pack_ref(args.pack).component
    except Exception:
        pass
    if component not in ("ffn", "attn"):
        components = orig.get("components") or []
        component = (
            "ffn" if "ffn" in components else (components[0] if components else "ffn")
        )
    scale_field = f"steering_{component}_scale"
    print(
        f"Sweeping {component} scale over {scales} with {len(prompts)} "
        f"prompt(s), {args.tokens} tokens each"
    )

    try:
        for scale in scales:
            resp = requests.put(
                settings_url,
                json={scale_field: scale},
                headers=headers,
                timeout=60,
            )
            if resp.status_code != 200:
                return _steer_fail("sweep", resp)
            for idx, prompt in enumerate(prompts, 1):
                body = {
                    "model": args.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0,
                    "max_tokens": args.tokens,
                }
                if args.nothink:
                    body["enable_thinking"] = False
                started = time.time()
                completion = requests.post(
                    f"{base_url}/v1/chat/completions",
                    json=body,
                    headers=headers,
                    timeout=max(300, args.tokens * 2),
                )
                if completion.status_code != 200:
                    return _steer_fail("sweep", completion)
                text = completion.json()["choices"][0]["message"].get("content") or ""
                print(
                    f"\n=== scale {scale} | prompt {idx}/{len(prompts)} "
                    f"| words {len(text.split())} "
                    f"| {time.time() - started:.1f}s"
                )
                print(text)
    finally:
        restore: dict = {}
        note = ""
        if orig_pack is None:
            # Neutralise instead of removing the pack (scale 0
            # re-enables instantly and shares the cache with unsteered runs).
            restore[scale_field] = 0.0
            note = (
                f"pack {args.pack!r} left set at scale 0 — remove it with: "
                f"omlx steer set {args.model} --off"
            )
        else:
            restore["steering_pack"] = orig_pack
            for name, value in orig_scales.items():
                restore[f"steering_{name}_scale"] = value
        resp = requests.put(settings_url, json=restore, headers=headers, timeout=60)
        if resp.status_code != 200:
            print(
                f"WARNING: failed to restore steering settings: "
                f"HTTP {resp.status_code}",
                file=sys.stderr,
            )
        elif note:
            print(f"\n{note}")
        else:
            print("\nOriginal steering settings restored")
    return 0


def _steer_verify(args) -> int:
    import time

    import requests

    scales = []
    for token in args.scales.split(","):
        token = token.strip()
        if token:
            scales.append(float(token))
    if not scales:
        print("No scales parsed from --scales", file=sys.stderr)
        return 1
    base_url, headers = _steer_server(args)
    model_url = _steer_model_url(base_url, args.model)
    if args.pack:
        resp = requests.put(
            f"{model_url}/settings",
            json={"steering_pack": args.pack},
            headers=headers,
            timeout=60,
        )
        if resp.status_code != 200:
            return _steer_fail("verify", resp)
        if resp.json().get("requires_reload"):
            print("Pack change reloads the engine; the first request may be slow.")
    payload = {
        "scopes": [s.strip() for s in args.scopes.split(",") if s.strip()],
        "scales": scales,
        "quick": bool(args.quick),
        "thinking": bool(args.thinking),
    }
    if args.samples:
        payload["samples"] = args.samples
    resp = requests.post(
        f"{model_url}/steering/verify", json=payload, headers=headers, timeout=60
    )
    if resp.status_code != 200:
        return _steer_fail("verify", resp)
    job = resp.json()
    job_id, total = job["job_id"], job.get("total", 0)
    print(f"Verifying {total} point(s) on {args.model} (job {job_id})")
    last_done = -1
    while True:
        time.sleep(2.0)
        poll = requests.get(
            f"{base_url}/admin/api/steering/jobs/{job_id}",
            headers=headers,
            timeout=10,
        )
        if poll.status_code != 200:
            return _steer_fail("verify", poll)
        job = poll.json()
        if job.get("done") != last_done:
            last_done = job.get("done", 0)
            print(f"  {last_done}/{total}")
        if job.get("status") == "error":
            print(f"Verify failed: {job.get('error')}", file=sys.stderr)
            return 1
        if job.get("status") == "done":
            break
    entries = job.get("quality") or []
    header = (
        f"\n{'scope':10s} {'scale':>6s} {'kl':>7s} {'nll d%':>7s} "
        f"{'rep dpt':>8s} {'struct dpt':>10s} {'effect%':>8s}  verdict"
    )
    print(header)
    for e in entries:
        if e.get("role") == "reference":
            print(
                f"{e['scope']:10s} {e['scale']:>6.2f} {e['kl']:>7.4f} "
                f"{'-':>7s} {'-':>8s} {'-':>10s} {'-':>8s}  reference"
            )
            continue
        verdict = (
            "PASS" if e.get("passed") else "FAIL: " + "; ".join(e.get("failed", []))
        )
        print(
            f"{e['scope']:10s} {e['scale']:>6.2f} {e['kl']:>7.4f} "
            f"{e.get('nll_delta_pct', 0.0):>7.1f} "
            f"{e.get('repeat_delta_pt', 0.0):>8.1f} "
            f"{e.get('structure_delta_pt', 0.0):>10.1f} "
            f"{e.get('effect_pct', 0.0):>8.1f}  {verdict}"
        )
    rec = job.get("recommended")
    if rec:
        print(f"\nRecommended: scale {rec['scale']} at scope {rec['scope']}")
    elif entries:
        print("\nNo passing point — no scale marked recommended")
    return 0


def _steer_set(args) -> int:
    import requests

    body: dict = {}
    if args.off:
        body["steering_pack"] = None
    elif args.pack:
        body["steering_pack"] = args.pack
    if args.ffn is not None:
        body["steering_ffn_scale"] = args.ffn
    if args.attn is not None:
        body["steering_attn_scale"] = args.attn
    if not body:
        print(
            "Nothing to set: pass --ffn/--attn and/or --pack/--off",
            file=sys.stderr,
        )
        return 1
    base_url, headers = _steer_server(args)
    resp = requests.put(
        f"{_steer_model_url(base_url, args.model)}/settings",
        json=body,
        headers=headers,
        timeout=60,
    )
    if resp.status_code != 200:
        return _steer_fail("set", resp)
    info = resp.json()
    print(f"Steering settings updated for {args.model}")
    if info.get("requires_reload"):
        if info.get("auto_reloaded"):
            print("Pack change: engine reloaded")
        else:
            print(
                "Pack change: engine unloaded; it reloads with the pack "
                "on the next request"
            )
    elif info.get("steering_applied_live"):
        print("Scales applied live")
    return 0


def _steer_show(args) -> int:
    import requests

    base_url, headers = _steer_server(args)
    resp = requests.get(
        f"{_steer_model_url(base_url, args.model)}/steering",
        headers=headers,
        timeout=10,
    )
    if resp.status_code != 200:
        return _steer_fail("show", resp)
    state = resp.json()
    print(f"model:      {state.get('model_id')}")
    print(f"pack:       {state.get('pack')}")
    if not state.get("installed"):
        if not state.get("loaded"):
            print("steering:   model not loaded")
        else:
            print("steering:   not installed on the loaded engine")
        return 0
    print(f"active:     {state.get('active')}")
    print(f"ffn scale:  {state.get('ffn_scale')}")
    print(f"attn scale: {state.get('attn_scale')}")
    print(
        f"scope:      {state.get('scope')} "
        f"(decode_tokens={state.get('decode_tokens')})"
    )
    print(
        f"method:     {state.get('method')} "
        f"(norm_preserve={state.get('norm_preserve')})"
    )
    print(f"components: {', '.join(state.get('components') or [])}")
    print(f"salt:       {state.get('salt')}  epoch: {state.get('epoch')}")
    if not state.get("debug"):
        print("hits:       (debug off)")
        return 0
    hits = state.get("hits") or {}
    shown = ", ".join(f"{k}={v}" for k, v in list(hits.items())[:10])
    suffix = f" … ({len(hits)} total)" if len(hits) > 10 else ""
    print(f"hits:       {shown or 'none'}{suffix}")
    return 0


def _steer_model_info(model_ref: str) -> dict | None:
    """Read model_type/hidden_size from a model directory's config.json."""
    import json
    from pathlib import Path

    config = Path(model_ref).expanduser() / "config.json"
    if not config.is_file():
        print(
            f"warning: {model_ref} has no config.json; " "skipping compatibility marks"
        )
        return None
    try:
        data = json.loads(config.read_text())
    except (OSError, ValueError) as exc:
        print(f"warning: cannot read {config}: {exc}")
        return None
    return {
        "model_type": data.get("model_type"),
        "hidden_size": data.get("hidden_size"),
    }


def _steer_list(args) -> int:
    from .steering.pack import SteeringPackError, get_steering_dir, load_pack

    sdir = get_steering_dir()
    if not sdir.is_dir():
        print(f"No steering directory at {sdir} — build or import a pack first")
        return 0
    model_info = _steer_model_info(args.model) if args.model else None
    for entry in sorted(sdir.iterdir()):
        if not entry.is_dir():
            continue
        try:
            pack = load_pack(entry)
        except (SteeringPackError, OSError, ValueError) as exc:
            print(f"{entry.name:24s} INVALID: {exc}")
            continue
        model = pack.manifest.get("model") or {}
        desc = str(pack.manifest.get("description") or "").splitlines()[0][:40]
        mark = ""
        if model_info is not None:
            same_type = model.get("model_type") == model_info.get("model_type")
            same_size = not model.get("hidden_size") or (
                model.get("hidden_size") == model_info.get("hidden_size")
            )
            mark = "  [compatible]" if same_type and same_size else "  [incompatible]"
        print(
            f"{pack.name:24s} {pack.component}/{pack.mode} "
            f"{pack.n_layers}x{pack.hidden_size} "
            f"model={model.get('model_type') or '?'}{mark}  {desc}"
        )
    return 0


def _steer_export(args) -> int:
    import zipfile
    from pathlib import Path

    from .steering.pack import get_steering_dir

    src = get_steering_dir() / args.name
    if not (src / "manifest.json").is_file():
        print(f"Not a steering pack: {src}", file=sys.stderr)
        return 1
    out = Path(args.output).expanduser()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(src.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(src))
    print(f"Exported {src} -> {out}")
    return 0


def _steer_zip_extract(archive, dest) -> None:
    """Extract zip members, refusing absolute paths and parent traversal."""
    import shutil
    from pathlib import Path

    from .steering.pack import SteeringPackError

    dest = Path(dest).resolve()
    for info in archive.infolist():
        if info.is_dir():
            continue
        member = Path(info.filename)
        if member.is_absolute() or ".." in member.parts:
            raise SteeringPackError(f"unsafe zip member: {info.filename!r}")
        target = dest / member
        if not target.resolve().is_relative_to(dest):
            raise SteeringPackError(f"unsafe zip member: {info.filename!r}")
        target.parent.mkdir(parents=True, exist_ok=True)
        with archive.open(info) as fsrc, open(target, "wb") as fdst:
            shutil.copyfileobj(fsrc, fdst)


def _steer_wrap_f32(args, path):
    """Wrap a bare DS4 .f32 file into a pack."""
    import json
    from pathlib import Path

    from .steering.pack import (
        SteeringPack,
        SteeringPackError,
        _normalize_rows,
        _read_f32,
    )

    if not args.json_manifest:
        raise SteeringPackError("a bare .f32 import needs --json FILE (base manifest)")
    if not args.model:
        raise SteeringPackError(
            "a bare .f32 import needs --model DIR (model directory that "
            "provides the vector shape)"
        )
    config_path = Path(args.model).expanduser() / "config.json"
    try:
        config = json.loads(config_path.read_text())
    except (OSError, ValueError) as exc:
        raise SteeringPackError(f"cannot read {config_path}: {exc}") from exc
    n_layers = config.get("num_hidden_layers")
    hidden = config.get("hidden_size")
    if not isinstance(n_layers, int) or not isinstance(hidden, int):
        raise SteeringPackError(f"{config_path} lacks num_hidden_layers/hidden_size")
    vectors, _ = _normalize_rows(_read_f32(path, (n_layers, hidden)))
    try:
        manifest = json.loads(Path(args.json_manifest).expanduser().read_text())
    except (OSError, ValueError) as exc:
        raise SteeringPackError(f"cannot read {args.json_manifest}: {exc}") from exc
    name = args.name or path.stem
    manifest["name"] = name
    model_block = dict(manifest.get("model") or {})
    model_block.update(
        {
            "model_type": config.get("model_type"),
            "hidden_size": hidden,
            "n_layers": n_layers,
        }
    )
    manifest["model"] = model_block
    # Bare DS4 vectors are ffn/block by definition.
    manifest.setdefault("hook", {"component": "ffn", "mode": "block"})
    return SteeringPack(name, vectors, manifest)


def _steer_import(args) -> int:
    import shutil
    import tempfile
    import zipfile
    from pathlib import Path

    from .steering.pack import (
        SteeringPackError,
        get_steering_dir,
        load_pack,
        save_pack,
    )

    source = str(args.source)
    ref = Path(source).expanduser()
    is_f32 = ref.is_file() and ref.suffix == ".f32"
    origin = None
    tmp = None
    try:
        if ref.is_file() and ref.suffix == ".zip":
            tmp = Path(tempfile.mkdtemp(prefix="omlx-steer-import-"))
            with zipfile.ZipFile(ref) as archive:
                _steer_zip_extract(archive, tmp)
            if (tmp / "manifest.json").is_file():
                origin = tmp
            else:
                packs = [d for d in tmp.iterdir() if (d / "manifest.json").is_file()]
                if len(packs) != 1:
                    raise SteeringPackError(
                        "the zip must contain exactly one steering pack "
                        "(a manifest.json at the root or in one subdirectory)"
                    )
                origin = packs[0]
        elif ref.is_dir():
            origin = ref
        elif is_f32:
            pass
        elif "/" in source:
            try:
                from huggingface_hub import snapshot_download
            except ImportError:
                print(
                    "huggingface_hub is not installed; "
                    "cannot import from the HF Hub",
                    file=sys.stderr,
                )
                return 1
            print(f"Downloading {source} from the HF Hub")
            tmp = Path(
                snapshot_download(
                    source,
                    allow_patterns=[
                        "*.f32",
                        "*.json",
                        "*.txt",
                        "*.md",
                        "LICENSE*",
                    ],
                )
            )
            origin = tmp
        else:
            raise SteeringPackError(f"not found: {source}")

        pack = _steer_wrap_f32(args, ref) if is_f32 else load_pack(origin)
    except (SteeringPackError, OSError, ValueError) as exc:
        print(f"Import failed: {exc}", file=sys.stderr)
        if tmp is not None:
            shutil.rmtree(tmp, ignore_errors=True)
        return 1

    name = args.name or pack.name
    dest = get_steering_dir() / name
    if dest.exists():
        print(f"Overwriting existing pack {dest}")
    if is_f32:
        save_pack(pack, dest)
    else:
        shutil.copytree(origin, dest, dirs_exist_ok=True)
    if tmp is not None:
        shutil.rmtree(tmp, ignore_errors=True)
    try:
        load_pack(dest)
    except (SteeringPackError, OSError, ValueError) as exc:
        print(f"Imported pack failed validation at {dest}: {exc}", file=sys.stderr)
        return 1
    print(f"Imported pack {name!r} -> {dest}")
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="omlx: Production-ready LLM server for Apple Silicon",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  omlx serve mlx-community/Llama-3.2-3B-Instruct-4bit --port 8000
  omlx launch codex --model qwen3.5
        """,
    )
    parser.add_argument(
        "--version",
        action="version",
        version=__version__,
        help="Print the oMLX version and exit",
    )
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    for name, help_text in (
        ("start", "Start oMLX as a managed background server"),
        ("stop", "Stop the managed background oMLX server"),
        ("restart", "Restart the managed background oMLX server"),
    ):
        lifecycle_parser = subparsers.add_parser(
            name,
            help=help_text,
            description=help_text,
        )
        lifecycle_parser.add_argument(
            "--timeout",
            type=float,
            default=60.0,
            help="Seconds to wait for the macOS app/server to reach the requested state",
        )
        if name in {"start", "restart"}:
            lifecycle_parser.add_argument(
                "--no-wait",
                action="store_true",
                help="Return after sending the request without waiting for server health",
            )

    # Serve command (multi-model)
    serve_parser = subparsers.add_parser(
        "serve",
        help="Start multi-model OpenAI-compatible server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="""
Start a multi-model inference server with LRU-based memory management.

Models are discovered from subdirectories of --model-dir. Each subdirectory
should contain a valid model with config.json and *.safetensors files.

Example directory structure:
  /path/to/models/
  ├── llama-3b/           → model_id: "llama-3b"
  │   ├── config.json
  │   └── model.safetensors
  ├── qwen-7b/            → model_id: "qwen-7b"
  └── mistral-7b/         → model_id: "mistral-7b"
""",
    )

    # Required arguments
    serve_parser.add_argument(
        "--model-dir",
        type=str,
        default=None,
        help="Directory containing model subdirectories (default: ~/.omlx/models)",
    )
    # Server options
    serve_parser.add_argument(
        "--host", type=str, default=None, help="Host to bind (default: 127.0.0.1)"
    )
    serve_parser.add_argument(
        "--port", type=int, default=None, help="Port to bind (default: 8000)"
    )
    serve_parser.add_argument(
        "--log-level",
        type=str,
        choices=["trace", "debug", "info", "warning", "error"],
        default=None,
        help="Log level (default: info). trace includes full message content",
    )
    serve_parser.add_argument(
        "--sse-keepalive-mode",
        type=str,
        choices=["chunk", "comment", "off"],
        default=None,
        help="SSE keepalive emission mode (default: chunk). 'chunk' emits "
        "protocol-aware no-op events compatible with strict clients like "
        "OpenClaw / WorkBuddy; 'comment' emits the legacy ': keep-alive' SSE "
        "comment; 'off' disables keepalive entirely",
    )
    serve_parser.add_argument(
        "--max-audio-upload-size",
        type=str,
        default=None,
        help="Maximum audio upload size for /v1/audio/transcriptions and "
        "/v1/audio/process (e.g. '100MB', '500MB'). Overrides the value "
        "in settings.json (built-in default: 100MB). Uploads are buffered "
        "in memory, so this is also a per-request RAM cap",
    )
    serve_parser.add_argument(
        "--max-image-upload-size",
        type=str,
        default=None,
        help="Maximum image payload size for VLM inputs (e.g. '50MB', '100MB'). "
        "Overrides the value in settings.json (built-in default: 50MB).",
    )
    serve_parser.add_argument(
        "--max-image-side-length",
        type=int,
        default=None,
        help="Maximum side length in pixels for VLM input images. Images exceeding "
        "this limit are downscaled preserving aspect ratio (built-in default: 2048, "
        "0 to disable).",
    )

    # Scheduler options (for BatchedEngine)
    serve_parser.add_argument(
        "--max-concurrent-requests",
        type=int,
        default=None,
        help="Max requests processed simultaneously. Higher values increase throughput but use more memory. (default: 8)",
    )
    serve_parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=None,
        help="Max embedding inputs processed in one forward pass. Higher values increase throughput but use more memory. (default: 32)",
    )

    # Memory guard options
    serve_parser.add_argument(
        "--memory-guard",
        type=str,
        choices=["off", "safe", "balanced", "aggressive"],
        default=None,
        help="Memory guard tier, or 'off' to disable the guard. The tier sets how much memory stays free for other apps: safe keeps about 20%% of RAM (6-16 GB), balanced about 8%% (3-8 GB) and may compress a quarter of other apps' memory, aggressive 2%% (1.5-4 GB) and may compress half of it. Passing a tier also turns the guard on. (default: balanced)",
    )
    serve_parser.add_argument(
        "--memory-guard-gb",
        type=_positive_float,
        default=None,
        help="Custom memory guard ceiling in GB. Sets memory guard tier to custom and turns the guard on.",
    )

    # paged SSD cache options
    serve_parser.add_argument(
        "--paged-ssd-cache-dir",
        type=str,
        default=None,
        help="Directory for paged SSD cache storage (enables oMLX prefix cache)",
    )
    serve_parser.add_argument(
        "--paged-ssd-cache-max-size",
        type=str,
        default=None,
        help="Maximum paged SSD cache size (e.g., '100GB', '50GB'). Default: 100GB",
    )
    serve_parser.add_argument(
        "--hot-cache-max-size",
        type=str,
        default=None,
        help="Maximum in-memory hot cache size (e.g., '8GB', '4GB'). Default: 0 (disabled)",
    )
    serve_parser.add_argument(
        "--hot-cache-write-through",
        action="store_true",
        default=None,
        help="Persist every hot-cache block to SSD immediately (write-through). "
        "Keeps RAM-speed resume while retaining SSD durability for all sessions.",
    )
    serve_parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable oMLX paged SSD cache. mlx-lm BatchGenerator still manages KV states internally.",
    )
    serve_parser.add_argument(
        "--initial-cache-blocks",
        type=int,
        default=None,
        help="Number of cache blocks to pre-allocate at startup (default: 256). "
        "Higher values reduce dynamic allocation overhead for large contexts.",
    )

    # MCP options
    serve_parser.add_argument(
        "--mcp-config",
        type=str,
        default=None,
        help="Path to MCP configuration file (JSON/YAML) for tool integration",
    )

    # HuggingFace options
    serve_parser.add_argument(
        "--hf-endpoint",
        type=str,
        default=None,
        help="Custom HuggingFace Hub endpoint URL (e.g., https://hf-mirror.com)",
    )
    serve_parser.add_argument(
        "--hf-cache",
        dest="hf_cache_enabled",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Discover models from the standard HuggingFace Hub local cache (default: enabled)",
    )

    # ModelScope options
    serve_parser.add_argument(
        "--ms-endpoint",
        type=str,
        default=None,
        help="Custom ModelScope Hub endpoint URL",
    )

    # Network options
    serve_parser.add_argument(
        "--http-proxy",
        type=str,
        default=None,
        help="HTTP proxy URL (e.g., http://proxy.company.com:8080)",
    )
    serve_parser.add_argument(
        "--https-proxy",
        type=str,
        default=None,
        help="HTTPS proxy URL (e.g., http://proxy.company.com:8080)",
    )
    serve_parser.add_argument(
        "--no-proxy",
        type=str,
        default=None,
        help="Comma-separated hosts/IPs to bypass proxy (e.g., localhost,127.0.0.1)",
    )
    serve_parser.add_argument(
        "--ca-bundle",
        type=str,
        default=None,
        help="Path to CA bundle PEM file for TLS interception environments",
    )

    # Base path and auth
    serve_parser.add_argument(
        "--base-path",
        type=str,
        default=None,
        help="Base directory for oMLX data (default: ~/.omlx)",
    )
    serve_parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key for authentication (required for non-loopback binds)",
    )
    serve_parser.add_argument(
        "--headless",
        action="store_true",
        help="Serve the inference and admin APIs without the web UI. "
        "Not saved to settings; same as OMLX_HEADLESS=1",
    )

    # Launch command
    launch_parser = subparsers.add_parser(
        "launch",
        help="Launch an external tool with oMLX integration",
        description=(
            "Configure and launch external coding tools (Claude Code, Claude Desktop, "
            "Copilot, Codex, Codex App, OpenCode, OpenClaw, Hermes Agent, Pi, Mistral Vibe, "
            "DeepSeek "
            "Harness) to use the running oMLX server."
        ),
    )
    launch_parser.add_argument(
        "tool",
        type=str,
        help=(
            "Tool to launch: claude, claude_desktop, copilot, codex, codex_app, opencode, "
            "openclaw, hermes, pi, vibe, dsh, or 'list' to show available"
        ),
    )
    launch_parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model to use (interactive selection if not specified)",
    )
    launch_parser.add_argument(
        "--host",
        type=str,
        default=None,
        help="oMLX server host (default: from settings or 127.0.0.1)",
    )
    launch_parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="oMLX server port (default: from settings or 8000)",
    )
    launch_parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key for oMLX server authentication",
    )
    launch_parser.add_argument(
        "--tools-profile",
        type=str,
        default="coding",
        choices=["minimal", "coding", "messaging", "full"],
        help="OpenClaw tools profile (default: coding)",
    )
    launch_parser.add_argument(
        "--opus",
        dest="opus_model",
        type=str,
        default=None,
        help="Claude Code Opus tier model (Claude integration only)",
    )
    launch_parser.add_argument(
        "--sonnet",
        dest="sonnet_model",
        type=str,
        default=None,
        help="Claude Code Sonnet tier model (Claude integration only)",
    )
    launch_parser.add_argument(
        "--haiku",
        dest="haiku_model",
        type=str,
        default=None,
        help="Claude Code Haiku tier model (Claude integration only)",
    )
    launch_parser.add_argument(
        "--cross-session",
        action="store_true",
        default=False,
        help=(
            "Allow the launched session to be reachable via Claude Code's "
            "cross-session messaging (ListAgents/SendMessage). This requires "
            "enabling telemetry and feature-flag traffic to Anthropic that is "
            "otherwise kept disabled by default (Claude integration only)."
        ),
    )

    # Diagnose command
    diagnose_parser = subparsers.add_parser(
        "diagnose",
        help="Diagnose installation or runtime issues",
        description="Run diagnostic checks and print recovery steps.",
    )
    diagnose_parser.add_argument(
        "target",
        type=str,
        choices=["menubar"],
        help="What to diagnose. 'menubar' checks Tahoe ControlCenter visibility.",
    )

    # Cluster diagnostics and planning for the first implementation slice.
    cluster_parser = subparsers.add_parser(
        "cluster",
        help="Inspect distributed-node readiness and exercise a local worker",
        description=(
            "Distributed-cluster diagnostics and unequal-memory planning. "
            "This command does not configure interfaces or initialize JACCL."
        ),
    )
    cluster_subparsers = cluster_parser.add_subparsers(
        dest="cluster_action",
        required=True,
        help="Cluster diagnostic command",
    )
    cluster_status_parser = cluster_subparsers.add_parser(
        "status",
        help="Report local memory, runtime, RDMA, and Thunderbolt readiness",
    )
    cluster_status_parser.add_argument(
        "--route-to",
        metavar="IP",
        default=None,
        help="Also inspect the active route to an IPv4 or IPv6 peer address",
    )
    cluster_status_parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON",
    )
    cluster_smoke_parser = cluster_subparsers.add_parser(
        "worker-smoke",
        help="Run a real isolated worker ready/ping/shutdown round trip",
    )
    cluster_smoke_parser.add_argument(
        "--timeout",
        type=_positive_float,
        default=5.0,
        help="Per-operation worker deadline in seconds (default: 5)",
    )
    cluster_smoke_parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON",
    )
    cluster_collective_parser = cluster_subparsers.add_parser(
        "collective-smoke",
        help="Run two local MLX ranks and verify a ring all-sum",
    )
    cluster_collective_parser.add_argument(
        "--timeout",
        type=_positive_float,
        default=20.0,
        help="Overall collective deadline in seconds (default: 20)",
    )
    cluster_collective_parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON",
    )
    cluster_pipeline_parser = cluster_subparsers.add_parser(
        "pipeline-smoke",
        help="Run an unequal two-rank hybrid Nemotron-H graph",
    )
    cluster_pipeline_parser.add_argument(
        "--timeout",
        type=_positive_float,
        default=30.0,
        help="Overall pipeline deadline in seconds (default: 30)",
    )
    cluster_pipeline_parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON",
    )
    cluster_plan_parser = cluster_subparsers.add_parser(
        "plan",
        help="Plan contiguous layers across unequal node memory budgets",
    )
    cluster_plan_source = cluster_plan_parser.add_mutually_exclusive_group(
        required=True
    )
    cluster_plan_source.add_argument(
        "--model",
        metavar="PATH",
        help="Inspect safetensors headers from a downloaded model directory",
    )
    cluster_plan_source.add_argument(
        "--model-size",
        metavar="SIZE",
        help="Plan an estimated model before download (for example 300GB)",
    )
    cluster_plan_parser.add_argument(
        "--layers",
        type=_positive_int,
        default=80,
        help="Layer count used with --model-size (default: 80)",
    )
    cluster_plan_parser.add_argument(
        "--node",
        action="append",
        metavar="NAME=SIZE",
        help=(
            "Node memory budget in rank order; repeat for each node. "
            "Defaults to this Mac's recommended working set."
        ),
    )
    cluster_plan_parser.add_argument(
        "--reserve",
        default="0",
        metavar="SIZE",
        help="Memory to reserve on every node for KV/activations (default: 0)",
    )
    cluster_plan_parser.add_argument(
        "--peer",
        action="append",
        metavar="SSH_HOST",
        help=(
            "SSH host that may hold the model; repeat for each. Used with "
            "--model when this Mac holds only its own stage: the first peer "
            "that can read a complete model is the one that measures it."
        ),
    )
    cluster_plan_parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON",
    )

    steer_parser = subparsers.add_parser(
        "steer",
        help="Build, apply, and inspect directional steering packs",
        description=(
            "Directional steering control (experimental, see "
            "docs/steering.md). build/sweep/"
            "set/show talk to a running oMLX server; list/import/export "
            "work on local pack files."
        ),
    )
    steer_subparsers = steer_parser.add_subparsers(
        dest="steer_action",
        required=True,
        help="Steering command",
    )

    def _add_steer_server_args(parser):
        parser.add_argument(
            "--host",
            type=str,
            default=None,
            help="oMLX server host (default: from settings or 127.0.0.1)",
        )
        parser.add_argument(
            "--port",
            type=int,
            default=None,
            help="oMLX server port (default: from settings or 8000)",
        )
        parser.add_argument(
            "--api-key",
            type=str,
            default=None,
            help="API key for oMLX server authentication",
        )

    steer_build_parser = steer_subparsers.add_parser(
        "build",
        help="Capture paired prompts on a loaded model and build a pack",
    )
    steer_build_parser.add_argument(
        "model",
        metavar="MODEL",
        help="Model id on the server (must be loaded)",
    )
    steer_build_parser.add_argument(
        "--good-file",
        default=None,
        metavar="G",
        help="Prompt file with the desired behaviour, one per line (# comments allowed)",
    )
    steer_build_parser.add_argument(
        "--bad-file",
        default=None,
        metavar="B",
        help="Prompt file with the undesired behaviour, one per line",
    )
    steer_build_parser.add_argument(
        "--preset",
        choices=sorted(_STEER_PRESETS),
        default=None,
        help=(
            "Use a bundled prompt set instead of files: 'verbosity' = 100 "
            "DS4 succinct/verbose pairs (good = succinct)"
        ),
    )
    steer_build_parser.add_argument(
        "--name",
        required=True,
        metavar="N",
        help="Name for the built pack",
    )
    steer_build_parser.add_argument(
        "--component",
        choices=("ffn", "attn"),
        default=None,
        help="Hook component to capture (default: the model's, attn on Qwen3.8)",
    )
    steer_build_parser.add_argument(
        "--system",
        default=None,
        metavar="S",
        help="System prompt used during capture",
    )
    steer_build_parser.add_argument(
        "--think",
        action="store_true",
        help="Capture with thinking enabled",
    )
    steer_build_parser.add_argument(
        "--pair-normalize",
        action="store_true",
        help="Average per-pair normalized diffs instead of the diff of means",
    )
    steer_build_parser.add_argument(
        "--no-orthogonalize",
        action="store_true",
        help="Keep the bad-behaviour component in the direction",
    )
    _add_steer_server_args(steer_build_parser)

    steer_sweep_parser = steer_subparsers.add_parser(
        "sweep",
        help=(
            "Generate at several steering scales on a loaded model "
            "(changes the engine-wide scale; run on an idle server)"
        ),
    )
    steer_sweep_parser.add_argument(
        "model",
        metavar="MODEL",
        help="Model id on the server",
    )
    steer_sweep_parser.add_argument(
        "--pack",
        required=True,
        metavar="N",
        help="Pack to apply for the sweep",
    )
    steer_sweep_parser.add_argument(
        "--prompts",
        default=None,
        metavar="P",
        help=(
            "Held-out prompts, one per line (# comments allowed; default: "
            "5 bundled DS4 evaluation prompts)"
        ),
    )
    steer_sweep_parser.add_argument(
        "--scales",
        default="-1,-0.5,0,0.5,1,2",
        help="Comma-separated scales (default: -1,-0.5,0,0.5,1,2)",
    )
    steer_sweep_parser.add_argument(
        "--tokens",
        type=_positive_int,
        default=180,
        help="Max new tokens per generation (default: 180)",
    )
    steer_sweep_parser.add_argument(
        "--nothink",
        action="store_true",
        help="Send enable_thinking=false with each request",
    )
    _add_steer_server_args(steer_sweep_parser)

    steer_verify_parser = steer_subparsers.add_parser(
        "verify",
        help="Run the steering quality gate for a pack at several scales",
    )
    steer_verify_parser.add_argument(
        "model",
        metavar="MODEL",
        help="Model id on the server (must be loaded with steering installed)",
    )
    steer_verify_parser.add_argument(
        "--pack",
        metavar="N",
        default=None,
        help="Pack to set on the model before verifying",
    )
    steer_verify_parser.add_argument(
        "--scopes",
        default="prefill+k",
        help="Comma-separated scopes (default: prefill+k)",
    )
    steer_verify_parser.add_argument(
        "--scales",
        default="-0.5,-1",
        help="Comma-separated scales (default: -0.5,-1)",
    )
    steer_verify_parser.add_argument(
        "--quick",
        action="store_true",
        help="KL, repetition and structure only, in minutes",
    )
    steer_verify_parser.add_argument(
        "--thinking",
        action="store_true",
        help="Run the benchmarks with thinking enabled",
    )
    steer_verify_parser.add_argument(
        "--samples",
        type=_positive_int,
        default=None,
        help="Benchmark sample size (default: server default of 300)",
    )
    _add_steer_server_args(steer_verify_parser)

    steer_set_parser = steer_subparsers.add_parser(
        "set",
        help="Set the steering pack and/or scales for a model",
    )
    steer_set_parser.add_argument(
        "model",
        metavar="MODEL",
        help="Model id on the server",
    )
    steer_set_parser.add_argument(
        "--ffn",
        type=float,
        default=None,
        metavar="F",
        help="FFN steering scale",
    )
    steer_set_parser.add_argument(
        "--attn",
        type=float,
        default=None,
        metavar="A",
        help="Attention steering scale",
    )
    steer_set_source = steer_set_parser.add_mutually_exclusive_group()
    steer_set_source.add_argument(
        "--pack",
        metavar="N",
        help="Pack name (or path) to set",
    )
    steer_set_source.add_argument(
        "--off",
        action="store_true",
        help="Remove the pack (drops the scales with it)",
    )
    _add_steer_server_args(steer_set_parser)

    steer_show_parser = steer_subparsers.add_parser(
        "show",
        help="Show the live steering state of a model",
    )
    steer_show_parser.add_argument(
        "model",
        metavar="MODEL",
        help="Model id on the server",
    )
    _add_steer_server_args(steer_show_parser)

    steer_list_parser = steer_subparsers.add_parser(
        "list",
        help="List local steering packs",
    )
    steer_list_parser.add_argument(
        "--model",
        metavar="M",
        default=None,
        help="Path to a model directory used to mark pack compatibility",
    )

    steer_export_parser = steer_subparsers.add_parser(
        "export",
        help="Zip a local steering pack for sharing",
    )
    steer_export_parser.add_argument(
        "name",
        metavar="NAME",
        help="Pack name under the steering directory",
    )
    steer_export_parser.add_argument(
        "-o",
        "--output",
        required=True,
        metavar="FILE",
        help="Output .zip path",
    )

    steer_import_parser = steer_subparsers.add_parser(
        "import",
        help="Import a pack from a .zip, directory, HF repo, or bare .f32",
    )
    steer_import_parser.add_argument(
        "source",
        metavar="SRC",
        help="Path to a .zip or pack directory, an HF repo id, or a bare .f32 file",
    )
    steer_import_parser.add_argument(
        "--name",
        metavar="N",
        default=None,
        help="Name for the imported pack (default: from the manifest)",
    )
    steer_import_parser.add_argument(
        "--json",
        dest="json_manifest",
        default=None,
        metavar="FILE",
        help="Base manifest for a bare .f32 import",
    )
    steer_import_parser.add_argument(
        "--model",
        default=None,
        metavar="M",
        help="Model directory providing the vector shape for a bare .f32 import",
    )

    # Split launch's forwarding separator before argparse. parse_known_args()
    # inconsistently retains it when known options precede it, and stripping it
    # afterward cannot distinguish it from a separator intended for the tool.
    argv = sys.argv[1:]
    if argv[:1] == ["launch"] and "--" in argv[2:]:
        separator_index = argv.index("--", 2)
        args, extra_args = parser.parse_known_args(argv[:separator_index])
        extra_args.extend(argv[separator_index + 1 :])
    else:
        args, extra_args = parser.parse_known_args(argv)

    if args.command == "launch":
        launch_command(args, extra_args=extra_args)
    else:
        if extra_args:
            parser.error(f"unrecognized arguments: {' '.join(extra_args)}")
        if args.command == "serve":
            if (
                getattr(args, "memory_guard", None) == "off"
                and getattr(args, "memory_guard_gb", None) is not None
            ):
                parser.error(
                    "--memory-guard off cannot be combined with "
                    "--memory-guard-gb (a custom ceiling needs the guard on)"
                )
            serve_command(args)
        elif args.command in {"start", "stop", "restart"}:
            sys.exit(lifecycle_command(args))
        elif args.command == "diagnose":
            sys.exit(diagnose_command(args))
        elif args.command == "cluster":
            sys.exit(cluster_command(args))
        elif args.command == "steer":
            sys.exit(steer_command(args))
        else:
            parser.print_help()
            sys.exit(1)


if __name__ == "__main__":
    main()
