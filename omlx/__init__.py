# SPDX-License-Identifier: Apache-2.0
"""
omlx: LLM inference server, optimized for your Mac

This package provides native Apple Silicon GPU acceleration using
Apple's MLX framework and mlx-lm for LLMs.

Features:
- Continuous batching via vLLM-style scheduler
- OpenAI-compatible API server
- Paged KV cache with prefix sharing
- Tiered cache (GPU + paged SSD offloading)
"""

import sys
import platform

# Install MLX and Harmony mocks for non-macOS environments BEFORE any other imports.
# This ensures that any module in the package (or its dependencies) that
# attempts to import 'mlx' or 'openai_harmony' gets the mock implementation.
if platform.system() != "Darwin":
    try:
        # Use absolute-like path discovery to avoid premature package loading
        import importlib.util
        import os
        
        # Path to mlx_mock.py
        mock_path = os.path.join(os.path.dirname(__file__), "utils", "mlx_mock.py")
        if os.path.exists(mock_path):
            spec = importlib.util.spec_from_file_location("omlx.utils.mlx_mock", mock_path)
            if spec and spec.loader:
                mock_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mock_module)
                mock_module.install_mock()
    except Exception:
        # Silently fail as this is a convenience for non-macOS
        pass

from omlx._version import __version__

_LAZY = {
    "Request": "omlx.request",
    "RequestOutput": "omlx.request",
    "RequestStatus": "omlx.request",
    "SamplingParams": "omlx.request",
    "Scheduler": "omlx.scheduler",
    "SchedulerConfig": "omlx.scheduler",
    "SchedulerOutput": "omlx.scheduler",
    "EngineCore": "omlx.engine_core",
    "AsyncEngineCore": "omlx.engine_core",
    "EngineConfig": "omlx.engine_core",
    "BlockAwarePrefixCache": "omlx.cache.prefix_cache",
    "PagedCacheManager": "omlx.cache.paged_cache",
    "CacheBlock": "omlx.cache.paged_cache",
    "BlockTable": "omlx.cache.paged_cache",
    "PrefixCacheStats": "omlx.cache.stats",
    "PagedCacheStats": "omlx.cache.stats",
    "CacheStats": "omlx.cache.stats",
    "get_registry": "omlx.model_registry",
    "ModelOwnershipError": "omlx.model_registry",
}


def __getattr__(name: str):
    import importlib
    if name in _LAZY:
        mod = importlib.import_module(_LAZY[name])
        attr = "PagedCacheStats" if name == "CacheStats" else name
        val = getattr(mod, attr)
        globals()[name] = val
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    # Request management
    "Request",
    "RequestOutput",
    "RequestStatus",
    "SamplingParams",
    # Scheduler
    "Scheduler",
    "SchedulerConfig",
    "SchedulerOutput",
    # Engine
    "EngineCore",
    "AsyncEngineCore",
    "EngineConfig",
    # Model registry
    "get_registry",
    "ModelOwnershipError",
    # Prefix cache (paged SSD-only)
    "BlockAwarePrefixCache",
    # Paged cache (memory efficiency)
    "PagedCacheManager",
    "CacheBlock",
    "BlockTable",
    "PagedCacheStats",
    "CacheStats",  # Backward compatibility alias
    # Version
    "__version__",
]
