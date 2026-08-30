# SPDX-License-Identifier: Apache-2.0
"""
Pytest configuration and fixtures for oMLX tests.

This module provides common fixtures used across test files.
"""

import importlib.util
import os
import sys
import types
import importlib.machinery
import importlib.abc
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import omlx  # Triggers global MLX mock installation for non-macOS
import pytest

import mlx.core as mx
import openai_harmony

# Install the torch stub before any test imports xgrammar (e.g. via @patch
# decorators that resolve the target at collection time). When real torch is
# present this is a no-op; in the DMG layout it satisfies xgrammar's
# import-time torch references so the package can load.
from omlx._torch_stub import install as _install_torch_stub
_install_torch_stub()

# Run tests under the same M5 sorted gather_qmm reroute the server
# installs at model load (issue #2267). Without it, kernel-sensitive
# tests (e.g. the SwitchGLU fusion bit-exactness test, whose inter=32
# down_proj runs at K=32) fail on M5 hardware. No-op elsewhere.
from omlx.patches.m5_gather_qmm import apply_m5_gather_qmm_workaround
apply_m5_gather_qmm_workaround()

from omlx.request import Request, SamplingParams


# ── Auto-skip Harmony-dependent tests when MLX mock is active ────────────────
# When the MLX mock is active (non-macOS/CI), openai_harmony is a NumPy-backed
# mock that cannot produce the exact token IDs the real library generates.
# Tests that depend on real Harmony token IDs would fail spuriously, so we
# skip them automatically — zero changes to test files needed.
_MOCKS_HARMONY = any(
    "MockMLXFinder" in str(f) for f in sys.meta_path
)


def _real_openai_harmony_available() -> bool:
    """Check if the real openai_harmony library is importable (not mocked)."""
    # If we're on macOS, the real library is available.
    if sys.platform == "darwin":
        return True
    # On Linux, check if the spec points to a real package (not our mock).
    spec = importlib.util.find_spec("openai_harmony")
    if spec is None:
        return False
    origin = spec.origin or ""
    # The mock's loader is MockMLXLoader; the real package has a .py file.
    return ".py" in origin and "mlx_mock" not in origin


def pytest_collection_modifyitems(config, items):
    """Skip tests that need real Harmony token IDs when the mock is active."""
    if _real_openai_harmony_available():
        return  # Real library available — run everything

    skip_marker = pytest.mark.skip(
        reason="MLX mock active — real openai_harmony encoding unavailable; "
               "token IDs would be incorrect",
    )

    for item in items:
        mod_file = getattr(item.module, "__file__", "") or ""
        mod_name = getattr(item.module, "__name__", "") or ""
        item_name = getattr(item, "name", "") or ""

        # Skip test files that import from omlx.adapter.harmony
        if "omlx.adapter.harmony" in mod_name:
            item.add_marker(skip_marker)
        # Also skip test files with "harmony" in their path/name
        elif "harmony" in mod_file.lower() or "harmony" in mod_name.lower():
            item.add_marker(skip_marker)
        # Skip test methods whose name contains "harmony" (inline harmony tests)
        elif "harmony" in item_name.lower():
            item.add_marker(skip_marker)

    # Skip xgrammar stub test on non-macOS — it requires a real torch stub env
    # that isn't available in this CI setup (pre-existing failure).
    xgrammar_skip = pytest.mark.skip(
        reason="xgrammar stub test requires real torch env — skip on non-macOS CI",
    )
    for item in items:
        if "test_xgrammar_imports_against_stub_only" in (getattr(item, "name", "") or ""):
            if sys.platform != "darwin" and not os.environ.get("CI_XGRAMMAR_TEST"):
                item.add_marker(xgrammar_skip)

    # ── Skip MLX-dependent tests when the mock is active ──────────────────────
    # On Linux/CI the MLX mock (NumPy-backed) cannot reproduce real MLX
    # internals such as RotatingKVCache, GenerationBatch.filter, quantize
    # return shapes, or the Gemma-4 real regex parser.  Rather than maintain
    # a fragile per-function shim we skip entire test files whose assertions
    # depend on real MLX behaviour.  This follows the project rule: "mlx test
    # must be skipped" on non-macOS — keep the mock minimal and maintainable.
    _MLX_SKIP_REASONS: list[tuple[str, str]] = [
        # MLX cache / MTP internals (RotatingKVCache, PoolingCache, etc.)
        ("test_mlx_lm_mtp_patch", "MLX mock active — RotatingKVCache internals unavailable"),
        ("test_deepseek_v4_patch", "MLX mock active — PoolingCache / cache patch internals unavailable"),
        ("test_mlx_vlm_diffusion_patch", "MLX mock active — MLX dequantize unavailable"),
        ("test_vlm_mtp", "MLX mock active — VLM MTP / MoE config internals unavailable"),
        # Quantisation & packing (mock returns wrong tuple shapes)
        ("test_oq", "MLX mock active — mx.quantize return shape differs from real MLX"),
        # TurboQuant (MLX quantisation internals)
        ("test_turboquant", "MLX mock active — TurboQuant MLX internals unavailable"),
        # Embedding / model loading (requires real MLX model graph)
        ("test_embedding", "MLX mock active — native MLX model loading unavailable"),
        # Cache type handlers (MLX array internals)
        ("test_cache_type_handlers", "MLX mock active — MLX cache handler internals unavailable"),
        # Reranker (MLX model.train() method)
        ("test_reranker_causal_lm", "MLX mock active — MLX model.train() unavailable"),
        # MiniMax M3 sparse attention (MLX operator internals)
        ("test_minimax_m3_sparse_attention_patch", "MLX mock active — _build_sparse_mask unavailable"),
        # Scheduler logits processor alignment (GenerationBatch.filter broken in mock)
        ("test_scheduler_logits_processors", "MLX mock active — GenerationBatch.filter unavailable in mock"),
        # Gemma-4 real parser (uses MLX regex engine)
        ("test_tool_calling", "MLX mock active — Gemma-4 real parser uses MLX regex"),
        # New MLX-dependent test files failing with mock
        ("test_glm_moe_dsa_patch", "MLX mock active — GLM MoE/DSA patch internals unavailable"),
        ("test_mlx_vlm_minimax_m3_compat", "MLX mock active — MiniMax M3 loader/architecture fallback unavailable"),
        ("test_qwen35_fa256_attention", "MLX mock active — Qwen 3.5 attention patch internals unavailable"),
        ("test_qwen35_gdn_prefill", "MLX mock active — Qwen 3.5 prefill patch internals unavailable"),
        ("test_qwen35_moe_weighted_sum", "MLX mock active — Qwen 3.5 MoE weighted sum unavailable"),
        ("test_qwen35_q4_mlp", "MLX mock active — Qwen 3.5 Q4 MLP patch internals unavailable"),
        ("test_scheduler", "MLX mock active — Scheduler SSD/TurboQuant cache layout signatures require real MLX"),
        ("test_sdpa256_attention", "MLX mock active — SDPA 256 attention patch internals unavailable"),
        ("test_glm_mtp_patch", "MLX mock active — GLM MTP patch internals unavailable"),
        # Bonsai quantized kernels (quantized_matmul, to_fp8, etc.)
        ("test_bonsai_qmv", "MLX mock active — Bonsai quantized kernel ops unavailable"),
        ("test_bonsai_t5_load", "MLX mock active — Bonsai T5 quantized ops unavailable"),
        # Laguna (FP8/nvfp4 sanitize requires real MLX dequantize)
        ("test_laguna_patch", "MLX mock active — Laguna sanitize requires real MLX"),
        # Vendored VLM patches (need real mlx-vlm model classes)
        ("test_mlx_vlm_pixtral_torch_free", "MLX mock active — vendored mlx-vlm classes unavailable"),
        ("test_mlx_vlm_unlimited_ocr_compat", "MLX mock active — vendored mlx-vlm classes unavailable"),
        # Nemotron-H MTP (needs real NemotronH mixer class)
        ("test_nemotron_mtp_patch", "MLX mock active — NemotronH mixer class unavailable in mock"),
        # Qwen3.5 MoE fused gate_up (needs real SwitchGLU / mlx.randint)
        ("test_qwen35_moe_gate_up", "MLX mock active — SwitchGLU / randint unavailable in mock"),
        # Prefix cache TurboQuant reconstruction (needs real MLX array ops)
        ("test_prefix_cache", "MLX mock active — TQ reconstruction needs real MLX array ops"),
        ("test_per_engine_threads", "MLX mock active — prefix cache stream ops need real MLX"),
        # Model loading (materialize_lazy_state needs real tree_flatten on mlx.array)
        ("test_model_loading", "MLX mock active — lazy state materialization needs real MLX"),
        # Server main (integration-style, depends on full server stack)
        ("test_server_main", "MLX mock active — server entry point needs full stack"),
        # New MLX-dependent test files from merge
        ("test_deepseek_v4_dspark", "MLX mock active — dSpark quantized kernel ops unavailable"),
        ("test_gemma4_verify_attention", "MLX mock active — Gemma-4 verify kernel unavailable"),
        ("test_gemma4_vlm_mtp_runtime", "MLX mock active — Gemma-4 VLM MTP internals unavailable"),
        ("test_mlx_vlm_inkling_compat", "MLX mock active — Inkling VLM vendor classes unavailable"),
        ("test_mtp_prompt_priming", "MLX mock active — MTP prompt priming needs real MLX model"),
        ("test_step3p7_patch", "MLX mock active — Step3.7 MTP sanitize needs real MLX"),
        ("test_dflash_laguna", "MLX mock active — DFlash/Laguna sanitize needs real MLX"),
        ("test_gemma4_text_model", "MLX mock active — Gemma-4 text model needs real MLX"),
        ("test_inkling_vlm_mtp", "MLX mock active — Inkling VLM MTP needs real MLX"),
        ("test_mimo_v2_patch", "MLX mock active — MIMO v2 quantized ops unavailable"),
        ("test_pooling_cache_delta", "MLX mock active — PoolingCache delta needs real MLX"),
        ("test_cache_ntuple_state", "MLX mock active — PoolingCache state arity differs in mock"),
        ("test_specprefill", "MLX mock active — specprefill array ops need real MLX"),
        ("test_engine_keepalive", "MLX mock active — MLXEmbeddingModel compile needs real MLX"),
        ("test_utils_tokenizer", "MLX mock active — tokenizer decode needs real MLX"),
        # New MLX-dependent test files from merge (Bailing / hybrid attention)
        ("test_bailing_hybrid_patch", "MLX mock active — BatchGenerator.insert / logaddexp / recurrent_kda / to_fp8 unavailable"),
        ("test_bailing_swiglu_clamp", "MLX mock active — mlx.nn / tree_flatten unavailable in mock"),
        # Cluster tests (need real MLX for pipeline smoke and model loading)
        ("test_cluster_autoconfigure", "MLX mock active — cluster autoconfigure needs real MLX"),
        ("test_cluster_cli", "MLX mock active — cluster CLI pipeline smoke needs real MLX"),
        ("test_cluster_inference_worker", "MLX mock active — cluster inference worker needs real MLX"),
        ("test_cluster_memory_guard", "MLX mock active — cluster memory guard needs real MLX"),
        ("test_cluster_performance", "MLX mock active — cluster performance tests need real MLX"),
        ("test_cluster_pipeline_compat", "MLX mock active — cluster pipeline compat needs real MLX"),
        ("test_cluster_progressive_loading", "MLX mock active — cluster progressive loading needs real MLX"),
        ("test_cluster_prompt_snapshot_cache", "MLX mock active — cluster prompt snapshot cache needs real MLX"),
        ("test_cluster_prompt_snapshot_integration", "MLX mock active — cluster prompt snapshot integration needs real MLX"),
        ("test_cluster_remote_planning", "MLX mock active — cluster remote planning needs real MLX"),
        ("test_cluster_telemetry", "MLX mock active — cluster telemetry needs real MLX"),
        # Hardware-dependent / probe tests (require real MLX or real hardware)
        ("test_cluster_probe", "MLX mock active — cluster RDMA probe requires real hardware on non-macOS"),
        ("test_custom_kernel_abi_probe", "MLX mock active — custom kernel ABI probe needs real MLX arrays"),
        # VLM / DFlash / quantisation tests
        ("test_dflash_muse_glimmer", "MLX mock active — DFlash/MuseGlimmer needs real MLX"),
        ("test_gdn_sidecar_quantization", "MLX mock active — hadamard_transform / isfinite unavailable in mock"),
        ("test_mlx_vlm_muse_glimmer_compat", "MLX mock active — MuseGlimmer loader/architecture unavailable in mock"),
        ("test_pooling_cache_append_inplace", "MLX mock active — PoolingCache append ops need real MLX arrays"),
        # MiniMax M3 pipeline (needs real MLX Model with .model attribute)
        ("test_minimax_m3_pipeline", "MLX mock active — MiniMax M3 pipeline needs real MLX model graph"),
        # Qwen3 sliding window (needs real ModelArgs with full constructor)
        ("test_qwen3_sliding_window_patch", "MLX mock active — Qwen3 ModelArgs constructor needs real MLX"),
        # MiniMax M3 mlx-lm patch (needs real MLX Model with .layers / nested config)
        ("test_minimax_m3_mlx_lm_patch", "MLX mock active — MiniMax M3 mlx-lm patch needs real MLX model graph"),
        # Qwen3.5 ANE prefill (needs QuantizedLinear.scales / nn.Module.modules)
        ("test_qwen35_ane_prefill", "MLX mock active — QuantizedLinear.scales and nn.Module.modules unavailable"),
        # New MLX-dependent test files from merge (GLM-5 Next / Qwen4 Exp compat)
        ("test_clone_mlx_model_fp16", "MLX mock active — mlx.isfinite() unavailable in mock"),
        ("test_mlx_vlm_glm5_next_compat", "MLX mock active — GLM-5 Next VLM vendor classes / PoolingCache / native indexer unavailable"),
        ("test_mlx_vlm_qwen4_exp_compat", "MLX mock active — Qwen4 Exp configure_ple_runtime/configure_mtp_runtime/to_fp8 unavailable"),
        ("test_vlm_qwen4_exp_loader", "MLX mock active — RotatingKVCache.is_trimmable / configure_ple_runtime unavailable"),
        # Qwen4 QSA decode/gather/native-indexer (native MLX decode kernels)
        ("test_qwen4_qsa_decode_gather", "MLX mock active — _decode_qsa_sdpa / contiguous_causal_gathered_qsa unavailable in mock"),
        ("test_qwen4_qsa_incremental_cache", "MLX mock active — contiguous_causal_gathered_qsa / array.swapaxes unavailable in mock"),
        ("test_qwen4_qsa_native_indexer", "MLX mock active — _native_indexer_scores native kernel unavailable in mock"),
        ("test_qwen4_qsa_sparse_gqa", "MLX mock active — _native_sparse_gqa_attention native kernel unavailable in mock"),
        ("test_qwen4_hc_projection", "MLX mock active — fuse_hyper_connection_projections native kernel unavailable in mock"),
        # Qwen4 QSA prefill memory routing (real MLX array precision + cache internals)
        ("test_qwen4_qsa_prefill_memory", "MLX mock active — tiled-SDPA precision & TurboQuant cache internals unavailable in mock"),
        # Qwen3.5 GDN fused prework parity (real mlx.nn / model-graph internals)
        ("test_qwen35_gdn_prework", "MLX mock active — fused GDN prework needs real mlx.nn / model-graph internals"),
        # Qwen4 vision grid compat (mlx.repeat unpacking + vision-grid MLX internals)
        ("test_qwen4_vision_grid_compat", "MLX mock active — vision grid MLX array internals unavailable in mock"),
    ]

    _mock_skip = pytest.mark.skip(
        reason="MLX mock active on non-macOS — real MLX internals unavailable; "
               "see tests/conftest.py _MLX_SKIP_REASONS",
    )

    for item in items:
        mod_file = getattr(item.module, "__file__", "") or ""
        for pattern, reason in _MLX_SKIP_REASONS:
            if pattern in mod_file:
                item.add_marker(pytest.mark.skip(reason=reason))
                break


class MockTokenizer:
    """Mock tokenizer for testing without loading real models."""

    def __init__(self, vocab_size: int = 32000):
        self.vocab_size = vocab_size
        self.eos_token_id = 2
        self.pad_token_id = 0
        self.bos_token_id = 1

    def encode(self, text: str, **kwargs: Any) -> List[int]:
        """Mock encoding: convert length to fake token IDs."""
        return [1] + [100] * len(text.split()) + [2]

    def decode(self, token_ids: List[int], skip_special_tokens: bool = False, **kwargs: Any) -> str:
        """Mock decoding: return fake text (simple simulation)."""
        if skip_special_tokens:
            token_ids = [
                t
                for t in token_ids
                if t not in (self.eos_token_id, self.pad_token_id, self.bos_token_id)
            ]
        # Return a placeholder string representing the token count
        return f"<decoded:{len(token_ids)} tokens>"

    def __call__(
        self,
        text: str,
        return_tensors: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Tokenize text and return dict with input_ids."""
        input_ids = self.encode(text)
        return {"input_ids": input_ids}


class MockModelConfig:
    """Mock model configuration for testing."""

    def __init__(
        self,
        hidden_size: int = 4096,
        num_hidden_layers: int = 32,
        num_attention_heads: int = 32,
        vocab_size: int = 32000,
        model_type: str = "llama",
    ):
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.vocab_size = vocab_size
        self.model_type = model_type


class MockModel:
    """Mock model for testing without loading real models."""

    def __init__(self, config: Optional[MockModelConfig] = None):
        self.config = config or MockModelConfig()
        self._parameters: Dict[str, Any] = {}
        # Mock layers for mlx-lm cache compatibility
        self.layers = [MagicMock() for _ in range(self.config.num_hidden_layers)]

    def __call__(self, input_ids: Any, **kwargs: Any) -> Any:
        """Forward pass (returns mock logits)."""
        mock_output = MagicMock()
        mock_output.shape = (1, len(input_ids) if hasattr(input_ids, "__len__") else 1, self.config.vocab_size)
        return mock_output

    def parameters(self) -> Dict[str, Any]:
        """Return model parameters."""
        return self._parameters


@pytest.fixture
def mock_tokenizer():
    """Fixture for a mock tokenizer."""
    return MockTokenizer()


@pytest.fixture
def mock_model():
    """Fixture for a mock model."""
    return MockModel()


@pytest.fixture
def temp_settings_dir(tmp_path):
    """Fixture for a temporary settings directory."""
    settings_dir = tmp_path / ".omlx"
    settings_dir.mkdir()
    return settings_dir


@pytest.fixture
def mock_request():
    """Fixture for a mock inference request."""
    return Request(
        request_id="test-req-123",
        prompt="Hello world",
        sampling_params=SamplingParams(temperature=0.7, max_tokens=20),
    )


@pytest.fixture
def sample_request_factory():
    """Factory fixture for creating multiple Request objects."""

    def _create_request(
        request_id: str = "test-request-001",
        prompt: str = "Hello, world!",
        max_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> Request:
        return Request(
            request_id=request_id,
            prompt=prompt,
            sampling_params=SamplingParams(
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            ),
        )

    return _create_request


@pytest.fixture
def real_model_dir() -> Path:
    """Return the path to real models directory.

    Note: Tests using this fixture may require actual model files
    and should be marked with @pytest.mark.slow.
    """
    return Path.home() / "Workspace" / "models"


@pytest.fixture(autouse=True)
def _reset_decode_activity_registry():
    """Keep the process-global decode-activity registry hermetic per test.

    Schedulers publish to it from step(); entries live for a short TTL, so
    without this a scheduler stepped in one test reads as cross-engine
    decode contention in the next.
    """
    from omlx.decode_activity import get_decode_activity

    get_decode_activity().clear()
    yield
    get_decode_activity().clear()
