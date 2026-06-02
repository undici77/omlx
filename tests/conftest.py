# SPDX-License-Identifier: Apache-2.0
"""
Pytest configuration and fixtures for oMLX tests.

This module provides common fixtures used across test files.
"""

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
from omlx.request import Request, SamplingParams


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
