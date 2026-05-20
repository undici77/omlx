# SPDX-License-Identifier: Apache-2.0
"""
Robust MLX and Harmony mocking for non-macOS environments.
This module provides a mock 'mlx' and 'openai_harmony' package structure
that satisfies isinstance checks and arithmetic operations required by tests.
"""

import sys
import types
import importlib.abc
import importlib.machinery
from unittest.mock import MagicMock
import numpy as np

class MockMLXLoader(importlib.abc.Loader):
    """Loader that creates mock modules for MLX and related packages."""

    class array:
        """Mock MLX array class that simulates basic MLX array behavior."""
        def __init__(self, data=None, dtype=None):
            self.dtype = dtype or "float32"
            if data is not None:
                if isinstance(data, (list, tuple, np.ndarray, int, float, bool, np.generic)):
                    self._data = np.array(data)
                elif hasattr(data, "_data"):
                    self._data = data._data.copy()
                    self.dtype = getattr(data, "dtype", self.dtype)
                elif isinstance(data, MagicMock):
                     self._data = np.zeros((128, 128))
                else:
                    try:
                        self._data = np.array(data)
                    except:
                        self._data = np.zeros((128, 128))
            else:
                self._data = np.zeros((128, 128)) # Large enough for most tests

            self.shape = self._data.shape
            self.size = self._data.size

        def view(self, dtype): return self
        def reshape(self, *args):
            if len(args) == 1 and isinstance(args[0], (tuple, list)):
                new_shape = tuple(args[0])
            else:
                new_shape = args
            try:
                return self.__class__(self._data.reshape(new_shape), dtype=self.dtype)
            except:
                return self

        def transpose(self, *axes):
            if not axes: return self.__class__(self._data.T, dtype=self.dtype)
            if len(axes) == 1 and isinstance(axes[0], (list, tuple)):
                axes = axes[0]
            try:
                return self.__class__(self._data.transpose(*axes), dtype=self.dtype)
            except:
                return self.__class__(self._data.T, dtype=self.dtype)

        def squeeze(self, axis=None):
            return self.__class__(self._data.squeeze(axis=axis), dtype=self.dtype)

        def astype(self, dtype):
            return self.__class__(self._data, dtype=str(dtype))

        def item(self):
            if self._data.size == 0: return 0.0
            return self._data.flat[0].item()

        def tolist(self):
            return self._data.tolist()

        def get(self, key, default=None):
            if key == "max_buffer_length": return 1 << 30
            return default

        def __getitem__(self, idx):
            try:
                res = self._data[idx]
                if isinstance(res, np.ndarray):
                    return self.__class__(res, dtype=self.dtype)
                if self.size >= 1 and isinstance(idx, int):
                    return self.__class__([res], dtype=self.dtype)
                return res
            except:
                return self

        def __setitem__(self, idx, value):
            if hasattr(value, "_data"):
                value = value._data
            try:
                self._data[idx] = value
            except:
                pass

        def __len__(self):
            if self._data.ndim == 0: return 0
            return len(self._data)

        def __iter__(self):
            if self._data.ndim == 0:
                yield self.item()
                return
            for x in self._data:
                if isinstance(x, np.ndarray):
                    yield self.__class__(x, dtype=self.dtype)
                else:
                    yield x

        def __array__(self, dtype=None):
            return self._data.astype(dtype) if dtype else self._data

        def __repr__(self): return f"mx.array(shape={self.shape}, dtype={self.dtype})"

        # Arithmetic
        def __add__(self, other): return self.__class__(self._data + (other._data if hasattr(other, "_data") else other))
        def __sub__(self, other): return self.__class__(self._data - (other._data if hasattr(other, "_data") else other))
        def __mul__(self, other): return self.__class__(self._data * (other._data if hasattr(other, "_data") else other))
        def __truediv__(self, other): return self.__class__(self._data / (other._data if hasattr(other, "_data") else other))
        def __pow__(self, other): return self.__class__(self._data ** (other._data if hasattr(other, "_data") else other))
        def __matmul__(self, other): return self.__class__(self._data @ (other._data if hasattr(other, "_data") else other))

        def __neg__(self): return self.__class__(-self._data, dtype=self.dtype)

        def __radd__(self, other): return self.__add__(other)
        def __rsub__(self, other): return self.__class__((other._data if hasattr(other, "_data") else other) - self._data)
        def __rmul__(self, other): return self.__mul__(other)
        def __rtruediv__(self, other): return self.__class__((other._data if hasattr(other, "_data") else other) / self._data)
        def __rpow__(self, other): return self.__class__((other._data if hasattr(other, "_data") else other) ** self._data)

        def __lt__(self, other): return self.__class__(self._data < (other._data if hasattr(other, "_data") else other))
        def __le__(self, other): return self.__class__(self._data <= (other._data if hasattr(other, "_data") else other))
        def __gt__(self, other): return self.__class__(self._data > (other._data if hasattr(other, "_data") else other))
        def __ge__(self, other): return self.__class__(self._data >= (other._data if hasattr(other, "_data") else other))

        def min(self, axis=None, keepdims=False):
            res = self._data.min(axis=axis, keepdims=keepdims)
            return self.__class__(res) if isinstance(res, np.ndarray) else res
        def max(self, axis=None, keepdims=False):
            res = self._data.max(axis=axis, keepdims=keepdims)
            return self.__class__(res) if isinstance(res, np.ndarray) else res
        def sum(self, axis=None, keepdims=False):
            res = self._data.sum(axis=axis, keepdims=keepdims)
            return self.__class__(res) if isinstance(res, np.ndarray) else res
        def mean(self, axis=None, keepdims=False):
            res = self._data.mean(axis=axis, keepdims=keepdims)
            return self.__class__(res) if isinstance(res, np.ndarray) else res

        @property
        def ndim(self): return self._data.ndim
        @property
        def T(self): return self.__class__(self._data.T, dtype=self.dtype)
        @property
        def nbytes(self): return self._data.nbytes

    def create_module(self, spec):
        if spec.name in sys.modules:
            return sys.modules[spec.name]

        class MockModule(types.ModuleType):
            def __init__(self, name, loader):
                super().__init__(name)
                self.__mock_items = {}
                self._loader = loader
                self.__spec__ = importlib.machinery.ModuleSpec(name, None)

            def __getattr__(self, name):
                if name in ("__path__", "__file__"):
                    return None
                if name == "__all__":
                    return []

                if name not in self.__mock_items:
                    # Smart dispatch for common MLX functions
                    if self.__name__ == "mlx.core":
                        if name == "logsumexp":
                            def _logsumexp(a, axis=None, keepdims=False):
                                a_np = np.array(a)
                                m = a_np.max(axis=axis, keepdims=True)
                                res = m + np.log(np.sum(np.exp(a_np - m), axis=axis, keepdims=keepdims))
                                return self._loader.array(res)
                            self.__mock_items[name] = _logsumexp
                        elif name == "argmax":
                            def _argmax(a, axis=None, keepdims=False):
                                a_np = np.array(a)
                                if a_np.size == 0: return self._loader.array(0)
                                return self._loader.array(np.argmax(a_np, axis=axis), dtype="int32")
                            self.__mock_items[name] = _argmax
                        elif name == "argsort":
                            def _argsort(a, axis=-1):
                                return self._loader.array(np.argsort(np.array(a), axis=axis), dtype="int32")
                            self.__mock_items[name] = _argsort
                        elif name == "sort":
                            def _sort(a, axis=-1):
                                return self._loader.array(np.sort(np.array(a), axis=axis))
                            self.__mock_items[name] = _sort
                        elif name == "argpartition":
                            def _argpartition(a, kth, axis=-1):
                                return self._loader.array(np.argpartition(np.array(a), kth, axis=axis), dtype="int32")
                            self.__mock_items[name] = _argpartition
                        elif name == "sum":
                            def _sum(a, axis=None, keepdims=False):
                                return self._loader.array(np.sum(np.array(a), axis=axis, keepdims=keepdims))
                            self.__mock_items[name] = _sum
                        elif name == "cumsum":
                            def _cumsum(a, axis=None, **kwargs):
                                return self._loader.array(np.cumsum(np.array(a), axis=axis))
                            self.__mock_items[name] = _cumsum
                        elif name == "take_along_axis":
                            def _take(a, indices, axis, **kwargs):
                                return self._loader.array(np.take_along_axis(np.array(a), np.array(indices).astype(int), axis))
                            self.__mock_items[name] = _take
                        elif name == "put_along_axis":
                            def _put(a, indices, values, axis, **kwargs):
                                a_np = np.array(a).copy()
                                np.put_along_axis(a_np, np.array(indices).astype(int), np.array(values), axis)
                                return self._loader.array(a_np)
                            self.__mock_items[name] = _put
                        elif name in ("zeros", "ones"):
                            def _const(shape, dtype=None, **kwargs):
                                val = 0 if name == "zeros" else 1
                                return self._loader.array(np.full(shape, val, dtype=dtype))
                            self.__mock_items[name] = _const
                        elif name in ("zeros_like", "ones_like"):
                            def _like(a, **kwargs):
                                a_np = np.array(a)
                                val = 0 if name == "zeros_like" else 1
                                return self._loader.array(np.full_like(a_np, val))
                            self.__mock_items[name] = _like
                        elif name == "arange":
                            def _arange(*args, **kwargs):
                                return self._loader.array(np.arange(*args, **kwargs), dtype="int32")
                            self.__mock_items[name] = _arange
                        elif name == "exp":
                            def _exp(a): return self._loader.array(np.exp(np.array(a)))
                            self.__mock_items[name] = _exp
                        elif name == "log":
                            def _log(a): return self._loader.array(np.log(np.array(a)))
                            self.__mock_items[name] = _log
                        elif name == "abs":
                            def _abs(a): return self._loader.array(np.abs(np.array(a)))
                            self.__mock_items[name] = _abs
                        elif name == "maximum":
                            def _maximum(a, b): return self._loader.array(np.maximum(np.array(a), np.array(b)))
                            self.__mock_items[name] = _maximum
                        elif name == "minimum":
                            def _minimum(a, b): return self._loader.array(np.minimum(np.array(a), np.array(b)))
                            self.__mock_items[name] = _minimum
                        elif name == "where":
                            def _where(c, a, b):
                                c_np = np.array(c)
                                a_np = np.array(a)
                                b_np = np.array(b)
                                return self._loader.array(np.where(c_np, a_np, b_np))
                            self.__mock_items[name] = _where
                        elif name in ("floor", "ceil", "round"):
                            def _round(a):
                                f = getattr(np, name)
                                return self._loader.array(f(np.array(a)))
                            self.__mock_items[name] = _round
                        elif name == "all":
                            def _all(a, **kwargs): return self._loader.array(np.all(np.array(a), **kwargs))
                            self.__mock_items[name] = _all
                        elif name == "any":
                            def _any(a, **kwargs): return self._loader.array(np.any(np.array(a), **kwargs))
                            self.__mock_items[name] = _any
                        elif name == "issubdtype":
                            def _issubdtype(a, b):
                                if b == "floating": return "float" in str(a)
                                if b == "integer": return "int" in str(a)
                                return str(a) == str(b)
                            self.__mock_items[name] = _issubdtype

                    if name not in self.__mock_items:
                        if name == "random" and self.__name__ == "mlx.core":
                            # Return a MockModule for random
                            return self._loader.create_module(importlib.machinery.ModuleSpec("mlx.core.random", self._loader))

                        m = MagicMock()
                        def _default_mock_func(*args, **kwargs):
                            # Try to infer shape from first array argument
                            for arg in args:
                                if hasattr(arg, "shape"):
                                    return self._loader.array(np.zeros(arg.shape))
                            return self._loader.array() # Default large shape
                        m.side_effect = _default_mock_func
                        self.__mock_items[name] = m

                return self.__mock_items[name]

            def __call__(self, *args, **kwargs):
                return MagicMock()(*args, **kwargs)

        # Handle random sub-module specially if it's being created directly
        if spec.name == "mlx.core.random":
            m = MockModule(spec.name, self)
            state_arr = self.array([0, 0])
            def _uniform(low=0.0, high=1.0, shape=None, **kwargs):
                state_arr._data[0] += 1
                return self.array(np.random.uniform(low, high, size=shape or ()))
            def _categorical(logits, num_samples=1, axis=-1, **kwargs):
                state_arr._data[0] += 1
                l_np = np.array(logits)
                # Stochastic enough for tests
                if num_samples == 1:
                    res = np.random.randint(0, l_np.shape[axis] if l_np.shape[axis] > 0 else 1, size=l_np.shape[:axis] + l_np.shape[axis+1:])
                else:
                    res = np.random.randint(0, l_np.shape[axis] if l_np.shape[axis] > 0 else 1, size=num_samples)
                return self.array(res, dtype="int32")
            m.uniform = _uniform
            m.categorical = _categorical
            m.seed = MagicMock()
            m.state = state_arr
            sys.modules[spec.name] = m
            return m

        m = MockModule(spec.name, self)
        sys.modules[spec.name] = m
        return m

    def exec_module(self, module):
        if module.__name__ == "mlx.core":
            module.float32 = "float32"
            module.float16 = "float16"
            module.bfloat16 = "bfloat16"
            module.int32 = "int32"
            module.int64 = "int64"
            module.uint64 = "uint64"
            module.bool_ = "bool"
            module.floating = "floating"
            module.integer = "integer"
            module.inf = float("inf")
            module.nan = float("nan")
            module.array = self.array
            module.allclose = MagicMock(return_value=self.array([True]))
            module.quantize = MagicMock(return_value=(self.array(), self.array(), self.array()))
            module.power = MagicMock(return_value=self.array())
            module.metal = MagicMock()
            module.metal.is_available = MagicMock(return_value=False)
            module.metal.device_info = MagicMock(return_value={"max_buffer_length": 1 << 30})
            module.get_active_memory = MagicMock(return_value=0)
            module.get_peak_memory = MagicMock(return_value=0)
        elif module.__name__ == "mlx.nn":
            module.Module = MagicMock
        elif module.__name__ == "openai_harmony":
            module.load_harmony_encoding = MagicMock()
            module.StreamableParser = MagicMock()
            module.Role = MagicMock()

class MockMLXFinder(importlib.abc.MetaPathFinder):
    """Finder that routes MLX and Harmony imports to the MockMLXLoader."""
    def find_spec(self, fullname, path, target=None):
        if fullname == "mlx" or fullname.startswith("mlx.") or fullname == "openai_harmony":
            return importlib.machinery.ModuleSpec(fullname, MockMLXLoader())
        return None

def install_mock():
    """Install the mock into sys.meta_path."""
    try:
        import mlx.core
    except ImportError:
        if not any(isinstance(f, MockMLXFinder) for f in sys.meta_path):
            sys.meta_path.insert(0, MockMLXFinder())
