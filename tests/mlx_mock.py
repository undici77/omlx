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

class MockMLXLoader(importlib.abc.Loader):
    """Loader that creates mock modules for MLX and related packages."""
    
    class array:
        """Mock MLX array class that simulates basic MLX array behavior."""
        def __init__(self, data=None, dtype=None):
            self.dtype = dtype or "float32"
            self._data = data
            if hasattr(data, "shape"):
                self.shape = data.shape
                self.size = data.size
            elif isinstance(data, (list, tuple)):
                try:
                    import numpy as np
                    a = np.array(data)
                    self.shape = a.shape
                    self.size = a.size
                    if a.dtype == bool: self.dtype = "bool"
                except:
                    self.shape = (len(data),)
                    self.size = len(data)
            elif isinstance(data, (int, float, bool)):
                self.shape = ()
                self.size = 1
                if isinstance(data, bool): self.dtype = "bool"
            else:
                self.shape = (8, 8) # Default mock shape
                self.size = 64
        
        def view(self, dtype): return self
        def reshape(self, *args):
            if len(args) == 1 and isinstance(args[0], (tuple, list)):
                self.shape = tuple(args[0])
            else:
                self.shape = args
            return self
        def astype(self, dtype):
            self.dtype = str(dtype)
            return self
        def item(self):
            if self.dtype == "bool":
                return bool(self._data)
            if isinstance(self._data, (list, tuple)) and len(self._data) == 1:
                return self._data[0]
            if isinstance(self._data, (int, float)):
                return self._data
            return 0.0
        def tolist(self):
            if isinstance(self._data, list):
                return self._data
            if isinstance(self._data, (int, float, bool)):
                return [self._data]
            return []
        def get(self, key, default=None):
            if key == "max_buffer_length": return 1 << 30
            return default
        def __getitem__(self, idx): return self
        def __len__(self): return self.shape[0] if self.shape else 0
        def __iter__(self): return iter([self, self, self, self, self, self])
        def __repr__(self): return f"mx.array(shape={self.shape}, dtype={self.dtype})"
        def __add__(self, other): return self
        def __sub__(self, other): return self
        def __mul__(self, other): return self
        def __truediv__(self, other): return self
        def __radd__(self, other): return self
        def __rsub__(self, other): return self
        def __rmul__(self, other): return self
        def __rtruediv__(self, other): return self
        def __lt__(self, other): return False
        def __le__(self, other): return True
        def __gt__(self, other): return False
        def __ge__(self, other): return True
        @property
        def ndim(self): return len(self.shape)
        @property
        def T(self): return self

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
                    m = MagicMock()
                    # By default, mock functions return a mock array
                    m.return_value = self._loader.array()
                    self.__mock_items[name] = m
                return self.__mock_items[name]

            def __call__(self, *args, **kwargs):
                return MagicMock()(*args, **kwargs)

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
            module.bool_ = "bool"
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
