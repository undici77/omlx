# SPDX-License-Identifier: Apache-2.0
import sys
import types
import importlib.abc
import importlib.machinery
import numpy as np

def _map_dtype(dtype):
    if dtype == "bfloat16": return "float32"
    if isinstance(dtype, str): return dtype
    if hasattr(dtype, "__name__"): return dtype.__name__
    return None

class MockMLXLoader(importlib.abc.Loader):
    class array:
        def __init__(self, data=None, dtype=None):
            if data is not None:
                if hasattr(data, "_data"):
                    self._data = data._data.copy()
                    if dtype:
                        self._data = self._data.astype(_map_dtype(dtype))
                    self.dtype = dtype or getattr(data, "dtype", "float32")
                elif isinstance(data, np.ndarray):
                    self._data = data.copy()
                    if dtype:
                        self._data = self._data.astype(_map_dtype(dtype))
                    self.dtype = dtype or str(data.dtype)
                else:
                    mapped = _map_dtype(dtype)
                    self._data = np.array(data, dtype=mapped if mapped else None)
                    self.dtype = dtype or str(self._data.dtype)
            else:
                mapped = _map_dtype(dtype)
                self._data = np.zeros((0,), dtype=mapped if mapped else "float32")
                self.dtype = dtype or "float32"
            
            self.shape = self._data.shape
            self.size = self._data.size
            self.ndim = self._data.ndim
            self.typecode = 'f' if "float" in str(self.dtype) else 'i'

        @property
        def nbytes(self): return self._data.nbytes

        @property
        def __array_interface__(self): return self._data.__array_interface__
        
        def __buffer__(self, flags): return self._data.__buffer__(flags)

        def view(self, dtype):
            return self.__class__(self._data.view(_map_dtype(dtype)), dtype=dtype)
        
        def reshape(self, *args):
            new_shape = args[0] if len(args) == 1 and isinstance(args[0], (tuple, list)) else args
            try: return self.__class__(self._data.reshape(new_shape), dtype=self.dtype)
            except: return self
            
        def transpose(self, *axes):
            if not axes:
                return self.__class__(self._data.T, dtype=self.dtype)
            if len(axes) == 1 and isinstance(axes[0], (list, tuple)):
                new_axes = axes[0]
            else:
                new_axes = axes
            try: return self.__class__(self._data.transpose(new_axes), dtype=self.dtype)
            except: return self
            
        def squeeze(self, axis=None):
            return self.__class__(self._data.squeeze(axis=axis), dtype=self.dtype)
            
        def astype(self, dtype):
            return self.__class__(self._data.astype(_map_dtype(dtype)), dtype=str(dtype))
            
        def item(self):
            return self._data.item() if self.size == 1 else self._data.flat[0].item()
            
        def tolist(self):
            return self._data.tolist()
            
        def flatten(self):
            return self.__class__(self._data.flatten(), dtype=self.dtype)

        def __getitem__(self, idx):
            res = self._data[idx]
            if isinstance(res, (np.ndarray, np.generic)):
                return self.__class__(res, dtype=self.dtype)
            return self.__class__(np.array(res), dtype=self.dtype)

        def __setitem__(self, idx, value):
            self._data[idx] = value._data if hasattr(value, "_data") else value

        def __len__(self):
            return len(self._data) if self.ndim > 0 else 0

        def __iter__(self):
            if self.ndim == 0:
                yield self
            else:
                for x in self._data:
                    yield self.__class__(x, dtype=self.dtype)

        def __array__(self, dtype=None):
            return self._data.astype(dtype) if dtype else self._data

        def __repr__(self):
            try: return f"mx.array(shape={self.shape}, dtype={self.dtype})"
            except: return "mx.array(error in repr)"

        def __int__(self): return int(self.item())
        def __index__(self): return int(self.item())
        def __float__(self): return float(self.item())
        def __bool__(self): return bool(self.item())

        def _to_np(self, other):
            if hasattr(other, "_data"): return other._data
            if isinstance(other, np.ndarray): return other
            return np.array(other)

        def __add__(self, other): return self.__class__(self._data + self._to_np(other))
        def __sub__(self, other): return self.__class__(self._data - self._to_np(other))
        def __mul__(self, other): return self.__class__(self._data * self._to_np(other))
        def __truediv__(self, other): return self.__class__(self._data / self._to_np(other))
        def __floordiv__(self, other): return self.__class__(self._data // self._to_np(other))
        def __pow__(self, other): return self.__class__(self._data ** self._to_np(other))
        def __rpow__(self, other): return self.__class__(self._to_np(other) ** self._data)
        def __matmul__(self, other): return self.__class__(self._data @ self._to_np(other))
        def __neg__(self): return self.__class__(-self._data)
        
        def __lshift__(self, other):
            return self.__class__(self._data.astype(np.int64) << self._to_np(other).astype(np.int64))
        def __rshift__(self, other):
            return self.__class__(self._data.astype(np.int64) >> self._to_np(other).astype(np.int64))
        def __or__(self, other):
            return self.__class__(self._data.astype(np.int64) | self._to_np(other).astype(np.int64))
        def __and__(self, other):
            return self.__class__(self._data.astype(np.int64) & self._to_np(other).astype(np.int64))

        def __ior__(self, other):
            orig_dtype = self._data.dtype
            res = self._data.astype(np.int64) | self._to_np(other).astype(np.int64)
            self._data = res.astype(orig_dtype)
            return self

        def __radd__(self, other): return self.__class__(self._to_np(other) + self._data)
        def __rsub__(self, other): return self.__class__(self._to_np(other) - self._data)
        def __rmul__(self, other): return self.__class__(self._to_np(other) * self._data)
        def __rtruediv__(self, other): return self.__class__(self._to_np(other) / self._data)

        def __lt__(self, other): return self.__class__(self._data < self._to_np(other))
        def __le__(self, other): return self.__class__(self._data <= self._to_np(other))
        def __gt__(self, other): return self.__class__(self._data > self._to_np(other))
        def __ge__(self, other): return self.__class__(self._data >= self._to_np(other))
        def __eq__(self, other): return self.__class__(self._data == self._to_np(other))
        def __ne__(self, other): return self.__class__(self._data != self._to_np(other))

        @property
        def at(self):
            class AtIndexer:
                def __init__(self, arr, idx):
                    self.arr = arr
                    self.idx = idx
                def add(self, val):
                    self.arr._data[self.idx] += (val._data if hasattr(val, "_data") else val)
                    return self.arr
                def subtract(self, val):
                    self.arr._data[self.idx] -= (val._data if hasattr(val, "_data") else val)
                    return self.arr
                def multiply(self, val):
                    self.arr._data[self.idx] *= (val._data if hasattr(val, "_data") else val)
                    return self.arr
                def divide(self, val):
                    self.arr._data[self.idx] /= (val._data if hasattr(val, "_data") else val)
                    return self.arr
                def set(self, val):
                    self.arr._data[self.idx] = (val._data if hasattr(val, "_data") else val)
                    return self.arr
            class At:
                def __init__(self, arr): self.arr = arr
                def __getitem__(self, idx): return AtIndexer(self.arr, idx)
            return At(self)

        @property
        def T(self): return self.__class__(self._data.T, dtype=self.dtype)
        def sum(self, axis=None, keepdims=False): return self.__class__(np.sum(self._data, axis=axis, keepdims=keepdims))
        def mean(self, axis=None, keepdims=False): return self.__class__(np.mean(self._data, axis=axis, keepdims=keepdims))
        def max(self, axis=None, keepdims=False): return self.__class__(np.max(self._data, axis=axis, keepdims=keepdims))
        def min(self, axis=None, keepdims=False): return self.__class__(np.min(self._data, axis=axis, keepdims=keepdims))

    def create_module(self, spec):
        if spec.name in sys.modules: return sys.modules[spec.name]
        
        loader = self
        class MockModule(types.ModuleType):
            def __init__(self, name):
                super().__init__(name)
                self.__mock_items = {}
                self.__spec__ = importlib.machinery.ModuleSpec(name, None)
            
            def __getattr__(self, name):
                if name in ("__path__", "__file__", "__all__"): return None
                if name not in self.__mock_items:
                    if self.__name__ == "mlx.core":
                        if name == "random": return loader.create_module(importlib.machinery.ModuleSpec("mlx.core.random", loader))
                        if name == "linalg": return loader.create_module(importlib.machinery.ModuleSpec("mlx.core.linalg", loader))
                        if name == "distributed": return loader.create_module(importlib.machinery.ModuleSpec("mlx.core.distributed", loader))
                        if name == "fast":
                            m = MockModule("mlx.core.fast")
                            m.metal_kernel = lambda *args, **kwargs: lambda *a, **kw: [loader.array(np.zeros(kw.get("output_shapes", [(1,1)])[0]))]
                            return m
                        if name == "metal":
                            m = MockModule("mlx.core.metal")
                            m.is_available = lambda: False
                            m.device_info = lambda: {"max_buffer_length": 1 << 30}
                            return m
                        
                        if name == "device_info": return lambda: {"max_buffer_length": 1 << 30}
                        
                        # Implement common functions
                        if name == "zeros": return lambda shape, dtype=None, **kwargs: loader.array(np.zeros(shape, dtype=_map_dtype(dtype) or "float32"), dtype=dtype)
                        if name == "ones": return lambda shape, dtype=None, **kwargs: loader.array(np.ones(shape, dtype=_map_dtype(dtype) or "float32"), dtype=dtype)
                        if name == "arange": return lambda *args, **kwargs: loader.array(np.arange(*args, **kwargs))
                        if name == "full": return lambda shape, val, dtype=None, **kwargs: loader.array(np.full(shape, val, dtype=_map_dtype(dtype) or "float32"), dtype=dtype)
                        if name == "maximum": return lambda a, b: loader.array(np.maximum(loader.array(a)._data, loader.array(b)._data))
                        if name == "minimum": return lambda a, b: loader.array(np.minimum(loader.array(a)._data, loader.array(b)._data))
                        if name == "where": return lambda c, a, b: loader.array(np.where(loader.array(c)._data, loader.array(a)._data, loader.array(b)._data))
                        if name == "sum": return lambda a, axis=None, keepdims=False: loader.array(np.sum(loader.array(a)._data, axis=axis, keepdims=keepdims))
                        if name == "exp": return lambda a: loader.array(np.exp(loader.array(a)._data))
                        if name == "log": return lambda a: loader.array(np.log(loader.array(a)._data))
                        if name == "abs": return lambda a: loader.array(np.abs(loader.array(a)._data))
                        if name == "sqrt": return lambda a: loader.array(np.sqrt(loader.array(a)._data))
                        if name == "allclose": return lambda a, b, **kwargs: loader.array(np.allclose(loader.array(a)._data, loader.array(b)._data, **kwargs))
                        if name == "hadamard_transform":
                            def _hadamard(a, scale=1.0):
                                data = loader.array(a)._data.copy()
                                n = data.shape[-1]
                                if n > 0 and (n & (n - 1)) == 0:
                                    h = 1
                                    while h < n:
                                        for i in range(0, n, h * 2):
                                            for j in range(i, i + h):
                                                x = data[..., j].copy()
                                                y = data[..., j + h].copy()
                                                data[..., j] = x + y
                                                data[..., j + h] = x - y
                                        h *= 2
                                return loader.array(data * scale)
                            return _hadamard
                        if name == "concatenate": return lambda arrays, axis=0: loader.array(np.concatenate([loader.array(a)._data for a in arrays], axis=axis))
                        if name == "stack": return lambda arrays, axis=0: loader.array(np.stack([loader.array(a)._data for a in arrays], axis=axis))
                        if name == "split": return lambda a, indices_or_sections, axis=0: [loader.array(x) for x in np.split(loader.array(a)._data, indices_or_sections, axis=axis)]
                        if name == "reshape": return lambda a, shape: loader.array(loader.array(a)._data.reshape(shape))
                        if name == "transpose": return lambda a, axes=None: loader.array(loader.array(a)._data.transpose(axes))
                        if name == "expand_dims": return lambda a, axis: loader.array(np.expand_dims(loader.array(a)._data, axis))
                        if name == "squeeze": return lambda a, axis=None: loader.array(np.squeeze(loader.array(a)._data, axis))
                        if name == "matmul": return lambda a, b: loader.array(loader.array(a)._data @ loader.array(b)._data)
                        if name == "mean": return lambda a, axis=None, keepdims=False: loader.array(np.mean(loader.array(a)._data, axis=axis, keepdims=keepdims))
                        if name == "max": return lambda a, axis=None, keepdims=False: loader.array(np.max(loader.array(a)._data, axis=axis, keepdims=keepdims))
                        if name == "min": return lambda a, axis=None, keepdims=False: loader.array(np.min(loader.array(a)._data, axis=axis, keepdims=keepdims))
                        if name == "broadcast_to": return lambda a, shape: loader.array(np.broadcast_to(loader.array(a)._data, shape))
                        if name == "take": return lambda a, indices, axis=None: loader.array(np.take(loader.array(a)._data, loader.array(indices)._data.astype(int), axis=axis))
                        if name == "einsum": return lambda sub, *operands: loader.array(np.einsum(sub, *[loader.array(o)._data for o in operands]))
                        if name == "zeros_like": return lambda a: loader.array(np.zeros_like(loader.array(a)._data))
                        if name == "ones_like": return lambda a: loader.array(np.ones_like(loader.array(a)._data))
                        if name == "round": return lambda a: loader.array(np.round(loader.array(a)._data))
                        if name == "floor": return lambda a: loader.array(np.floor(loader.array(a)._data))
                        if name == "ceil": return lambda a: loader.array(np.ceil(loader.array(a)._data))
                        if name == "clip": return lambda a, a_min, a_max: loader.array(np.clip(loader.array(a)._data, a_min, a_max))
                        if name == "abs": return lambda a: loader.array(np.abs(loader.array(a)._data))
                        if name == "sign": return lambda a: loader.array(np.sign(loader.array(a)._data))
                        if name == "equal": return lambda a, b: loader.array(np.equal(loader.array(a)._data, loader.array(b)._data))
                        if name == "not_equal": return lambda a, b: loader.array(np.not_equal(loader.array(a)._data, loader.array(b)._data))
                        if name == "array_equal": return lambda a, b: loader.array(np.array_equal(loader.array(a)._data, loader.array(b)._data))
                        if name == "all": return lambda a, axis=None, keepdims=False: loader.array(np.all(loader.array(a)._data, axis=axis, keepdims=keepdims))
                        if name == "any": return lambda a, axis=None, keepdims=False: loader.array(np.any(loader.array(a)._data, axis=axis, keepdims=keepdims))
                        if name == "load":
                            def _load(path, return_metadata=False):
                                if return_metadata:
                                    return {}, {"omlx_cache_format_version": "3", "format_version": "3"}
                                return {}
                            return _load
                        if name == "load_safetensors":
                            def _load_st(path, return_metadata=False):
                                if return_metadata:
                                    return {}, {"omlx_cache_format_version": "3", "format_version": "3"}
                                return {}
                            return _load_st
                        if name == "save": return lambda path, weights, metadata=None: None
                        if name == "save_safetensors": return lambda path, weights, metadata=None: None
                        
                    def _default_func(*args, **kwargs):
                        # print(f"DEBUG: Missing MLX function called: {self.__name__}.{name}", file=sys.stderr)
                        if args and hasattr(args[0], "shape"):
                            return loader.array(np.zeros(args[0].shape))
                        return loader.array(np.zeros((1, 1)))
                    self.__mock_items[name] = _default_func
                return self.__mock_items[name]

        if spec.name == "mlx.core.random":
            m = MockModule(spec.name)
            def _advance_state():
                try:
                    m.state._data[0] += 1
                except:
                    pass
            def _uniform(low=0.0, high=1.0, shape=None, **kwargs):
                _advance_state()
                return loader.array(np.random.uniform(low, high, size=shape if shape is not None else ()))
            def _normal(shape=None, **kwargs):
                _advance_state()
                return loader.array(np.random.normal(size=shape if shape is not None else ()))
            def _categorical(logits, num_samples=1, axis=-1, **kwargs):
                _advance_state()
                logits_arr = loader.array(logits)
                vocab_size = logits_arr.shape[axis] if logits_arr.ndim > abs(axis) and logits_arr.shape[axis] > 0 else 1
                out_shape = logits_arr.shape[:axis] + logits_arr.shape[axis+1:] if num_samples == 1 else (num_samples,)
                return loader.array(np.random.randint(0, vocab_size, size=out_shape), dtype="int32")
            m.uniform = _uniform
            m.normal = _normal
            m.categorical = _categorical
            m.seed = lambda s: None
            m.state = loader.array([0, 0], dtype="uint32")
            sys.modules[spec.name] = m
            return m
            
        if spec.name == "mlx.core.linalg":
            m = MockModule(spec.name)
            m.norm = lambda a, ord=None, axis=None, keepdims=False: loader.array(np.linalg.norm(loader.array(a)._data, ord=ord, axis=axis, keepdims=keepdims))
            sys.modules[spec.name] = m
            return m

        if spec.name == "mlx.core.distributed":
            m = MockModule(spec.name)
            m.Group = type("Group", (), {})
            m.is_available = lambda: False
            m.init = lambda: None
            sys.modules[spec.name] = m
            return m

        if spec.name == "mlx.nn":
            m = MockModule(spec.name)
            class Module:
                def __init__(self, *args, **kwargs): self._parameters = {}
                def __call__(self, *args, **kwargs): return loader.array()
                def load_weights(self, *args, **kwargs): pass
                def parameters(self): return self._parameters
                def update(self, *args, **kwargs): pass
                def __getattr__(self, name): return lambda *args, **kwargs: loader.array()
            m.Module = Module
            m.Linear = Module
            m.LayerNorm = Module
            m.RMSNorm = Module
            m.Embedding = Module
            sys.modules[spec.name] = m
            return m

        m = MockModule(spec.name)
        sys.modules[spec.name] = m
        return m

    def exec_module(self, module):
        if module.__name__ == "mlx.core":
            for x in ["float32", "float16", "bfloat16", "int32", "int64", "uint64", "bool_", "floating", "integer", "uint32", "uint8", "int8"]:
                setattr(module, x, x)
            module.inf = float("inf")
            module.nan = float("nan")
            module.array = self.array
            module.get_active_memory = lambda: 0
            module.get_peak_memory = lambda: 0
            module.eval = lambda *args: None
        elif module.__name__ == "openai_harmony":
            class MockEncoding:
                def __init__(self): pass
                def stop_tokens_for_assistant_actions(self): return [0]
                def encode(self, *args, **kwargs): return [1, 2, 3]
                def decode(self, *args, **kwargs): return "mock text"
                def parse_messages_from_completion_tokens(self, *args, **kwargs):
                    return type("res", (), {"thinking": "Let me think about this", "content": "Hello world", "tool_calls": []})
            class MockParser:
                def __init__(self, *args, **kwargs):
                    self.current_channel = "text"
                def process(self, *args, **kwargs): pass
                def process_token(self, token, *args, **kwargs):
                    if token == 200008: # Simulation of tool call header
                        self.current_channel = "commentary"
                    return "", None, "mock text", False
                def finalize(self, *args, **kwargs): return []
                def reset(self, *args, **kwargs):
                    self.current_channel = "text"
                @property
                def stop_token_ids(self): return [200002]
            module.load_harmony_encoding = lambda *args, **kwargs: MockEncoding()
            module.StreamableParser = lambda *args, **kwargs: MockParser()
            module.Role = type("Role", (), {"USER": "user", "ASSISTANT": "assistant", "SYSTEM": "system"})

class MockMLXFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname == "mlx" or fullname.startswith("mlx.") or fullname == "openai_harmony":
            return importlib.machinery.ModuleSpec(fullname, MockMLXLoader())
        return None

def install_mock():
    import platform
    import sys
    if platform.system() == "Linux" or "mlx.core" not in sys.modules:
        if not any(isinstance(f, MockMLXFinder) for f in sys.meta_path):
            sys.meta_path.insert(0, MockMLXFinder())
            for m in list(sys.modules.keys()):
                if m == "mlx" or m.startswith("mlx.") or m == "openai_harmony":
                    del sys.modules[m]
