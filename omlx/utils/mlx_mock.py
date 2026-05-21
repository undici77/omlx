# SPDX-License-Identifier: Apache-2.0
import sys
import types
import importlib.abc
import importlib.machinery
import numpy as np

def _map_dtype(dtype):
    if dtype == "bfloat16": return "float16"
    if isinstance(dtype, str):
        if dtype.startswith("uint"):
            try:
                bits = int(dtype[4:])
                if bits <= 8: return "uint8"
                if bits <= 16: return "uint16"
                if bits <= 32: return "uint32"
                return "uint64"
            except ValueError: pass
        return dtype
    if hasattr(dtype, "__name__"): return dtype.__name__
    return None

class MockMLXLoader(importlib.abc.Loader):
    def _save_safetensors_impl(self, path, weights, metadata=None):
        import json, struct
        try:
            header = {}
            if metadata: header["__metadata__"] = metadata
            mlx_to_st = {
                "float32": "F32", "float16": "F16", "bfloat16": "BF16",
                "int32": "I32", "int64": "I64", "uint8": "U8", "uint32": "U32",
                "bool_": "BOOL", "uint4": "U8", "uint2": "U8", "uint5": "U8", "uint6": "U8"
            }
            offset = 0
            tensors_data = []
            for name, arr in weights.items():
                arr_mock = self.array(arr)
                st_dtype = mlx_to_st.get(str(arr_mock.dtype), "F32")
                data = arr_mock._data.tobytes()
                length = len(data)
                header[name] = {"dtype": st_dtype, "shape": list(arr_mock.shape), "data_offsets": [offset, offset + length]}
                tensors_data.append(data)
                offset += length
            header_json = json.dumps(header).encode("utf-8")
            header_size = len(header_json)
            with open(path, "wb") as f:
                f.write(struct.pack("<Q", header_size))
                f.write(header_json)
                for d in tensors_data:
                    f.write(d)
        except Exception: pass

    def _load_safetensors_impl(self, path, return_metadata=False):
        import json, struct
        try:
            with open(path, "rb") as f:
                header_size_bytes = f.read(8)
                if len(header_size_bytes) < 8: return ({}, {}) if return_metadata else {}
                header_size = struct.unpack("<Q", header_size_bytes)[0]
                header_json = f.read(header_size).decode("utf-8")
                header = json.loads(header_json)
                metadata = header.pop("__metadata__", {})
                tensors = {}
                st_to_np = {"F16": np.float16, "F32": np.float32, "F64": np.float64, "I8": np.int8, "I16": np.int16, "I32": np.int32, "I64": np.int64, "U8": np.uint8, "U16": np.uint16, "U32": np.uint32, "U64": np.uint64, "BOOL": np.bool_, "BF16": np.uint16}
                data_start = 8 + header_size
                for name, info in header.items():
                    dtype_str = info["dtype"]
                    shape = info["shape"]
                    offsets = info["data_offsets"]
                    f.seek(data_start + offsets[0])
                    raw_data = f.read(offsets[1] - offsets[0])
                    np_dtype = st_to_np.get(dtype_str, np.float32)
                    arr = np.frombuffer(raw_data, dtype=np_dtype).reshape(shape)
                    if dtype_str == "BF16":
                        tensors[name] = self.array(arr.view(np.float16), dtype="bfloat16")
                    else:
                        st_to_mlx = {"F16": "float16", "F32": "float32", "U8": "uint8", "I32": "int32", "I64": "int64", "BOOL": "bool_", "U32": "uint32"}
                        tensors[name] = self.array(arr, dtype=st_to_mlx.get(dtype_str, "float32"))
                return (tensors, metadata) if return_metadata else tensors
        except Exception: return ({}, {}) if return_metadata else {}

    class array:
        def __init__(self, data=None, dtype=None):
            if hasattr(data, "_data"):
                data = data._data
            if isinstance(data, np.ndarray):
                self._data = data.copy()
            elif data is None:
                self._data = np.array([], dtype=_map_dtype(dtype) or "float32")
            else:
                self._data = np.array(data)

            mapped = _map_dtype(dtype)
            if mapped:
                self._data = self._data.astype(mapped)

            # Map float64 to float32 by default
            if self._data.dtype == np.float64:
                self._data = self._data.astype(np.float32)

            self.dtype = dtype or str(self._data.dtype)
            self.shape, self.size, self.ndim = self._data.shape, self._data.size, self._data.ndim
            self.nbytes = self._data.nbytes

        def _to_np(self, other):
            if hasattr(other, "_data"): return other._data
            if isinstance(other, np.ndarray): return other
            return np.array(other)

        def item(self): return self._data.item()
        def tolist(self): return self._data.tolist()
        def astype(self, dtype): return self.__class__(self._data, dtype=dtype)
        def reshape(self, *shape):
            if len(shape) == 1 and isinstance(shape[0], (list, tuple)): shape = shape[0]
            return self.__class__(self._data.reshape(shape), dtype=self.dtype)
        def flatten(self, axis=0):
            if axis == 0: return self.__class__(self._data.flatten(), dtype=self.dtype)
            new_shape = list(self.shape[:axis]) + [-1]
            return self.__class__(self._data.reshape(new_shape), dtype=self.dtype)
        def transpose(self, *axes):
            if len(axes) == 1 and isinstance(axes[0], (list, tuple)): axes = axes[0]
            return self.__class__(self._data.transpose(axes if axes else None), dtype=self.dtype)
        def squeeze(self, axis=None): return self.__class__(np.squeeze(self._data, axis=axis), dtype=self.dtype)
        def abs(self): return self.__class__(np.abs(self._data), dtype=self.dtype)
        def max(self, axis=None, keepdims=False): return self.__class__(np.max(self._data, axis=axis, keepdims=keepdims), dtype=self.dtype)
        def min(self, axis=None, keepdims=False): return self.__class__(np.min(self._data, axis=axis, keepdims=keepdims), dtype=self.dtype)
        def sum(self, axis=None, keepdims=False): return self.__class__(np.sum(self._data, axis=axis, keepdims=keepdims), dtype=self.dtype)
        def mean(self, axis=None, keepdims=False): return self.__class__(np.mean(self._data, axis=axis, keepdims=keepdims), dtype=self.dtype)
        def __repr__(self):
            try:
                dtype_str = self.dtype if isinstance(self.dtype, str) else getattr(self.dtype, "__name__", str(self.dtype))
                return f"mx.array(shape={self.shape}, dtype={dtype_str})"
            except Exception: return "mx.array(error)"
        def __bool__(self): return bool(self.item())
        def __len__(self): return len(self._data) if self.ndim > 0 else 0
        def __getitem__(self, i):
            if isinstance(i, tuple): i = tuple(idx._data if hasattr(idx, "_data") else idx for idx in i)
            elif hasattr(i, "_data"): i = i._data
            return self.__class__(self._data[i])
        def __setitem__(self, i, v):
            if hasattr(v, "_data"): v = v._data
            self._data[i] = v
        def __add__(self, other): return self.__class__(self._data + self._to_np(other))
        def __sub__(self, other): return self.__class__(self._data - self._to_np(other))
        def __mul__(self, other): return self.__class__(self._data * self._to_np(other))
        def __truediv__(self, other): return self.__class__(self._data / self._to_np(other))
        def __lshift__(self, other): return self.__class__(self._data.astype(int) << (other._data.astype(int) if hasattr(other, "_data") else int(other)))
        def __rshift__(self, other): return self.__class__(self._data.astype(int) >> (other._data.astype(int) if hasattr(other, "_data") else int(other)))
        def __and__(self, other): return self.__class__(self._data.astype(int) & (self._to_np(other).astype(int)))
        def __or__(self, other): return self.__class__(self._data.astype(int) | (self._to_np(other).astype(int)))
        def __xor__(self, other): return self.__class__(self._data.astype(int) ^ (self._to_np(other).astype(int)))
        def __ior__(self, other):
            self._data = (self._data.astype(int) | (self._to_np(other).astype(int))).astype(self._data.dtype)
            return self
        def __matmul__(self, other): return self.__class__(self._data @ self._to_np(other))
        def __ne__(self, other): return self.__class__(self._data != self._to_np(other), dtype="bool_")
        def __eq__(self, other): return self.__class__(self._data == self._to_np(other), dtype="bool_")

        def __floordiv__(self, other): return self.__class__(self._data // self._to_np(other))
        def __pow__(self, other): return self.__class__(self._data ** self._to_np(other))
        def __radd__(self, other): return self.__class__(self._to_np(other) + self._data)
        def __rsub__(self, other): return self.__class__(self._to_np(other) - self._data)
        def __rmul__(self, other): return self.__class__(self._to_np(other) * self._data)
        def __rtruediv__(self, other): return self.__class__(self._to_np(other) / self._data)
        def __rfloordiv__(self, other): return self.__class__(self._to_np(other) // self._data)
        def __rpow__(self, other): return self.__class__(self._to_np(other) ** self._data)
        def __neg__(self): return self.__class__(-self._data)
        def __eq__(self, other): return self.__class__(self._data == self._to_np(other), dtype="bool_")
        def __lt__(self, other): return self.__class__(self._data < self._to_np(other), dtype="bool_")
        def __gt__(self, other): return self.__class__(self._data > self._to_np(other), dtype="bool_")
        def __le__(self, other): return self.__class__(self._data <= self._to_np(other), dtype="bool_")
        def __ge__(self, other): return self.__class__(self._data >= self._to_np(other), dtype="bool_")
        def __iter__(self):
            if self.ndim == 0: yield self
            else:
                for x in self._data: yield self.__class__(x, dtype=self.dtype)
        def __array__(self, dtype=None): return self._data.astype(dtype) if dtype else self._data
        def view(self, dtype):
            np_dtype = _map_dtype(dtype) or dtype
            try:
                result = self.__class__(self._data.view(np_dtype))
            except Exception:
                result = self.__class__(self._data.astype(np_dtype))
            # Preserve MLX dtype label (e.g. "bfloat16" instead of mapped "float16")
            mlx_dtype = dtype if isinstance(dtype, str) else getattr(dtype, "__name__", None)
            if mlx_dtype:
                result.dtype = mlx_dtype
            return result
        def __int__(self): return int(self._data.item())
        def __index__(self): return int(self._data.item())
        def __buffer__(self, flags): return self._data.__buffer__(flags)
        @property
        def at(self):
            """MLX scatter-update syntax: arr.at[idx].add(val) / arr.at[idx].set(val)."""
            outer = self
            class _AtIndexer:
                def __init__(self, idx): self._idx = idx
                def add(self, val):
                    d = outer._data.copy()
                    v = val._data if hasattr(val, "_data") else np.array(val)
                    np.add.at(d, self._idx, v)
                    return outer.__class__(d, dtype=outer.dtype)
                def set(self, val):
                    d = outer._data.copy()
                    v = val._data if hasattr(val, "_data") else np.array(val)
                    d[self._idx] = v
                    return outer.__class__(d, dtype=outer.dtype)
            class _At:
                def __getitem__(self, idx): return _AtIndexer(idx)
            return _At()
        @property
        def T(self): return self.__class__(self._data.T, dtype=self.dtype)

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
                            m.metal_kernel = lambda *a, **k: lambda *aa, **kk: [loader.array(np.zeros(kk.get("output_shapes", [(1,1)])[0]))]
                            return m
                        if name == "metal":
                            m = MockModule("mlx.core.metal")
                            m.is_available = lambda: False
                            m.device_info = lambda: {"max_buffer_length": 1 << 30}
                            return m
                        if name == "device_info": return lambda: {"max_buffer_length": 1 << 30}
                        if name == "compile": return lambda f=None, **k: (lambda f: f) if f is None else f
                        if name == "quantize":
                            def _quantize(w, group_size=64, bits=4, mode="affine"):
                                w = loader.array(w)
                                qw = loader.array(w._data, dtype=f"uint{bits}" if bits <= 8 else "uint32")
                                s_shape = list(w.shape)
                                s_shape[-1] = max(1, s_shape[-1] // group_size)
                                scales = loader.array(np.zeros(s_shape), dtype="float16")
                                if mode == "affine":
                                    biases = loader.array(np.zeros(s_shape), dtype="float16")
                                    return (qw, scales, biases)
                                return (qw, scales)
                            return _quantize
                        if name == "dequantize":
                            def _dequantize(qw, scales, biases=None, group_size=64, bits=4, mode="affine"):
                                return loader.array(np.zeros(qw.shape), dtype="float32")
                            return _dequantize
                        if name == "argpartition":
                            def _argpartition(a, kth, axis=-1):
                                return loader.array(np.argpartition(loader.array(a)._data, kth, axis=axis), dtype="int32")
                            return _argpartition
                        if name == "take_along_axis":
                            def _take_along_axis(a, i, axis):
                                arr = loader.array(a)._data
                                if arr.ndim == 0: arr = arr.reshape(1)
                                idx = loader.array(i)._data.astype(int)
                                if idx.ndim == 0: idx = idx.reshape(1)
                                return loader.array(np.take_along_axis(arr, idx, axis=axis))
                            return _take_along_axis
                        if name == "put_along_axis":
                            def _put(a, i, v, axis):
                                d = loader.array(a)._data.copy()
                                if d.ndim == 0: d = d.reshape(1)
                                idx = loader.array(i)._data.astype(int)
                                if idx.ndim == 0: idx = idx.reshape(1)
                                val = loader.array(v)._data
                                if val.ndim == 0: val = val.reshape(1)
                                np.put_along_axis(d, idx, val, axis=axis)
                                return loader.array(d)
                            return _put
                        if name in ("equal", "not_equal", "array_equal", "all", "any"):
                            def _logic(*a, **k):
                                res = getattr(np, name)(*[loader.array(x)._data if hasattr(x, "_data") else x for x in a], **k)
                                return loader.array(res, dtype="bool_")
                            return _logic
                        if name == "load":
                            def _load(path, return_metadata=False):
                                if str(path).endswith(".safetensors"): return loader._load_safetensors_impl(path, return_metadata)
                                return ({}, {"omlx_cache_format_version": "3", "format_version": "3"}) if return_metadata else {}
                            return _load
                        if name == "load_safetensors": return lambda p, rm=False: loader._load_safetensors_impl(p, rm)
                        if name in ("save", "save_safetensors"):
                            def _save(p, w, metadata=None, **k):
                                return loader._save_safetensors_impl(p, w, metadata)
                            return _save
                        if name == "softmax":
                            def _softmax(x, axis=-1):
                                d = loader.array(x)._data
                                if d.size == 0: return loader.array(d)
                                a_max = np.max(d, axis=axis, keepdims=True)
                                e_x = np.exp(d - a_max)
                                return loader.array(e_x / e_x.sum(axis=axis, keepdims=True))
                            return _softmax
                        if name == "logsumexp":
                            def _lse(a, axis=None, keepdims=False):
                                d = loader.array(a)._data
                                a_max = np.max(d, axis=axis, keepdims=True)
                                lse = a_max + np.log(np.sum(np.exp(d - a_max), axis=axis, keepdims=keepdims))
                                if not keepdims: lse = np.squeeze(lse, axis=axis)
                                return loader.array(lse)
                            return _lse
                        if name == "allclose": return lambda a, b, **k: loader.array(np.allclose(loader.array(a)._data, loader.array(b)._data, **k), dtype="bool_")
                        if name in ("concatenate", "stack"): return lambda arrays, axis=0: loader.array(getattr(np, name)([loader.array(a)._data for a in arrays], axis=axis))
                        if name == "split": return lambda a, i, axis=0: [loader.array(x) for x in np.split(loader.array(a)._data, i, axis=axis)]
                        if name in ("zeros", "ones"):
                            return lambda shape, dtype=None, **k: loader.array(getattr(np, name)(shape, dtype=_map_dtype(dtype) or "float32"), dtype=dtype)
                        if name == "full":
                            return lambda shape, fill_value, dtype=None, **k: loader.array(np.full(shape, fill_value, dtype=_map_dtype(dtype) or "float32"), dtype=dtype)
                        if name == "arange":
                            return lambda *a, dtype=None, **k: loader.array(np.arange(*a, dtype=_map_dtype(dtype) or "float32"), dtype=dtype)
                        if name in ("reshape", "transpose", "expand_dims", "squeeze", "broadcast_to", "argmax", "argmin", "zeros_like", "ones_like", "argsort"):
                            return lambda a, *args, **kwargs: loader.array(getattr(np, name)(loader.array(a)._data, *args, **kwargs))
                        if name == "matmul": return lambda a, b: loader.array(loader.array(a)._data @ loader.array(b)._data)
                        if name in ("mean", "max", "min", "sum"): return lambda a, axis=None, keepdims=False: loader.array(getattr(np, name)(loader.array(a)._data, axis=axis, keepdims=keepdims))
                        if name == "take": return lambda a, i, axis=None: loader.array(np.take(loader.array(a)._data, loader.array(i)._data.astype(int), axis=axis))
                        if name == "einsum": return lambda sub, *operands: loader.array(np.einsum(sub, *[loader.array(o)._data for o in operands]))
                        if name == "clip": return lambda a, a_min, a_max: loader.array(np.clip(loader.array(a)._data, a_min, a_max))
                        if name in ("exp", "log", "abs", "sqrt", "round", "floor", "ceil", "sign", "cos", "sin", "tan", "tanh", "sigmoid"): return lambda a: loader.array(getattr(np, name)(loader.array(a)._data))
                        if name == "cumsum":
                            return lambda a, axis=None, **k: loader.array(np.cumsum(loader.array(a)._data, axis=axis))
                        if name == "pad":
                            return lambda a, pad_width, mode="constant", **k: loader.array(np.pad(loader.array(a)._data, pad_width, mode=mode, **{kk: vv for kk, vv in k.items() if kk == "constant_values"}))
                        if name == "copy": return lambda a, **k: loader.array(loader.array(a)._data.copy())
                        if name == "stream":
                            import contextlib
                            @contextlib.contextmanager
                            def _stream_cm(*a, **k): yield
                            return _stream_cm
                        if name in ("maximum", "minimum", "where"): return lambda *a: loader.array(getattr(np, name)(*[loader.array(x)._data if hasattr(x, "_data") else x for x in a]))
                        if name == "contiguous": return lambda a, **k: a
                        if name in ("stop_gradient", "eval"): return lambda *a, **k: a[0] if a else None
                        if name == "power": return lambda a, b: loader.array(np.power(loader.array(a)._data, loader.array(b)._data if hasattr(b, "_data") else b))
                        if name == "from_fp8": return lambda a, dtype=None, **k: loader.array(loader.array(a)._data.astype(_map_dtype(dtype) or "float32"), dtype=dtype)
                        if name == "clear_cache": return lambda *a, **k: None

                    _captured_name = name
                    def _default_func(*args, _n=_captured_name, **kwargs):
                        raise NotImplementedError(f"mlx.{_n}() is not implemented in the MLX mock. Add it to omlx/utils/mlx_mock.py.")
                    self.__mock_items[name] = _default_func
                return self.__mock_items[name]

        if spec.name == "mlx.utils":
            m = MockModule(spec.name)
            def _tree_flatten(tree):
                if isinstance(tree, dict): return list(tree.items())
                if isinstance(tree, (list, tuple)): return [(str(i), v) for i, v in enumerate(tree)]
                return [("", tree)]
            m.tree_flatten = _tree_flatten
            sys.modules[spec.name] = m; return m
        if spec.name == "mlx.core.random":
            m = MockModule(spec.name)
            m.state = loader.array([0, 0], dtype="uint32")
            def _advance():
                try: m.state._data[0] += 1
                except Exception: pass
            m.uniform = lambda l=0, h=1, s=None, **k: (_advance() or loader.array(np.random.uniform(l, h, size=s if s is not None else k.get("shape", ()))))
            m.normal = lambda s=None, **k: (_advance() or loader.array(np.random.normal(size=s if s is not None else k.get("shape", ()))))
            m.categorical = lambda l, n=1, a=-1, **k: (_advance() or loader.array(np.random.randint(0, loader.array(l).shape[a] if loader.array(l).ndim > abs(a) else 1, size=loader.array(l).shape[:a] if n == 1 else (n,)), dtype="int32"))
            m.seed = lambda s: None; sys.modules[spec.name] = m; return m
        if spec.name == "mlx.core.linalg":
            m = MockModule(spec.name)
            m.norm = lambda a, ord=None, axis=None, keepdims=False: loader.array(np.linalg.norm(loader.array(a)._data, ord=ord, axis=axis, keepdims=keepdims))
            sys.modules[spec.name] = m; return m
        if spec.name == "mlx.core.distributed":
            m = MockModule(spec.name)
            m.Group = type("Group", (), {})
            m.is_available = lambda: False
            m.init = lambda: None
            sys.modules[spec.name] = m; return m
        if spec.name == "mlx.nn":
            m = MockModule(spec.name)
            class Module:
                def __init__(self, *args, **kwargs): self._parameters = {}
                def __setattr__(self, name, value):
                    super().__setattr__(name, value)
                    if hasattr(value, "parameters"):
                        for k, v in value.parameters().items(): self._parameters[f"{name}.{k}" if k else name] = v
                    elif isinstance(value, loader.array): self._parameters[name] = value
                def __call__(self, *args, **kwargs):
                    if args and hasattr(args[0], "shape"): return args[0]
                    return loader.array(np.zeros((1, 1)))
                def load_weights(self, *args, **kwargs): pass
                def parameters(self): return self._parameters
                def update(self, *args, **kwargs): pass
                def __getattr__(self, name): return lambda *a, **k: loader.array(np.zeros((1, 1)))
            m.Module = Module
            for name in ("Linear", "LayerNorm", "RMSNorm", "Embedding", "Dropout", "SiLU", "GELU", "ReLU", "Tanh", "Softmax"):
                if name == "Linear": m.Linear = type("Linear", (Module,), {"__init__": lambda s, i, o, **k: super(type(s), s).__init__(**k) or setattr(s, "weight", loader.array(np.zeros((o, i))))})
                elif name == "Embedding": m.Embedding = type("Embedding", (Module,), {"__init__": lambda s, n, d, **k: super(type(s), s).__init__(**k) or setattr(s, "weight", loader.array(np.zeros((n, d))))})
                elif name in ("LayerNorm", "RMSNorm"): setattr(m, name, type(name, (Module,), {"__init__": lambda s, d, **k: super(type(s), s).__init__(**k) or setattr(s, "weight", loader.array(np.zeros((d,))))}))
                else: setattr(m, name, Module)
            sys.modules[spec.name] = m; return m
        m = MockModule(spec.name)
        sys.modules[spec.name] = m
        return m

    def exec_module(self, module):
        if module.__name__ == "mlx.core":
            for x in ["float32", "float16", "bfloat16", "int32", "int64", "uint64", "bool_", "floating", "integer", "uint32", "uint16", "uint8", "int8"]: setattr(module, x, x)
            module.inf, module.nan, module.array = float("inf"), float("nan"), self.array
            module.get_active_memory, module.get_peak_memory, module.eval = (lambda: 0), (lambda: 0), (lambda *a: None)
            module.compile = lambda f=None, **k: (lambda f: f) if f is None else f
            module.synchronize = lambda *a: None
            module.cpu = "cpu"
            module.gpu = "gpu"
            module.metal = type("Metal", (), {"is_available": lambda *a, **k: False, "device_info": lambda *a, **k: {"max_buffer_length": 1 << 30}})()

class MockMLXFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname == "mlx" or fullname.startswith("mlx."):
            return importlib.machinery.ModuleSpec(fullname, MockMLXLoader())
        return None

def install_mock():
    import platform, sys
    if platform.system() != "Darwin":
        if not any(isinstance(f, MockMLXFinder) for f in sys.meta_path):
            sys.meta_path.insert(0, MockMLXFinder())
            for m in list(sys.modules.keys()):
                if m == "mlx" or m.startswith("mlx."): del sys.modules[m]
