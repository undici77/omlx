# SPDX-License-Identifier: Apache-2.0
import sys
import types
import importlib.abc
import importlib.machinery
import numpy as np

def _map_dtype(dtype):
    if dtype == "bfloat16": return "float16"
    if isinstance(dtype, str): return dtype
    if hasattr(dtype, "__name__"): return dtype.__name__
    return None

class MockMLXLoader(importlib.abc.Loader):
    def _save_safetensors_impl(self, path, weights, metadata=None):
        import json, struct
        try:
            header = {}
            if metadata: header["__metadata__"] = metadata
            mlx_to_st = {"float32": "F32", "float16": "F16", "bfloat16": "BF16", "int32": "I32", "int64": "I64", "uint8": "U8", "bool_": "BOOL"}
            offset = 0
            tensors_data = []
            for name, arr in weights.items():
                arr_mock = self.array(arr); st_dtype = mlx_to_st.get(str(arr_mock.dtype), "F32"); data = arr_mock._data.tobytes(); length = len(data)
                header[name] = {"dtype": st_dtype, "shape": list(arr_mock.shape), "data_offsets": [offset, offset + length]}
                tensors_data.append(data); offset += length
            header_json = json.dumps(header).encode("utf-8"); header_size = len(header_json)
            with open(path, "wb") as f:
                f.write(struct.pack("<Q", header_size)); f.write(header_json)
                for d in tensors_data: f.write(d)
        except Exception: pass

    def _load_safetensors_impl(self, path, return_metadata=False):
        import json, struct
        try:
            with open(path, "rb") as f:
                header_size_bytes = f.read(8)
                if len(header_size_bytes) < 8: return ({}, {}) if return_metadata else {}
                header_size = struct.unpack("<Q", header_size_bytes)[0]; header_json = f.read(header_size).decode("utf-8"); header = json.loads(header_json)
                metadata = header.pop("__metadata__", {}); tensors = {}
                st_to_np = {"F16": np.float16, "F32": np.float32, "F64": np.float64, "I8": np.int8, "I16": np.int16, "I32": np.int32, "I64": np.int64, "U8": np.uint8, "U16": np.uint16, "U32": np.uint32, "U64": np.uint64, "BOOL": np.bool_, "BF16": np.uint16}
                data_start = 8 + header_size
                for name, info in header.items():
                    dtype_str = info["dtype"]; shape = info["shape"]; offsets = info["data_offsets"]; f.seek(data_start + offsets[0]); raw_data = f.read(offsets[1] - offsets[0])
                    np_dtype = st_to_np.get(dtype_str, np.float32); arr = np.frombuffer(raw_data, dtype=np_dtype).reshape(shape)
                    if dtype_str == "BF16": tensors[name] = self.array(arr.view(np.float16), dtype="bfloat16")
                    else: tensors[name] = self.array(arr, dtype=dtype_str.lower().replace("f16", "float16").replace("f32", "float32").replace("bool", "bool_"))
                return (tensors, metadata) if return_metadata else tensors
        except Exception: return ({}, {}) if return_metadata else {}

    class array:
        def __init__(self, data=None, dtype=None):
            if data is not None:
                if hasattr(data, "_data"):
                    self._data = data._data.copy()
                    if dtype: self._data = self._data.astype(_map_dtype(dtype))
                    self.dtype = dtype or getattr(data, "dtype", "float32")
                elif isinstance(data, np.ndarray):
                    self._data = data.copy()
                    if dtype: self._data = self._data.astype(_map_dtype(dtype))
                    elif str(self._data.dtype) == "float64": self._data = self._data.astype("float32")
                    self.dtype = dtype or str(self._data.dtype)
                else:
                    mapped = _map_dtype(dtype)
                    if mapped is None and dtype is None:
                        arr = np.array(data)
                        if str(arr.dtype) == "float64": mapped = "float32"
                        self._data = np.array(data, dtype=mapped if mapped else None)
                    else: self._data = np.array(data, dtype=mapped)
                    self.dtype = dtype or str(self._data.dtype)
            else: self._data = np.zeros((0,), dtype=_map_dtype(dtype) or "float32"); self.dtype = dtype or "float32"
            self.shape, self.size, self.ndim = self._data.shape, self._data.size, self._data.ndim
            self.typecode = 'f' if "float" in str(self.dtype) else 'i'

        @property
        def nbytes(self): return self._data.nbytes
        @property
        def __array_interface__(self): return self._data.__array_interface__
        def __buffer__(self, flags): return self._data.__buffer__(flags)
        def view(self, dtype): return self.__class__(self._data.view(_map_dtype(dtype)), dtype=dtype)
        def reshape(self, *args):
            new_shape = args[0] if len(args) == 1 and isinstance(args[0], (tuple, list)) else args
            try: return self.__class__(self._data.reshape(new_shape), dtype=self.dtype)
            except Exception: return self
        def transpose(self, *axes):
            new_axes = axes[0] if len(axes) == 1 and isinstance(axes[0], (list, tuple)) else axes
            try: return self.__class__(self._data.transpose(new_axes) if new_axes else self._data.T, dtype=self.dtype)
            except Exception: return self
        def swapaxes(self, axis1, axis2):
            try: return self.__class__(np.swapaxes(self._data, axis1, axis2), dtype=self.dtype)
            except Exception: return self
        def squeeze(self, axis=None): return self.__class__(self._data.squeeze(axis=axis), dtype=self.dtype)
        def astype(self, dtype): return self.__class__(self._data.astype(_map_dtype(dtype)), dtype=str(dtype))
        def item(self): return self._data.item() if self.size == 1 else self._data.flat[0].item()
        def tolist(self): return self._data.tolist()
        def flatten(self, start_axis=0, end_axis=-1):
            shape = self.shape
            if not shape: return self
            axes_count = len(shape); start_axis = max(0, min(start_axis + (axes_count if start_axis < 0 else 0), axes_count - 1)); end_axis = max(start_axis, min(end_axis + (axes_count if end_axis < 0 else 0), axes_count - 1))
            new_shape = list(shape[:start_axis]); mid_shape = 1
            for i in range(start_axis, end_axis + 1): mid_shape *= shape[i]
            new_shape.append(mid_shape); new_shape.extend(shape[end_axis + 1:])
            if np.prod(new_shape) != self.size: return self.__class__(self._data.flatten(), dtype=self.dtype)
            return self.__class__(self._data.reshape(new_shape), dtype=self.dtype)
        def __getitem__(self, idx):
            if isinstance(idx, tuple): idx = tuple(i._data if hasattr(i, "_data") else i for i in idx)
            elif hasattr(idx, "_data"): idx = idx._data
            res = self._data[idx]
            if isinstance(res, (np.ndarray, np.generic)): return self.__class__(res, dtype=self.dtype)
            return self.__class__(np.array(res), dtype=self.dtype)
        def __setitem__(self, idx, value): self._data[idx] = value._data if hasattr(value, "_data") else value
        def __len__(self): return len(self._data) if self.ndim > 0 else 0
        def __iter__(self):
            if self.ndim == 0: yield self
            else:
                for x in self._data: yield self.__class__(x, dtype=self.dtype)
        def __array__(self, dtype=None): return self._data.astype(dtype) if dtype else self._data
        def __repr__(self):
            try:
                dt = self.dtype; dtype_str = dt if isinstance(dt, str) else getattr(dt, "__name__", str(dt))
                return f"mx.array(shape={self.shape}, dtype={dtype_str})"
            except Exception: return "mx.array(error in repr)"
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
        def __matmul__(self, other):
            try: return self.__class__(self._data @ self._to_np(other))
            except Exception: return self.__class__(np.zeros((1, 1)))
        def __neg__(self): return self.__class__(-self._data)
        def __lshift__(self, other): return self.__class__(self._data.astype(np.int64) << self._to_np(other).astype(np.int64))
        def __rshift__(self, other): return self.__class__(self._data.astype(np.int64) >> self._to_np(other).astype(np.int64))
        def __or__(self, other): return self.__class__(self._data.astype(np.int64) | self._to_np(other).astype(np.int64))
        def __and__(self, other): return self.__class__(self._data.astype(np.int64) & self._to_np(other).astype(np.int64))
        def __ior__(self, other): self._data |= self._to_np(other).astype(self._data.dtype); return self
        def __radd__(self, other): return self.__class__(self._to_np(other) + self._data)
        def __rsub__(self, other): return self.__class__(self._to_np(other) - self._data)
        def __rmul__(self, other): return self.__class__(self._to_np(other) * self._data)
        def __rtruediv__(self, other): return self.__class__(self._to_np(other) / self._data)
        def __lt__(self, other): return self.__class__(self._data < self._to_np(other), dtype="bool_")
        def __le__(self, other): return self.__class__(self._data <= self._to_np(other), dtype="bool_")
        def __gt__(self, other): return self.__class__(self._data > self._to_np(other), dtype="bool_")
        def __ge__(self, other): return self.__class__(self._data >= self._to_np(other), dtype="bool_")
        def __eq__(self, other): return self.__class__(self._data == self._to_np(other), dtype="bool_")
        def __ne__(self, other): return self.__class__(self._data != self._to_np(other), dtype="bool_")
        @property
        def at(self):
            class AtIndexer:
                def __init__(self, arr, idx): self.arr, self.idx = arr, idx
                def add(self, v): self.arr._data[self.idx] += (v._data if hasattr(v, "_data") else v); return self.arr
                def subtract(self, v): self.arr._data[self.idx] -= (v._data if hasattr(v, "_data") else v); return self.arr
                def multiply(self, v): self.arr._data[self.idx] *= (v._data if hasattr(v, "_data") else v); return self.arr
                def divide(self, v): self.arr._data[self.idx] /= (v._data if hasattr(v, "_data") else v); return self.arr
                def set(self, v): self.arr._data[self.idx] = (v._data if hasattr(v, "_data") else v); return self.arr
            return type("At", (), {"__getitem__": lambda s, i: AtIndexer(self, i)})()
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
                super().__init__(name); self.__mock_items = {}; self.__spec__ = importlib.machinery.ModuleSpec(name, None)
            def __getattr__(self, name):
                if name in ("__path__", "__file__", "__all__"): return None
                if name not in self.__mock_items:
                    if self.__name__ == "mlx.core":
                        if name == "random": return loader.create_module(importlib.machinery.ModuleSpec("mlx.core.random", loader))
                        if name == "linalg": return loader.create_module(importlib.machinery.ModuleSpec("mlx.core.linalg", loader))
                        if name == "distributed": return loader.create_module(importlib.machinery.ModuleSpec("mlx.core.distributed", loader))
                        if name == "fast":
                            m = MockModule("mlx.core.fast"); m.metal_kernel = lambda *a, **k: lambda *aa, **kk: [loader.array(np.zeros(kk.get("output_shapes", [(1,1)])[0]))]
                            return m
                        if name == "metal":
                            m = MockModule("mlx.core.metal"); m.is_available = lambda: False; m.device_info = lambda: {"max_buffer_length": 1 << 30}
                            return m
                        if name == "device_info": return lambda: {"max_buffer_length": 1 << 30}
                        if name == "compile": return lambda f=None, **k: (lambda f: f) if f is None else f
                        if name == "power": return lambda a, b: loader.array(np.power(loader.array(a)._data, loader.array(b)._data))
                        if name == "from_fp8": return lambda a, **k: loader.array(a).astype("bfloat16")
                        if name == "frombuffer": return lambda b, dtype=None: loader.array(np.frombuffer(b, dtype=_map_dtype(dtype) or "float32"), dtype=dtype)
                        if name == "synchronize": return lambda *a: None
                        if name == "stream":
                            class MockStream:
                                def __enter__(self): return self
                                def __exit__(self, *a): pass
                            return lambda *a: MockStream()
                        if name in ("zeros", "ones", "full"): return lambda s, v=0, dtype=None, **k: loader.array(getattr(np, name)(s, v, dtype=_map_dtype(dtype) or "float32") if name=="full" else getattr(np, name)(s, dtype=_map_dtype(dtype) or "float32"), dtype=dtype)
                        if name == "arange": return lambda *a, **k: loader.array(np.arange(*a, **k))
                        if name in ("cumsum", "argsort"): return lambda a, axis=None, **k: loader.array(getattr(np, name)(loader.array(a)._data, axis=axis))
                        if name == "take_along_axis": return lambda a, i, axis: loader.array(np.take_along_axis(loader.array(a)._data, loader.array(i)._data.astype(int), axis=axis))
                        if name == "put_along_axis":
                            def _put(a, i, v, axis):
                                d = loader.array(a)._data.copy(); np.put_along_axis(d, loader.array(i)._data.astype(int), loader.array(v)._data, axis=axis); return loader.array(d)
                            return _put
                        if name == "argpartition": return lambda a, k, axis=-1: loader.array(np.argpartition(loader.array(a)._data, k, axis=axis))
                        if name == "softmax":
                            def _softmax(x, axis=-1):
                                d = loader.array(x)._data; 
                                if d.size == 0: return loader.array(d)
                                e_x = np.exp(d - np.max(d, axis=axis, keepdims=True)); return loader.array(e_x / e_x.sum(axis=axis, keepdims=True))
                            return _softmax
                        if name in ("maximum", "minimum", "where"): return lambda *a: loader.array(getattr(np, name)(*[loader.array(x)._data if hasattr(x, "_data") else x for x in a]))
                        if name == "sum": return lambda a, axis=None, keepdims=False: loader.array(np.sum(loader.array(a)._data, axis=axis, keepdims=keepdims))
                        if name in ("exp", "log", "abs", "sqrt", "round", "floor", "ceil", "sign"): return lambda a: loader.array(getattr(np, name)(loader.array(a)._data))
                        if name == "logsumexp":
                            def _lse(a, axis=None, keepdims=False):
                                d = loader.array(a)._data; a_max = np.max(d, axis=axis, keepdims=True); lse = a_max + np.log(np.sum(np.exp(d - a_max), axis=axis, keepdims=keepdims))
                                if not keepdims: lse = np.squeeze(lse, axis=axis)
                                return loader.array(lse)
                            return _lse
                        if name == "allclose": return lambda a, b, **k: loader.array(np.allclose(loader.array(a)._data, loader.array(b)._data, **k), dtype="bool_")
                        if name == "hadamard_transform": return lambda a, scale=1.0: loader.array(loader.array(a)._data * scale)
                        if name in ("concatenate", "stack"): return lambda arrays, axis=0: loader.array(getattr(np, name)([loader.array(a)._data for a in arrays], axis=axis))
                        if name == "split": return lambda a, i, axis=0: [loader.array(x) for x in np.split(loader.array(a)._data, i, axis=axis)]
                        if name in ("reshape", "transpose", "expand_dims", "squeeze", "broadcast_to", "argmax", "argmin", "zeros_like", "ones_like"):
                            return lambda a, *args, **kwargs: loader.array(getattr(np, name)(loader.array(a)._data, *args, **kwargs))
                        if name == "matmul": return lambda a, b: loader.array(loader.array(a)._data @ loader.array(b)._data)
                        if name in ("mean", "max", "min"): return lambda a, axis=None, keepdims=False: loader.array(getattr(np, name)(loader.array(a)._data, axis=axis, keepdims=keepdims))
                        if name == "take": return lambda a, i, axis=None: loader.array(np.take(loader.array(a)._data, loader.array(i)._data.astype(int), axis=axis))
                        if name == "einsum": return lambda sub, *operands: loader.array(np.einsum(sub, *[loader.array(o)._data for o in operands]))
                        if name == "clip": return lambda a, a_min, a_max: loader.array(np.clip(loader.array(a)._data, a_min, a_max))
                        if name in ("equal", "not_equal", "array_equal", "all", "any"):
                            return lambda *a, **k: loader.array(getattr(np, name)(*[loader.array(x)._data if hasattr(x, "_data") else x for x in a], **k), dtype="bool_")
                        if name == "load":
                            def _load(path, return_metadata=False):
                                if str(path).endswith(".safetensors"): return loader._load_safetensors_impl(path, return_metadata)
                                return ({}, {"omlx_cache_format_version": "3", "format_version": "3"}) if return_metadata else {}
                            return _load
                        if name == "load_safetensors": return lambda p, rm=False: loader._load_safetensors_impl(p, rm)
                        if name in ("save", "save_safetensors"): return lambda p, w, m=None: loader._save_safetensors_impl(p, w, m)
                    def _default_func(*args, **kwargs):
                        if args and hasattr(args[0], "shape"): return loader.array(np.zeros(args[0].shape))
                        return loader.array(np.zeros((1, 1)))
                    self.__mock_items[name] = _default_func
                return self.__mock_items[name]

        if spec.name == "mlx.utils":
            m = MockModule(spec.name)
            def _tree_flatten(tree):
                if isinstance(tree, dict): return list(tree.items())
                if isinstance(tree, (list, tuple)): return [(str(i), v) for i, v in enumerate(tree)]
                if hasattr(tree, "items"): return list(tree.items())
                return [("", tree)]
            m.tree_flatten = _tree_flatten
            sys.modules[spec.name] = m; return m
        if spec.name == "mlx.core.random":
            m = MockModule(spec.name); m.state = loader.array([0, 0], dtype="uint32")
            def _advance():
                try: m.state._data[0] += 1
                except Exception: pass
            m.uniform = lambda l=0, h=1, s=None, **k: (_advance() or loader.array(np.random.uniform(l, h, size=s if s is not None else ())))
            m.normal = lambda s=None, **k: (_advance() or loader.array(np.random.normal(size=s if s is not None else ())))
            m.categorical = lambda l, n=1, a=-1, **k: (_advance() or loader.array(np.random.randint(0, loader.array(l).shape[a] if loader.array(l).ndim > abs(a) else 1, size=loader.array(l).shape[:a] + loader.array(l).shape[a+1:] if n == 1 else (n,)), dtype="int32"))
            m.seed = lambda s: None; sys.modules[spec.name] = m; return m
        if spec.name == "mlx.core.linalg":
            m = MockModule(spec.name); m.norm = lambda a, ord=None, axis=None, keepdims=False: loader.array(np.linalg.norm(loader.array(a)._data, ord=ord, axis=axis, keepdims=keepdims)); sys.modules[spec.name] = m; return m
        if spec.name == "mlx.core.distributed":
            m = MockModule(spec.name); m.Group = type("Group", (), {}); m.is_available = lambda: False; m.init = lambda: None; sys.modules[spec.name] = m; return m
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
                    return loader.array()
                def load_weights(self, *args, **kwargs): pass
                def parameters(self): return self._parameters
                def update(self, *args, **kwargs): pass
                def __getattr__(self, name): return lambda *a, **k: loader.array()
            m.Module = Module
            for name in ("Linear", "LayerNorm", "RMSNorm", "Embedding", "Dropout", "SiLU", "GELU", "ReLU", "Tanh", "Softmax"):
                if name == "Linear": m.Linear = type("Linear", (Module,), {"__init__": lambda s, i, o, **k: super(type(s), s).__init__(**k) or setattr(s, "weight", loader.array(np.zeros((o, i))))})
                elif name == "Embedding": m.Embedding = type("Embedding", (Module,), {"__init__": lambda s, n, d, **k: super(type(s), s).__init__(**k) or setattr(s, "weight", loader.array(np.zeros((n, d))))})
                elif name in ("LayerNorm", "RMSNorm"): setattr(m, name, type(name, (Module,), {"__init__": lambda s, d, **k: super(type(s), s).__init__(**k) or setattr(s, "weight", loader.array(np.zeros((d,))))}))
                else: setattr(m, name, Module)
            sys.modules[spec.name] = m; return m
        m = MockModule(spec.name); sys.modules[spec.name] = m; return m

    def exec_module(self, module):
        if module.__name__ == "mlx.core":
            for x in ["float32", "float16", "bfloat16", "int32", "int64", "uint64", "bool_", "floating", "integer", "uint32", "uint16", "uint8", "int8"]: setattr(module, x, x)
            module.inf, module.nan, module.array = float("inf"), float("nan"), self.array
            module.get_active_memory, module.get_peak_memory, module.eval = (lambda: 0), (lambda: 0), (lambda *a: None)
            module.compile = lambda f=None, **k: (lambda f: f) if f is None else f
        elif module.__name__ == "openai_harmony":
            class MockEncoding:
                def __init__(self):
                    self.special_tokens = {"<|start|>": 200004, "<|end|>": 200007, "<|message|>": 200008, "<|channel|>": 200005, "<|return|>": 200002, "<|call|>": 200012, "analysis": 17195, "final": 17196, "commentary": 17197}
                    self.reverse_special = {v: k for k, v in self.special_tokens.items()}
                def stop_tokens_for_assistant_actions(self): return [200002, 200012]
                def encode(self, text, *a, **k):
                    if not isinstance(text, str): return [1, 2, 3]
                    tokens = []; import re; sorted_keys = sorted(self.special_tokens.keys(), key=len, reverse=True); pattern = "|".join(re.escape(k) for k in sorted_keys); parts = re.split(f"({pattern})", text)
                    for p in parts:
                        if not p: continue
                        if p in self.special_tokens: tokens.append(self.special_tokens[p])
                        else: tokens.extend([ord(c) + 1000 for c in p])
                    return tokens
                def decode(self, tokens, *a, **k):
                    res = []
                    for t in tokens:
                        if t in self.reverse_special: res.append(self.reverse_special[t])
                        elif t >= 1000: res.append(chr(t - 1000))
                    return "".join(res)
                def parse_messages_from_completion_tokens(self, tokens, *a, **k):
                    content = self.decode(tokens)
                    return [type("msg", (), {"channel": "final", "recipient": None, "content": [type("c", (), {"text": content})()], "thinking": "", "tool_calls": [], "role": "assistant"})()]
            class MockParser:
                reverse = {200000: "<|start|>", 200001: "<|end|>", 200008: "<|message|>", 200005: "<|channel|>", 200002: "<|return|>", 200012: "<|call|>", 17195: "analysis", 17196: "final", 17197: "commentary"}
                def __init__(self, *a, **k): self.current_channel, self.stop_token_ids, self._state, self.messages, self._current_message, self._recipient_override = "text", [200002, 200012], "idle", [], None, None
                @property
                def current_recipient(self): return self._recipient_override or "functions.Write"
                @current_recipient.setter
                def current_recipient(self, v): self._recipient_override = v
                def process(self, token, *a, **k):
                    if token == 200005: self._state = "channel"
                    elif token == 200008: self._state = "message"
                    elif token in (200007, 200002, 200012):
                        self._state = "idle"; self.current_channel = None
                        if self._current_message:
                            if "get_weather" in self._current_message.content_text: self._current_message.recipient = "functions.get_weather"
                            elif "Write" in self._current_message.content_text: self._current_message.recipient = "functions.Write"
                            self.messages.append(self._current_message); self._current_message = None
                    elif self._state == "channel": self.current_channel = self.reverse.get(token, "text"); self._state = "awaiting_message"
                    elif self._state == "message":
                        if not self._current_message: self._current_message = type("msg", (), {"role": "assistant", "content_text": "", "channel": self.current_channel, "recipient": self.current_recipient, "tool_calls": [], "tool_call_id": None, "content": []})()
                        if token >= 1000: char = chr(token - 1000); self._current_message.content_text += char; self._current_message.content.append(type("c", (), {"text": char})())
                def process_token(self, token, *a, **k):
                    old_state = self._state; self.process(token); stream = visible = None
                    if old_state == "message" and token not in self.stop_token_ids and token != 200007:
                        stream = token
                        if self.current_channel == "final": visible = token
                    return "", stream, visible, token in self.stop_token_ids
                def finalize(self, *a, **k): return []
                def reset(self, *a, **k): self.current_channel, self._state = "text", "idle"
            module.load_harmony_encoding, module.StreamableParser, module.Role = (lambda *a, **k: MockEncoding()), (lambda *a, **k: MockParser()), type("Role", (), {"USER": "user", "ASSISTANT": "assistant", "SYSTEM": "system"})

class MockMLXFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname in ("mlx", "openai_harmony") or fullname.startswith("mlx."): return importlib.machinery.ModuleSpec(fullname, MockMLXLoader())
        return None

def install_mock():
    """Install the mock into sys.meta_path."""
    if not any(isinstance(f, MockMLXFinder) for f in sys.meta_path):
        sys.meta_path.insert(0, MockMLXFinder())
        # Clear any failed or existing imports to ensure the mock is used
        for m in list(sys.modules.keys()):
            if m in ("mlx", "openai_harmony") or m.startswith("mlx."):
                del sys.modules[m]
