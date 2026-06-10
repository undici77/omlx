# SPDX-License-Identifier: Apache-2.0
import contextlib
import importlib.abc
import importlib.machinery
import json
import math
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np


def _map_dtype(dtype):
    if dtype == "bfloat16":
        return "float16"
    if isinstance(dtype, str) and dtype.startswith("uint"):
        try:
            bits = int(dtype[4:])
            if bits <= 8:
                return "uint8"
            if bits <= 16:
                return "uint16"
            if bits <= 32:
                return "uint32"
            return "uint64"
        except ValueError:
            pass
    if hasattr(dtype, "__name__"):
        return dtype.__name__
    return dtype


class MockMLXLoader(importlib.abc.Loader):
    class array:
        def __init__(self, data=None, dtype=None):
            if hasattr(data, "_data"):
                data = data._data
            if isinstance(data, (list, tuple)):
                def _unwrap_nested(value):
                    if hasattr(value, "_data"):
                        return value._data
                    if isinstance(value, (list, tuple)):
                        return [_unwrap_nested(v) for v in value]
                    return value
                data = _unwrap_nested(data)
            if isinstance(data, np.ndarray):
                self._data = data.copy()
            elif data is None:
                self._data = np.array([], dtype=_map_dtype(dtype) or "float32")
            else:
                try:
                    self._data = np.array(data, dtype=_map_dtype(dtype))
                except Exception:
                    self._data = np.array(data)
            mapped = _map_dtype(dtype)
            if mapped:
                try:
                    self._data = self._data.astype(mapped)
                except Exception:
                    pass
            if self._data.dtype == np.float64:
                self._data = self._data.astype(np.float32)
            self.dtype = dtype or str(self._data.dtype)
            self.shape, self.size, self.ndim = (
                self._data.shape,
                self._data.size,
                self._data.ndim,
            )

        def item(self):
            return self._data.item()

        def tolist(self):
            return self._data.tolist()

        def moveaxis(self, src, dst):
            return self.__class__(np.moveaxis(self._data, src, dst), dtype=self.dtype)

        def __float__(self):
            return float(self.item())

        def __int__(self):
            return int(self.item())

        def astype(self, dtype):
            return self.__class__(self._data, dtype=dtype)

        def reshape(self, *shape):
            if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
                shape = shape[0]
            return self.__class__(self._data.reshape(shape), dtype=self.dtype)

        def squeeze(self, axis=None):
            return self.__class__(np.squeeze(self._data, axis=axis), dtype=self.dtype)

        def transpose(self, *axes):
            return self.__class__(self._data.transpose(*axes), dtype=self.dtype)

        def view(self, dtype):
            mapped = _map_dtype(dtype)
            if dtype == "bfloat16":
                if self._data.dtype == np.uint16:
                    return self.__class__(self._data.view(np.float16), dtype=dtype)
                return self.__class__(self._data, dtype=dtype)
            if mapped:
                try:
                    return self.__class__(self._data.view(mapped), dtype=dtype)
                except Exception:
                    pass
            return self.astype(dtype)

        def max(self, axis=None, keepdims=False):
            return self.__class__(np.max(self._data, axis=axis, keepdims=keepdims))

        def min(self, axis=None, keepdims=False):
            return self.__class__(np.min(self._data, axis=axis, keepdims=keepdims))

        def sum(self, axis=None, keepdims=False):
            return self.__class__(np.sum(self._data, axis=axis, keepdims=keepdims))

        def mean(self, axis=None, keepdims=False):
            return self.__class__(np.mean(self._data, axis=axis, keepdims=keepdims))

        def flatten(self, order="C"):
            if isinstance(order, int):
                axis = order if order >= 0 else self._data.ndim + order
                new_shape = self._data.shape[:axis] + (-1,)
                return self.__class__(self._data.reshape(new_shape), dtype=self.dtype)
            return self.__class__(self._data.flatten(order), dtype=self.dtype)

        def __repr__(self):
            return f"mx.array(shape={self.shape}, dtype={self.dtype})"

        def __bool__(self):
            return bool(self.item())

        def __len__(self):
            return len(self._data) if self.ndim > 0 else 0

        def __iter__(self):
            for item in self._data:
                yield self.__class__(item)

        def __getitem__(self, i):
            return self.__class__(self._data[self._unwrap_index(i)], dtype=self.dtype)

        def __setitem__(self, i, v):
            self._data[self._unwrap_index(i)] = v._data if hasattr(v, "_data") else v

        @staticmethod
        def _unwrap_index(idx):
            if hasattr(idx, "_data"):
                return idx._data
            if isinstance(idx, tuple):
                return tuple(MockMLXLoader.array._unwrap_index(x) for x in idx)
            return idx

        @staticmethod
        def _unwrap(other):
            return other._data if hasattr(other, "_data") else other

        def __add__(self, other):
            return self.__class__(self._data + self._unwrap(other))

        def __radd__(self, other):
            return self.__class__(self._unwrap(other) + self._data)

        def __sub__(self, other):
            return self.__class__(self._data - self._unwrap(other))

        def __rsub__(self, other):
            return self.__class__(self._unwrap(other) - self._data)

        def __mul__(self, other):
            return self.__class__(self._data * self._unwrap(other))

        def __rmul__(self, other):
            return self.__class__(self._unwrap(other) * self._data)

        def __truediv__(self, other):
            return self.__class__(self._data / self._unwrap(other))

        def __rtruediv__(self, other):
            return self.__class__(self._unwrap(other) / self._data)

        def __matmul__(self, other):
            return self.__class__(self._data @ self._unwrap(other))

        def __eq__(self, other):
            return self.__class__(self._data == self._unwrap(other), dtype="bool_")

        def __lt__(self, other):
            return self.__class__(self._data < self._unwrap(other), dtype="bool_")

        def __le__(self, other):
            return self.__class__(self._data <= self._unwrap(other), dtype="bool_")

        def __gt__(self, other):
            return self.__class__(self._data > self._unwrap(other), dtype="bool_")

        def __ge__(self, other):
            return self.__class__(self._data >= self._unwrap(other), dtype="bool_")

        def __and__(self, other):
            return self.__class__(self._data & self._unwrap(other), dtype="bool_")

        def __rand__(self, other):
            return self.__class__(self._unwrap(other) & self._data, dtype="bool_")

        def __or__(self, other):
            return self.__class__(self._data | self._unwrap(other), dtype="bool_")

        def __ror__(self, other):
            return self.__class__(self._unwrap(other) | self._data, dtype="bool_")

        def __neg__(self):
            return self.__class__(-self._data)

        def __pow__(self, other):
            return self.__class__(self._data ** self._unwrap(other))

        def __rpow__(self, other):
            return self.__class__(self._unwrap(other) ** self._data)

        def __array__(self, dtype=None):
            return self._data.astype(dtype) if dtype else self._data

        def __int__(self):
            return int(self.item())

        @property
        def nbytes(self):
            return self._data.nbytes

        def __buffer__(self, flags):
            return self._data.__buffer__(flags)

        @property
        def at(self):
            outer = self

            class _AtIndexer:
                def __getitem__(self, idx):
                    idx = MockMLXLoader.array._unwrap_index(idx)

                    class _AtOp:
                        def add(self, val):
                            arr = outer._data.copy()
                            arr[idx] += MockMLXLoader.array._unwrap(val)
                            return outer.__class__(arr, dtype=outer.dtype)

                        def set(self, val):
                            arr = outer._data.copy()
                            arr[idx] = MockMLXLoader.array._unwrap(val)
                            return outer.__class__(arr, dtype=outer.dtype)

                    return _AtOp()

            return _AtIndexer()

        @property
        def T(self):
            return self.__class__(self._data.T, dtype=self.dtype)

    def create_module(self, spec):
        if spec.name in sys.modules:
            return sys.modules[spec.name]
        loader = self

        class MockModule(types.ModuleType):
            def __init__(self, name):
                super().__init__(name)
                self.__mock_items = {}
                self.__spec__ = importlib.machinery.ModuleSpec(name, None)

            def __getattr__(self, name):
                if name == "__file__":
                    return "mock_mlx.py"
                if name in ("__path__", "__all__"):
                    return None
                if name not in self.__mock_items:
                    if self.__name__ == "mlx.core":
                        if name == "gpu":
                            return type("Device", (), {})()
                        if name == "cpu":
                            return type("Device", (), {})()
                        if name == "metal":
                            m = MockModule("mlx.core.metal")
                            m.is_available = lambda: False
                            m.device_info = lambda: {
                                "memory_size": 16 * 1024 * 1024 * 1024,
                                "max_recommended_working_set_size": 12 * 1024 * 1024 * 1024,
                                "device_name": "Apple M2 Mock",
                                "max_buffer_length": 1 << 30,
                            }
                            m.clear_cache = lambda: None
                            return m
                        if name == "device_info":
                            return lambda: {
                                "memory_size": 16 * 1024 * 1024 * 1024,
                                "max_recommended_working_set_size": 12 * 1024 * 1024 * 1024,
                                "device_name": "Apple M2 Mock",
                                "max_buffer_length": 1 << 30,
                            }
                        if name == "copy":
                            return lambda a: loader.array(loader.array(a)._data.copy(), dtype=getattr(a, "dtype", None))
                        if name in ("random", "linalg", "distributed", "fast"):
                            return loader.create_module(
                                importlib.machinery.ModuleSpec(
                                    f"mlx.core.{name}", loader
                                )
                            )
                        if name == "stream":
                            return lambda *a, **k: contextlib.nullcontext()
                        if name == "contiguous":
                            return lambda a, **k: a
                        if name == "compile":
                            return lambda f, *a, **k: f
                        if name == "get_active_memory":
                            return lambda: 0
                        if name == "get_cache_memory":
                            return lambda: 0
                        if name == "softmax":
                            return lambda x, axis=-1: loader.array(
                                np.exp(loader.array(x)._data - np.max(loader.array(x)._data, axis=axis, keepdims=True))
                                / np.sum(np.exp(loader.array(x)._data - np.max(loader.array(x)._data, axis=axis, keepdims=True)), axis=axis, keepdims=True)
                            )
                        if name == "logsumexp":
                            return lambda a, axis=None, keepdims=False: loader.array(
                                np.log(np.sum(np.exp(loader.array(a)._data), axis=axis, keepdims=keepdims))
                            )
                        if name == "matmul":
                            return lambda a, b: loader.array(
                                loader.array(a)._data @ loader.array(b)._data
                            )
                        if name == "power":
                            return lambda a, b: loader.array(
                                np.power(
                                    loader.array(a)._data,
                                    loader.array(b)._data if hasattr(b, "_data") else b,
                                )
                            )
                        if name == "argmax":
                            return lambda a, axis=None: loader.array(np.argmax(loader.array(a)._data, axis=axis))
                        if name == "moveaxis":
                            return lambda a, src, dst: loader.array(np.moveaxis(loader.array(a)._data, src, dst))
                        if name == "unflatten":
                            return lambda a, axis, shape: loader.array(np.reshape(loader.array(a)._data, loader.array(a)._data.shape[:axis] + tuple(shape) + loader.array(a)._data.shape[axis+1:]))
                        if name == "put_along_axis":
                            return lambda a, indices, values, axis=-1: loader.array((lambda arr: (np.put_along_axis(arr, loader.array(indices)._data.astype(int), loader.array(values)._data if hasattr(values, "_data") or isinstance(values, (list, tuple, np.ndarray)) else values, axis=axis), arr)[1])(loader.array(a)._data.copy()))
                        if name == "issubdtype":
                            return lambda dt, kind: bool(np.issubdtype(np.dtype(_map_dtype(dt) or "float32"), np.floating if kind in ("floating", getattr(np, "floating", object())) else np.dtype(_map_dtype(kind) or "float32")))
                        if name == "from_fp8":
                            return lambda x, dtype=None, **k: loader.array(loader.array(x)._data, dtype=dtype)
                        if name == "dequantize":
                            return lambda qw, scales=None, biases=None, **k: loader.array(loader.array(qw)._data.astype(np.float32))
                        if name == "quantize":
                            return lambda w, group_size=64, bits=4, mode="affine", **k: (loader.array(np.round(loader.array(w)._data).astype(np.uint32), dtype="uint32"), loader.array(np.ones((loader.array(w).shape[0], max(1, loader.array(w).shape[1] // max(group_size, 1))), dtype=np.float32)), loader.array(np.zeros((loader.array(w).shape[0], max(1, loader.array(w).shape[1] // max(group_size, 1))), dtype=np.float32)))
                        if name == "clip":
                            return lambda a, a_min, a_max: loader.array(
                                np.clip(
                                    loader.array(a)._data,
                                    loader.array(a_min)._data if hasattr(a_min, "_data") else a_min,
                                    loader.array(a_max)._data if hasattr(a_max, "_data") else a_max,
                                )
                            )
                        if name == "get_peak_memory":
                            return lambda: 0
                        if name in ("mean", "max", "min", "sum", "all", "any"):
                            return lambda a, axis=None, keepdims=False: loader.array(
                                getattr(np, name)(
                                    loader.array(a)._data,
                                    axis=axis,
                                    keepdims=keepdims,
                                )
                            )
                        if name == "concatenate":
                            return lambda arrays, axis=0: loader.array(
                                np.concatenate(
                                    [loader.array(a)._data for a in arrays], axis=axis
                                )
                            )
                        if name == "stack":
                            return lambda arrays, axis=0: loader.array(
                                np.stack([loader.array(a)._data for a in arrays], axis=axis)
                            )
                        if name == "split":
                            return lambda a, indices_or_sections, axis=0: [
                                loader.array(x)
                                for x in np.split(
                                    loader.array(a)._data,
                                    indices_or_sections,
                                    axis=axis,
                                )
                            ]
                        if name == "repeat":
                            return lambda a, repeats, axis=None: loader.array(
                                np.repeat(loader.array(a)._data, repeats, axis=axis)
                            )
                        if name in ("zeros", "ones"):
                            return lambda s, dtype=None, **k: loader.array(
                                getattr(np, name)(s), dtype=dtype
                            )
                        if name in ("zeros_like", "ones_like"):
                            return lambda a, dtype=None: loader.array(
                                getattr(np, name)(loader.array(a)._data),
                                dtype=dtype or getattr(a, "dtype", None),
                            )
                        if name == "full":
                            return lambda s, v, dtype=None, **k: loader.array(
                                np.full(s, v), dtype=dtype
                            )
                        if name == "arange":
                            return lambda *a, dtype=None, **k: loader.array(
                                np.arange(*a), dtype=dtype
                            )
                        if name in (
                            "reshape",
                            "transpose",
                            "expand_dims",
                            "squeeze",
                            "broadcast_to",
                            "argsort",
                            "argpartition",
                            "cumsum",
                            "cos",
                            "sin",
                        ):
                            np_name = {
                                "expand_dims": "expand_dims",
                                "cumsum": "cumsum",
                                "cos": "cos",
                                "sin": "sin",
                            }.get(name, name)
                            return lambda a, *args, **kwargs: loader.array(
                                getattr(np, np_name)(
                                    loader.array(a)._data, *args, **kwargs
                                )
                            )
                        if name in ("equal", "array_equal", "allclose"):
                            return lambda a, b, **k: loader.array(
                                np.array_equal(
                                    loader.array(a)._data, loader.array(b)._data
                                )
                                if name != "allclose"
                                else np.allclose(
                                    loader.array(a)._data,
                                    loader.array(b)._data,
                                    **k,
                                ),
                                dtype="bool_",
                            )
                        if name == "maximum":
                            return lambda a, b: loader.array(
                                np.maximum(
                                    loader.array(a)._data,
                                    loader.array(b)._data if hasattr(b, "_data") or isinstance(b, (list, tuple, np.ndarray)) else b,
                                )
                            )
                        if name == "where":
                            def _where(cond, *rest):
                                cond_arr = loader.array(cond)._data
                                if not rest:
                                    return np.where(cond_arr)
                                if len(rest) != 2:
                                    raise TypeError("where() expects condition[, x, y]")
                                x, y = rest
                                x_arr = loader.array(x)._data if hasattr(x, "_data") or isinstance(x, (list, tuple, np.ndarray)) else x
                                y_arr = loader.array(y)._data if hasattr(y, "_data") or isinstance(y, (list, tuple, np.ndarray)) else y
                                return loader.array(np.where(cond_arr, x_arr, y_arr))
                            return _where
                        if name in ("exp", "log", "abs", "sqrt", "pad"):
                            return lambda a, *args, **kwargs: loader.array(
                                getattr(np, name)(loader.array(a)._data, *args, **kwargs)
                            )
                        if name == "load":
                            def _load(path, return_metadata=False, rm=False):
                                import struct

                                with open(path, "rb") as f:
                                    header_len = struct.unpack("<Q", f.read(8))[0]
                                    header = json.loads(f.read(header_len).decode("utf-8"))
                                    base = 8 + header_len
                                    tensors = {}
                                    metadata = dict(header.get("__metadata__") or {})
                                    dtype_map = {
                                        "F16": np.float16,
                                        "F32": np.float32,
                                        "BF16": np.uint16,
                                        "I8": np.int8,
                                        "I16": np.int16,
                                        "I32": np.int32,
                                        "I64": np.int64,
                                        "U8": np.uint8,
                                        "U16": np.uint16,
                                        "U32": np.uint32,
                                        "U64": np.uint64,
                                        "BOOL": np.bool_,
                                    }
                                    for key, info in header.items():
                                        if key == "__metadata__":
                                            continue
                                        start, end = info["data_offsets"]
                                        f.seek(base + start)
                                        buf = f.read(end - start)
                                        arr = np.frombuffer(buf, dtype=dtype_map[info["dtype"]]).reshape(info["shape"])
                                        tensors[key] = loader.array(arr).view("bfloat16") if info["dtype"] in ("BF16", "U16") else loader.array(arr)
                                if rm or return_metadata:
                                    return tensors, metadata
                                return tensors

                            return _load
                        if name == "save_safetensors":
                            def _save_safetensors(path, tensors, metadata=None):
                                from safetensors.numpy import save_file

                                save_file(
                                    {
                                        k: (
                                            np.array(v.view("uint16")._data)
                                            if hasattr(v, "dtype") and getattr(v, "dtype", None) == "bfloat16"
                                            else np.array(v._data if hasattr(v, "_data") else v)
                                        )
                                        for k, v in tensors.items()
                                    },
                                    str(path),
                                    metadata=metadata,
                                )
                                return None

                            return _save_safetensors
                        if name in ("eval", "async_eval", "synchronize"):
                            return lambda *a, **k: None
                        if name == "default_device":
                            return lambda: type("Device", (), {})()
                        if name == "new_thread_local_stream":
                            return lambda d: type("Stream", (), {})()
                        if name == "clear_cache":
                            return lambda: None
                        if name == "get_message_json":
                            return lambda *a, **k: ""
                        if name == "take_along_axis":
                            return lambda a, i, axis=None: loader.array(
                                np.take_along_axis(
                                    loader.array(a)._data,
                                    loader.array(i)._data.astype(int),
                                    axis=axis,
                                )
                            )
                        if name == "tree_flatten":
                            return lambda a, **k: list(a.items()) if hasattr(a, "items") else list(a)
                        if name == "stop_gradient":
                            return lambda a: a
                        if name == "load_tool_module":
                            return lambda *a, **k: _make_tool_module()

                    if self.__name__ in ("mlx_lm.models", "mlx_vlm.models") and name and name[0].islower():
                        mod = loader.create_module(
                            importlib.machinery.ModuleSpec(f"{self.__name__}.{name}", loader)
                        )
                        self.__mock_items[name] = mod
                        return mod

                    if self.__name__ in ("dflash_mlx.engine", "dflash_mlx.runtime") and name and name[0].islower():
                        mod = loader.create_module(
                            importlib.machinery.ModuleSpec(f"{self.__name__}.{name}", loader)
                        )
                        self.__mock_items[name] = mod
                        return mod

                    # --- openai_harmony specific classes (before generic uppercase catch-all) ---
                    if self.__name__ == "openai_harmony":
                        if name == "HarmonyEncoding":
                            self.__mock_items[name] = _MockHarmonyEncoding
                            return self.__mock_items[name]
                        if name == "StreamableParser":
                            self.__mock_items[name] = _MockStreamableParser
                            return self.__mock_items[name]
                        if name == "Role":
                            self.__mock_items[name] = _MockRole
                            return self.__mock_items[name]
                        if name == "HarmonyMessage":
                            self.__mock_items[name] = _MockHarmonyMessage
                            return self.__mock_items[name]

                    if name and name[0].isupper():
                        class MockClass:
                            def __init__(self, *a, **k):
                                self.keys = None
                                self.values = None
                                self.offset = 0
                                for key, val in k.items():
                                    setattr(self, key, val)
                                if a and hasattr(a[0], "vocab_size"):
                                    self.vocab_size = a[0].vocab_size
                                elif a and isinstance(a[0], dict) and "vocab_size" in a[0]:
                                    self.vocab_size = a[0]["vocab_size"]

                            def __call__(self, *a, **k):
                                if a:
                                    inputs = a[0]
                                    if hasattr(inputs, "shape"):
                                        B, L = inputs.shape[0], inputs.shape[1]
                                        vocab_size = getattr(self, "vocab_size", 1024)
                                        if hasattr(self, "args") and hasattr(self.args, "vocab_size"):
                                            vocab_size = self.args.vocab_size
                                        return loader.array(np.zeros((B, L, vocab_size)))
                                return loader.array(np.zeros((1, 1)))

                            @classmethod
                            def from_dict(cls, d):
                                return cls(**d)

                            @classmethod
                            def from_cache(cls, *a, **k):
                                return cls()

                            def make_cache(self):
                                cache_mod = sys.modules.get("mlx_lm.models.cache")
                                rotating_cls = getattr(cache_mod, "RotatingKVCache", object) if cache_mod else object
                                return [rotating_cls() for _ in range(10)]

                            def sanitize(self, weights):
                                return weights

                            def prompt(self, *a, **k):
                                return None

                            def _step(self, *a, **k):
                                pass

                            def next(self, *a, **k):
                                return None

                            def filter(self, *a, **k):
                                pass

                            def extend(self, *a, **k):
                                pass

                            def update_and_fetch(self, k, v):
                                self.keys, self.values = k, v
                                self.offset = getattr(k, "shape", (0, 0, 0, 0))[2] if hasattr(k, "shape") and len(k.shape) > 2 else 0
                                return k, v

                            @property
                            def state(self):
                                return (self.keys, self.values)

                            @state.setter
                            def state(self, s):
                                if s and len(s) >= 2:
                                    self.keys, self.values = s[:2]

                            def dequantize(self, *a, **k):
                                return (
                                    loader.array(np.zeros((1, 1))),
                                    loader.array(np.zeros((1, 1))),
                                )

                        MockClass.__name__ = name
                        MockClass.__module__ = self.__name__
                        self.__mock_items[name] = MockClass
                        return MockClass

                    if self.__name__.startswith(("mlx", "dflash_mlx", "openai_harmony")):
                        _captured_name = name

                        def _default_func(*args, _n=_captured_name, **kwargs):
                            # --- mlx.core arithmetic helpers ---
                            if _n == "floor":
                                arr = args[0]._data if hasattr(args[0], "_data") else np.asarray(args[0])
                                return loader.array(np.floor(arr))
                            if _n == "log":
                                arr = args[0]._data if hasattr(args[0], "_data") else np.asarray(args[0])
                                return loader.array(np.log(arr))
                            if _n == "log2":
                                arr = args[0]._data if hasattr(args[0], "_data") else np.asarray(args[0])
                                return loader.array(np.log2(arr))
                            if _n == "sqrt":
                                arr = args[0]._data if hasattr(args[0], "_data") else np.asarray(args[0])
                                return loader.array(np.sqrt(arr))
                            if _n == "square":
                                arr = args[0]._data if hasattr(args[0], "_data") else np.asarray(args[0])
                                return loader.array(arr ** 2)
                            if _n == "exp":
                                arr = args[0]._data if hasattr(args[0], "_data") else np.asarray(args[0])
                                return loader.array(np.exp(arr))
                            if _n == "sigmoid":
                                arr = args[0]._data if hasattr(args[0], "_data") else np.asarray(args[0])
                                return loader.array(1.0 / (1.0 + np.exp(-arr)))
                            if _n == "tanh":
                                arr = args[0]._data if hasattr(args[0], "_data") else np.asarray(args[0])
                                return loader.array(np.tanh(arr))
                            if _n == "abs":
                                arr = args[0]._data if hasattr(args[0], "_data") else np.asarray(args[0])
                                return loader.array(np.abs(arr))
                            if _n == "sign":
                                arr = args[0]._data if hasattr(args[0], "_data") else np.asarray(args[0])
                                return loader.array(np.sign(arr).astype(np.float32))
                            if _n == "sin":
                                arr = args[0]._data if hasattr(args[0], "_data") else np.asarray(args[0])
                                return loader.array(np.sin(arr))
                            if _n == "cos":
                                arr = args[0]._data if hasattr(args[0], "_data") else np.asarray(args[0])
                                return loader.array(np.cos(arr))
                            if _n == "reciprocal":
                                arr = args[0]._data if hasattr(args[0], "_data") else np.asarray(args[0])
                                return loader.array(1.0 / arr)
                            if _n == "negative":
                                arr = args[0]._data if hasattr(args[0], "_data") else np.asarray(args[0])
                                return loader.array(-arr)
                            if _n == "round":
                                arr = args[0]._data if hasattr(args[0], "_data") else np.asarray(args[0])
                                return loader.array(np.round(arr))
                            if _n == "ceil":
                                arr = args[0]._data if hasattr(args[0], "_data") else np.asarray(args[0])
                                return loader.array(np.ceil(arr))
                            if _n == "trunc":
                                arr = args[0]._data if hasattr(args[0], "_data") else np.asarray(args[0])
                                return loader.array(np.trunc(arr))
                            if _n == "isinf":
                                arr = args[0]._data if hasattr(args[0], "_data") else np.asarray(args[0])
                                return loader.array(np.isinf(arr))
                            if _n == "isnan":
                                arr = args[0]._data if hasattr(args[0], "_data") else np.asarray(args[0])
                                return loader.array(np.isnan(arr))
                            if _n == "greater":
                                a = args[0]._data if hasattr(args[0], "_data") else np.asarray(args[0])
                                b = args[1]._data if hasattr(args[1], "_data") else np.asarray(args[1])
                                return loader.array(a > b)
                            if _n == "less":
                                a = args[0]._data if hasattr(args[0], "_data") else np.asarray(args[0])
                                b = args[1]._data if hasattr(args[1], "_data") else np.asarray(args[1])
                                return loader.array(a < b)
                            if _n == "equal":
                                a = args[0]._data if hasattr(args[0], "_data") else np.asarray(args[0])
                                b = args[1]._data if hasattr(args[1], "_data") else np.asarray(args[1])
                                return loader.array(a == b)
                            if _n == "greater_equal":
                                a = args[0]._data if hasattr(args[0], "_data") else np.asarray(args[0])
                                b = args[1]._data if hasattr(args[1], "_data") else np.asarray(args[1])
                                return loader.array(a >= b)
                            if _n == "less_equal":
                                a = args[0]._data if hasattr(args[0], "_data") else np.asarray(args[0])
                                b = args[1]._data if hasattr(args[1], "_data") else np.asarray(args[1])
                                return loader.array(a <= b)
                            if _n == "not_equal":
                                a = args[0]._data if hasattr(args[0], "_data") else np.asarray(args[0])
                                b = args[1]._data if hasattr(args[1], "_data") else np.asarray(args[1])
                                return loader.array(a != b)
                            if _n == "logical_and":
                                a = args[0]._data if hasattr(args[0], "_data") else np.asarray(args[0])
                                b = args[1]._data if hasattr(args[1], "_data") else np.asarray(args[1])
                                return loader.array(np.logical_and(a, b))
                            if _n == "logical_or":
                                a = args[0]._data if hasattr(args[0], "_data") else np.asarray(args[0])
                                b = args[1]._data if hasattr(args[1], "_data") else np.asarray(args[1])
                                return loader.array(np.logical_or(a, b))
                            if _n == "logical_not":
                                arr = args[0]._data if hasattr(args[0], "_data") else np.asarray(args[0])
                                return loader.array(np.logical_not(arr))
                            if _n == "maximum":
                                a = args[0]._data if hasattr(args[0], "_data") else np.asarray(args[0])
                                b = args[1]._data if hasattr(args[1], "_data") else np.asarray(args[1])
                                return loader.array(np.maximum(a, b))
                            if _n == "minimum":
                                a = args[0]._data if hasattr(args[0], "_data") else np.asarray(args[0])
                                b = args[1]._data if hasattr(args[1], "_data") else np.asarray(args[1])
                                return loader.array(np.minimum(a, b))
                            if _n == "add":
                                a = args[0]._data if hasattr(args[0], "_data") else np.asarray(args[0])
                                b = args[1]._data if hasattr(args[1], "_data") else np.asarray(args[1])
                                return loader.array(a + b)
                            if _n == "subtract":
                                a = args[0]._data if hasattr(args[0], "_data") else np.asarray(args[0])
                                b = args[1]._data if hasattr(args[1], "_data") else np.asarray(args[1])
                                return loader.array(a - b)
                            if _n == "multiply":
                                a = args[0]._data if hasattr(args[0], "_data") else np.asarray(args[0])
                                b = args[1]._data if hasattr(args[1], "_data") else np.asarray(args[1])
                                return loader.array(a * b)
                            if _n == "divide":
                                a = args[0]._data if hasattr(args[0], "_data") else np.asarray(args[0])
                                b = args[1]._data if hasattr(args[1], "_data") else np.asarray(args[1])
                                return loader.array(a / b)
                            if _n == "power":
                                a = args[0]._data if hasattr(args[0], "_data") else np.asarray(args[0])
                                b = args[1]._data if hasattr(args[1], "_data") else np.asarray(args[1])
                                return loader.array(a ** b)
                            if _n == "remainder":
                                a = args[0]._data if hasattr(args[0], "_data") else np.asarray(args[0])
                                b = args[1]._data if hasattr(args[1], "_data") else np.asarray(args[1])
                                return loader.array(a % b)
                            if _n == "matmul":
                                a = args[0]._data if hasattr(args[0], "_data") else np.asarray(args[0])
                                b = args[1]._data if hasattr(args[1], "_data") else np.asarray(args[1])
                                return loader.array(np.matmul(a, b))
                            if _n == "stack":
                                arrays = [a._data if hasattr(a, "_data") else np.asarray(a) for a in args]
                                axis = kwargs.get("axis", 0)
                                return loader.array(np.stack(arrays, axis=axis))
                            if _n == "concatenate":
                                arrays = [a._data if hasattr(a, "_data") else np.asarray(a) for a in args]
                                axis = kwargs.get("axis", 0)
                                return loader.array(np.concatenate(arrays, axis=axis))
                            if _n == "split":
                                arr = args[0]._data if hasattr(args[0], "_data") else np.asarray(args[0])
                                indices_or_sections = args[1] if len(args) > 1 else kwargs.get("indices_or_sections", 1)
                                axis = kwargs.get("axis", 0)
                                return [loader.array(s) for s in np.split(arr, indices_or_sections, axis=axis)]
                            if _n == "argmax":
                                arr = args[0]._data if hasattr(args[0], "_data") else np.asarray(args[0])
                                axis = kwargs.get("axis", None)
                                return loader.array(np.argmax(arr, axis=axis))
                            if _n == "argmin":
                                arr = args[0]._data if hasattr(args[0], "_data") else np.asarray(args[0])
                                axis = kwargs.get("axis", None)
                                return loader.array(np.argmin(arr, axis=axis))
                            if _n == "softmax":
                                arr = args[0]._data if hasattr(args[0], "_data") else np.asarray(args[0])
                                axis = kwargs.get("axis", -1)
                                exp_arr = np.exp(arr - np.max(arr, axis=axis, keepdims=True))
                                return loader.array(exp_arr / exp_arr.sum(axis=axis, keepdims=True))
                            if _n == "log_softmax":
                                arr = args[0]._data if hasattr(args[0], "_data") else np.asarray(args[0])
                                axis = kwargs.get("axis", -1)
                                max_arr = np.max(arr, axis=axis, keepdims=True)
                                exp_arr = np.exp(arr - max_arr)
                                return loader.array(arr - max_arr - np.log(exp_arr.sum(axis=axis, keepdims=True)))
                            if _n == "softmax_with_temperature":
                                arr = args[0]._data if hasattr(args[0], "_data") else np.asarray(args[0])
                                temperature = args[1] if len(args) > 1 else kwargs.get("temperature", 1.0)
                                axis = kwargs.get("axis", -1)
                                arr = arr / temperature
                                exp_arr = np.exp(arr - np.max(arr, axis=axis, keepdims=True))
                                return loader.array(exp_arr / exp_arr.sum(axis=axis, keepdims=True))
                            if _n == "topk":
                                arr = args[0]._data if hasattr(args[0], "_data") else np.asarray(args[0])
                                k = args[1] if len(args) > 1 else kwargs.get("k", 1)
                                axis = kwargs.get("axis", -1)
                                indices = np.argpartition(arr, -k, axis=axis)[:, -k:] if arr.ndim > 1 else np.argpartition(arr, -k)[-k:]
                                sorted_idx = np.argsort(arr.take(indices, axis=axis), axis=axis)[..., ::-1]
                                values = np.take_along_axis(arr, sorted_idx, axis=axis)
                                indices = np.take_along_axis(indices, sorted_idx, axis=axis)
                                return loader.array(values), loader.array(indices)
                            if _n == "is_floating_point":
                                arr = args[0]._data if hasattr(args[0], "_data") else np.asarray(args[0])
                                return np.issubdtype(arr.dtype, np.floating)
                            if _n == "broadcast_to":
                                arr = args[0]._data if hasattr(args[0], "_data") else np.asarray(args[0])
                                shape = args[1] if len(args) > 1 else kwargs.get("shape", arr.shape)
                                return loader.array(np.broadcast_to(arr, shape))
                            if _n == "expand_dims":
                                arr = args[0]._data if hasattr(args[0], "_data") else np.asarray(args[0])
                                axis = args[1] if len(args) > 1 else kwargs.get("axis", 0)
                                return loader.array(np.expand_dims(arr, axis=axis))
                            if _n == "squeeze":
                                arr = args[0]._data if hasattr(args[0], "_data") else np.asarray(args[0])
                                axis = args[1] if len(args) > 1 else kwargs.get("axis", None)
                                return loader.array(np.squeeze(arr, axis=axis))
                            if _n == "slice":
                                arr = args[0]._data if hasattr(args[0], "_data") else np.asarray(args[0])
                                starts = args[1] if len(args) > 1 else kwargs.get("starts", [0])
                                ends = args[2] if len(args) > 2 else kwargs.get("ends", arr.shape)
                                strides = args[3] if len(args) > 3 else kwargs.get("strides", [1])
                                if not isinstance(starts, (list, tuple)):
                                    starts = [starts]
                                if not isinstance(ends, (list, tuple)):
                                    ends = [ends]
                                if not isinstance(strides, (list, tuple)):
                                    strides = [strides]
                                slices = [slice(s, e, st) for s, e, st in zip(starts, ends, strides)]
                                return loader.array(arr[tuple(slices)])
                            if _n == "full":
                                shape = args[0] if args else kwargs.get("shape", ())
                                fill_value = args[1] if len(args) > 1 else kwargs.get("fill_value", 0)
                                dtype = kwargs.get("dtype", None)
                                return loader.array(np.full(shape, fill_value, dtype=dtype))
                            if _n == "zeros":
                                shape = args[0] if args else kwargs.get("shape", ())
                                dtype = kwargs.get("dtype", None)
                                return loader.array(np.zeros(shape, dtype=dtype))
                            if _n == "ones":
                                shape = args[0] if args else kwargs.get("shape", ())
                                dtype = kwargs.get("dtype", None)
                                return loader.array(np.ones(shape, dtype=dtype))
                            if _n == "arange":
                                stop = args[0] if len(args) > 0 else kwargs.get("stop", 0)
                                start = args[1] if len(args) > 1 else kwargs.get("start", 0)
                                step = args[2] if len(args) > 2 else kwargs.get("step", 1)
                                return loader.array(np.arange(start, stop, step))
                            if _n == "linspace":
                                start = args[0] if len(args) > 0 else kwargs.get("start", 0)
                                stop = args[1] if len(args) > 1 else kwargs.get("stop", 1)
                                num = args[2] if len(args) > 2 else kwargs.get("num", 50)
                                return loader.array(np.linspace(start, stop, num))
                            if _n == "eye":
                                n = args[0] if len(args) > 0 else kwargs.get("n", 1)
                                m = args[1] if len(args) > 1 else kwargs.get("m", n)
                                return loader.array(np.eye(n, m))
                            if _n == "random_uniform":
                                shape = args[0] if len(args) > 0 else kwargs.get("shape", ())
                                low = kwargs.get("low", 0.0)
                                high = kwargs.get("high", 1.0)
                                return loader.array(np.random.uniform(low, high, shape))
                            if _n == "random_normal":
                                shape = args[0] if len(args) > 0 else kwargs.get("shape", ())
                                mean = kwargs.get("mean", 0.0)
                                std = kwargs.get("std", 1.0)
                                return loader.array(np.random.normal(mean, std, shape))
                            if _n == "random_bernoulli":
                                shape = args[0] if len(args) > 0 else kwargs.get("shape", ())
                                p = kwargs.get("p", 0.5)
                                return loader.array(np.random.binomial(1, p, shape))
                            if _n == "random_categorical":
                                shape = args[0] if len(args) > 0 else kwargs.get("shape", ())
                                logits = args[1] if len(args) > 1 else kwargs.get("logits", None)
                                if logits is not None:
                                    arr = logits._data if hasattr(logits, "_data") else np.asarray(logits)
                                    probs = np.exp(arr - np.max(arr))
                                    probs = probs / probs.sum()
                                else:
                                    probs = None
                                return loader.array(np.random.choice(np.prod(shape) if shape else 1, p=probs))
                            if _n == "random_choice":
                                shape = args[0] if len(args) > 0 else kwargs.get("shape", ())
                                high = args[1] if len(args) > 1 else kwargs.get("high", None)
                                if high is not None:
                                    low = args[2] if len(args) > 2 else kwargs.get("low", 0)
                                    return loader.array(np.random.randint(low, high, shape))
                                return loader.array(np.random.choice(high if isinstance(high, (list, np.ndarray)) else shape[0] if shape else 1, size=shape))
                            if _n == "random_permutation":
                                x = args[0] if len(args) > 0 else kwargs.get("x", None)
                                if x is not None:
                                    arr = x._data if hasattr(x, "_data") else np.asarray(x)
                                    return loader.array(np.random.permutation(arr))
                                n = args[0] if len(args) > 0 else kwargs.get("n", 10)
                                return loader.array(np.random.permutation(n))
                            if _n == "random_gamma":
                                shape = args[0] if len(args) > 0 else kwargs.get("shape", ())
                                alpha = args[1] if len(args) > 1 else kwargs.get("alpha", 1.0)
                                beta = args[2] if len(args) > 2 else kwargs.get("beta", 1.0)
                                return loader.array(np.random.gamma(alpha, beta, shape))
                            if _n == "all":
                                arr = args[0]._data if hasattr(args[0], "_data") else np.asarray(args[0])
                                axis = kwargs.get("axis", None)
                                keepdims = kwargs.get("keepdims", False)
                                return loader.array(np.all(arr, axis=axis, keepdims=keepdims))
                            if _n == "any":
                                arr = args[0]._data if hasattr(args[0], "_data") else np.asarray(args[0])
                                axis = kwargs.get("axis", None)
                                keepdims = kwargs.get("keepdims", False)
                                return loader.array(np.any(arr, axis=axis, keepdims=keepdims))
                            if _n == "sum":
                                arr = args[0]._data if hasattr(args[0], "_data") else np.asarray(args[0])
                                axis = kwargs.get("axis", None)
                                keepdims = kwargs.get("keepdims", False)
                                return loader.array(np.sum(arr, axis=axis, keepdims=keepdims))
                            if _n == "mean":
                                arr = args[0]._data if hasattr(args[0], "_data") else np.asarray(args[0])
                                axis = kwargs.get("axis", None)
                                keepdims = kwargs.get("keepdims", False)
                                return loader.array(np.mean(arr, axis=axis, keepdims=keepdims))
                            if _n == "std":
                                arr = args[0]._data if hasattr(args[0], "_data") else np.asarray(args[0])
                                axis = kwargs.get("axis", None)
                                keepdims = kwargs.get("keepdims", False)
                                return loader.array(np.std(arr, axis=axis, keepdims=keepdims))
                            if _n == "var":
                                arr = args[0]._data if hasattr(args[0], "_data") else np.asarray(args[0])
                                axis = kwargs.get("axis", None)
                                keepdims = kwargs.get("keepdims", False)
                                return loader.array(np.var(arr, axis=axis, keepdims=keepdims))
                            if _n == "max":
                                arr = args[0]._data if hasattr(args[0], "_data") else np.asarray(args[0])
                                axis = kwargs.get("axis", None)
                                keepdims = kwargs.get("keepdims", False)
                                return loader.array(np.max(arr, axis=axis, keepdims=keepdims))
                            if _n == "min":
                                arr = args[0]._data if hasattr(args[0], "_data") else np.asarray(args[0])
                                axis = kwargs.get("axis", None)
                                keepdims = kwargs.get("keepdims", False)
                                return loader.array(np.min(arr, axis=axis, keepdims=keepdims))
                            if _n == "prod":
                                arr = args[0]._data if hasattr(args[0], "_data") else np.asarray(args[0])
                                axis = kwargs.get("axis", None)
                                keepdims = kwargs.get("keepdims", False)
                                return loader.array(np.prod(arr, axis=axis, keepdims=keepdims))
                            if _n == "cumsum":
                                arr = args[0]._data if hasattr(args[0], "_data") else np.asarray(args[0])
                                axis = kwargs.get("axis", None)
                                return loader.array(np.cumsum(arr, axis=axis))
                            if _n == "cumprod":
                                arr = args[0]._data if hasattr(args[0], "_data") else np.asarray(args[0])
                                axis = kwargs.get("axis", None)
                                return loader.array(np.cumprod(arr, axis=axis))
                            if _n == "clip":
                                arr = args[0]._data if hasattr(args[0], "_data") else np.asarray(args[0])
                                a_min = args[1] if len(args) > 1 else kwargs.get("a_min", None)
                                a_max = args[2] if len(args) > 2 else kwargs.get("a_max", None)
                                return loader.array(np.clip(arr, a_min, a_max))
                            if _n == "where":
                                condition = args[0]._data if hasattr(args[0], "_data") else np.asarray(args[0])
                                x = args[1]._data if hasattr(args[1], "_data") else np.asarray(args[1]) if len(args) > 1 else None
                                y = args[2]._data if hasattr(args[2], "_data") else np.asarray(args[2]) if len(args) > 2 else None
                                if x is not None and y is not None:
                                    return loader.array(np.where(condition, x, y))
                                return loader.array(np.where(condition))
                            if _n == "nonzero":
                                arr = args[0]._data if hasattr(args[0], "_data") else np.asarray(args[0])
                                return [loader.array(idx) for idx in np.nonzero(arr)]
                            if _n == "unique":
                                arr = args[0]._data if hasattr(args[0], "_data") else np.asarray(args[0])
                                return loader.array(np.unique(arr))
                            if _n == "unique_counts":
                                arr = args[0]._data if hasattr(args[0], "_data") else np.asarray(args[0])
                                values, counts = np.unique(arr, return_counts=True)
                                return loader.array(values), loader.array(counts)
                            if _n == "unique_inverse":
                                arr = args[0]._data if hasattr(args[0], "_data") else np.asarray(args[0])
                                values, inverse = np.unique(arr, return_inverse=True)
                                return loader.array(values), loader.array(inverse)
                            if _n == "bincount":
                                arr = args[0]._data if hasattr(args[0], "_data") else np.asarray(args[0])
                                weights = args[1]._data if len(args) > 1 and hasattr(args[1], "_data") else args[1] if len(args) > 1 else None
                                minlength = kwargs.get("minlength", 0)
                                if weights is not None:
                                    return loader.array(np.bincount(arr.astype(int), weights=weights, minlength=minlength))
                                return loader.array(np.bincount(arr.astype(int), minlength=minlength))
                            if _n == "histogram":
                                arr = args[0]._data if hasattr(args[0], "_data") else np.asarray(args[0])
                                bins = args[1] if len(args) > 1 else kwargs.get("bins", 10)
                                counts, edges = np.histogram(arr, bins=bins)
                                return loader.array(counts), loader.array(edges)
                            if _n == "convolve":
                                arr = args[0]._data if hasattr(args[0], "_data") else np.asarray(args[0])
                                kernel = args[1]._data if len(args) > 1 and hasattr(args[1], "_data") else np.asarray(args[1])
                                mode = kwargs.get("mode", "full")
                                return loader.array(np.convolve(arr, kernel, mode=mode))
                            if _n == "pad":
                                arr = args[0]._data if hasattr(args[0], "_data") else np.asarray(args[0])
                                pad_width = args[1] if len(args) > 1 else kwargs.get("pad_width", 0)
                                constant_values = kwargs.get("constant_values", 0)
                                return loader.array(np.pad(arr, pad_width, constant_values=constant_values))
                            if _n == "flip":
                                arr = args[0]._data if hasattr(args[0], "_data") else np.asarray(args[0])
                                axis = args[1] if len(args) > 1 else kwargs.get("axis", None)
                                return loader.array(np.flip(arr, axis=axis))
                            if _n == "roll":
                                arr = args[0]._data if hasattr(args[0], "_data") else np.asarray(args[0])
                                shift = args[1] if len(args) > 1 else kwargs.get("shift", 1)
                                axis = args[2] if len(args) > 2 else kwargs.get("axis", None)
                                return loader.array(np.roll(arr, shift, axis=axis))
                            if _n == "swapaxes":
                                arr = args[0]._data if hasattr(args[0], "_data") else np.asarray(args[0])
                                axis1 = args[1] if len(args) > 1 else kwargs.get("axis1", 0)
                                axis2 = args[2] if len(args) > 2 else kwargs.get("axis2", 1)
                                return loader.array(np.swapaxes(arr, axis1, axis2))
                            if _n == "transpose":
                                arr = args[0]._data if hasattr(args[0], "_data") else np.asarray(args[0])
                                axes = args[1] if len(args) > 1 else kwargs.get("axes", None)
                                return loader.array(np.transpose(arr, axes))
                            if _n == "repeat":
                                arr = args[0]._data if hasattr(args[0], "_data") else np.asarray(args[0])
                                repeats = args[1] if len(args) > 1 else kwargs.get("repeats", 1)
                                axis = args[2] if len(args) > 2 else kwargs.get("axis", None)
                                return loader.array(np.repeat(arr, repeats, axis=axis))
                            if _n == "tile":
                                arr = args[0]._data if hasattr(args[0], "_data") else np.asarray(args[0])
                                reps = args[1] if len(args) > 1 else kwargs.get("reps", 1)
                                return loader.array(np.tile(arr, reps))
                            if _n == "take":
                                arr = args[0]._data if hasattr(args[0], "_data") else np.asarray(args[0])
                                indices = args[1]._data if len(args) > 1 and hasattr(args[1], "_data") else np.asarray(args[1])
                                axis = kwargs.get("axis", None)
                                return loader.array(np.take(arr, indices.astype(int), axis=axis))
                            if _n == "take_along_axis":
                                arr = args[0]._data if hasattr(args[0], "_data") else np.asarray(args[0])
                                indices = args[1]._data if len(args) > 1 and hasattr(args[1], "_data") else np.asarray(args[1])
                                axis = kwargs.get("axis", 0)
                                return loader.array(np.take_along_axis(arr, indices, axis=axis))
                            if _n == "moveaxis":
                                arr = args[0]._data if hasattr(args[0], "_data") else np.asarray(args[0])
                                source = args[1] if len(args) > 1 else kwargs.get("source", 0)
                                destination = args[2] if len(args) > 2 else kwargs.get("destination", 0)
                                return loader.array(np.moveaxis(arr, source, destination))
                            if _n == "broadcast_arrays":
                                arrays = [a._data if hasattr(a, "_data") else np.asarray(a) for a in args]
                                result = np.broadcast_arrays(*arrays)
                                return [loader.array(r) for r in result]
                            if _n == "asarray":
                                data = args[0]
                                dtype = kwargs.get("dtype", None)
                                return loader.array(data, dtype=dtype)
                            if _n == "asarray_like":
                                data = args[0]
                                like = args[1] if len(args) > 1 else kwargs.get("like", None)
                                dtype = kwargs.get("dtype", like.dtype if hasattr(like, "dtype") else None)
                                return loader.array(data, dtype=dtype)
                            if _n == "astype":
                                arr = args[0]._data if hasattr(args[0], "_data") else np.asarray(args[0])
                                dtype = args[1] if len(args) > 1 else kwargs.get("dtype", None)
                                mapped = _map_dtype(dtype) if dtype else None
                                return loader.array(arr, dtype=mapped)
                            if _n == "astype_like":
                                arr = args[0]._data if hasattr(args[0], "_data") else np.asarray(args[0])
                                like = args[1] if len(args) > 1 else kwargs.get("like", None)
                                dtype = like.dtype if hasattr(like, "dtype") else None
                                return loader.array(arr, dtype=dtype)
                            if _n == "real":
                                arr = args[0]._data if hasattr(args[0], "_data") else np.asarray(args[0])
                                return loader.array(np.real(arr))
                            if _n == "imag":
                                arr = args[0]._data if hasattr(args[0], "_data") else np.asarray(args[0])
                                return loader.array(np.imag(arr))
                            if _n == "angle":
                                arr = args[0]._data if hasattr(args[0], "_data") else np.asarray(args[0])
                                return loader.array(np.angle(arr))
                            if _n == "conj":
                                arr = args[0]._data if hasattr(args[0], "_data") else np.asarray(args[0])
                                return loader.array(np.conj(arr))
                            if _n == "conjugate":
                                arr = args[0]._data if hasattr(args[0], "_data") else np.asarray(args[0])
                                return loader.array(np.conjugate(arr))
                            if _n == "real_if_close":
                                arr = args[0]._data if hasattr(args[0], "_data") else np.asarray(args[0])
                                return loader.array(arr.real if np.iscomplexobj(arr) else arr)
                            if _n == "disp":
                                if args:
                                    print(args[0])
                                return None
                            if _n == "squeeze_like":
                                arr = args[0]._data if hasattr(args[0], "_data") else np.asarray(args[0])
                                like = args[1] if len(args) > 1 else kwargs.get("like", None)
                                return loader.array(np.squeeze(arr))
                            if _n == "result_type":
                                return "float32"
                            if _n == "promote_types":
                                return "float32"
                            if _n == "can_cast":
                                return True
                            if _n == "issubdtype":
                                return True
                            if _n == "isscalar":
                                return isinstance(args[0], (int, float, str, bool))
                            if _n == "iscomplex":
                                arr = args[0]._data if hasattr(args[0], "_data") else np.asarray(args[0])
                                return bool(np.iscomplexobj(arr))
                            if _n == "isreal":
                                arr = args[0]._data if hasattr(args[0], "_data") else np.asarray(args[0])
                                return bool(not np.iscomplexobj(arr))
                            if _n == "ndim":
                                arr = args[0]._data if hasattr(args[0], "_data") else np.asarray(args[0])
                                return int(np.ndim(arr))
                            if _n == "shape":
                                arr = args[0]._data if hasattr(args[0], "_data") else np.asarray(args[0])
                                return arr.shape
                            if _n == "size":
                                arr = args[0]._data if hasattr(args[0], "_data") else np.asarray(args[0])
                                return int(arr.size)
                            if _n == "dtype":
                                return "float32"
                            if _n == "eval":
                                if args:
                                    arr = args[0]._data if hasattr(args[0], "_data") else np.asarray(args[0])
                                    return loader.array(arr)
                                return loader.array(np.array([]))
                            if _n == "sync":
                                return None
                            if _n == "save":
                                return None
                            if _n == "load":
                                return loader.array(np.array([1.0, 2.0, 3.0]))
                            if _n == "save_safetensors":
                                return None
                            if _n == "load_safetensors":
                                return loader.array(np.array([1.0, 2.0, 3.0]))
                            if _n == "serialize":
                                return b"mock_serialized_data"
                            if _n == "deserialize":
                                return loader.array(np.array([1.0, 2.0, 3.0]))
                            if _n == "default_device":
                                return MockModule("mlx.core.device")
                            if _n == "default_stream":
                                return None
                            if _n == "device":
                                return MockModule("mlx.core.device")
                            if _n == "stream":
                                return MockModule("mlx.core.stream")
                            if _n == "gpu":
                                return MockModule("mlx.core.gpu")
                            if _n == "cpu":
                                return MockModule("mlx.core.cpu")
                            if _n == "metal":
                                return MockModule("mlx.core.metal")

                            # --- openai_harmony functions ---
                            if _n == "load_harmony_encoding":
                                return _mock_load_harmony_encoding(args[0] if args else "HarmonyGptOss")
                            if _n == "HarmonyEncoding":
                                return _MockHarmonyEncoding
                            if _n == "StreamableParser":
                                return _MockStreamableParser
                            if _n == "Role":
                                return _MockRole
                            if _n == "HarmonyMessage":
                                return _MockHarmonyMessage

                            # --- mlx_lm functions (already handled above, keep as fallback) ---
                            if _n == "is_applied":
                                return False
                            if _n == "is_mtp_active":
                                return False
                            if _n == "_infer_tool_parser":
                                return "json"
                            if _n == "extract_text_from_content":
                                return _extract_text_from_content(args[0] if args else None)
                            if _n == "get_message_json":
                                return _get_message_json(*args, **kwargs)
                            if _n == "load_tool_module":
                                return _make_tool_module()
                            if _n == "tree_flatten":
                                return list(args[0].items()) if args and hasattr(args[0], "items") else list(args[0])
                            if _n == "load_config":
                                return {"model_type": "test"}
                            if _n == "load_chat_template":
                                return None
                            if _n == "make_logits_processors":
                                return []
                            if _n == "runtime_config_from_defaults":
                                from types import SimpleNamespace
                                defaults = {"draft_window_size": 1024, "draft_sink_size": 64, "verify_mode": "adaptive"}
                                defaults.update({k: v for k, v in kwargs.items() if v is not None})
                                return SimpleNamespace(**defaults)
                            raise NotImplementedError(
                                f"mlx.{_n}() is not implemented in the MLX mock. Add it to omlx/utils/mlx_mock.py."
                            )

                        self.__mock_items[name] = _default_func
                    else:
                        self.__mock_items[name] = MockModule(f"{self.__name__}.{name}")
                return self.__mock_items[name]

        def _extract_text_from_content(content):
            if isinstance(content, list):
                parts = []
                for item in content:
                    if isinstance(item, str):
                        parts.append(item)
                    elif isinstance(item, dict):
                        if item.get("type") in ("text", "input_text"):
                            parts.append(item.get("text", ""))
                return " ".join(p for p in parts if p)
            if content is None:
                return ""
            return content

        def _get_message_json(
            model_type,
            content,
            role,
            skip_image_token=True,
            skip_audio_token=True,
            num_images=0,
            num_audios=0,
            **kwargs,
        ):
            text = _extract_text_from_content(content)
            if role == "user" and not skip_image_token and num_images > 0:
                parts = []
                if text:
                    parts.append({"type": "text", "text": text})
                parts.extend({"type": "image"} for _ in range(num_images))
                return {"role": role, "content": parts}
            if role == "user" and not skip_audio_token and num_audios > 0:
                parts = []
                if text:
                    parts.append({"type": "text", "text": text})
                parts.extend({"type": "audio"} for _ in range(num_audios))
                return {"role": role, "content": parts}
            return {"role": role, "content": text if isinstance(text, str) else str(text)}

        def _infer_tool_parser(template):
            if not template:
                return None
            if "<tool_call>" in template and "<function=" in template:
                return "qwen3_coder"
            if "<tool_call>" in template or "tool_call.name" in template:
                return "json_tools"
            return None

        def _make_tool_module():
            m = MockModule("tool_module")
            m.tool_call_start = "<tool_call>"
            m.tool_call_end = "</tool_call>"
            m.parse_tool_call = lambda *a, **k: {}
            return m

        if spec.name == "mlx.nn":
            m = MockModule(spec.name)

            class Module:
                def __init__(self, *args, **kwargs):
                    super().__setattr__("_parameters", {})
                    super().__setattr__("_modules", {})
                    super().__setattr__("_module_lists", {})

                def __setattr__(self, name, value):
                    super().__setattr__(name, value)
                    if isinstance(value, loader.array):
                        self._parameters[name] = value
                    elif hasattr(value, "parameters"):
                        self._modules[name] = value
                    elif isinstance(value, (list, tuple)) and any(hasattr(v, "parameters") for v in value):
                        self._module_lists[name] = list(value)

                def __call__(self, *args, **kwargs):
                    return args[0] if args else loader.array(np.zeros((1, 1)))

                def parameters(self, prefix=""):
                    params = {f"{prefix}{name}": value for name, value in self._parameters.items()}
                    for name, module in self._modules.items():
                        params.update(module.parameters(prefix=f"{prefix}{name}."))
                    for name, modules in self._module_lists.items():
                        for idx, module in enumerate(modules):
                            params.update(module.parameters(prefix=f"{prefix}{name}.{idx}."))
                    return params

                def load_weights(self, weights, strict=True):
                    for name, value in weights:
                        target = self
                        parts = name.split(".")
                        for part in parts[:-1]:
                            if part.isdigit():
                                target = target[int(part)]
                            else:
                                target = getattr(target, part)
                        setattr(target, parts[-1], loader.array(value))

            class Linear(Module):
                def __init__(self, in_features, out_features, bias=True, *args, **kwargs):
                    super().__init__()
                    self.weight = loader.array(np.zeros((out_features, in_features), dtype=np.float32))
                    if bias:
                        self.bias = loader.array(np.zeros((out_features,), dtype=np.float32))

            class Embedding(Module):
                def __init__(self, num_embeddings, embedding_dim, *args, **kwargs):
                    super().__init__()
                    self.weight = loader.array(np.zeros((num_embeddings, embedding_dim), dtype=np.float32))

            class LayerNorm(Module):
                def __init__(self, normalized_shape, eps=1e-5, *args, **kwargs):
                    super().__init__()
                    size = normalized_shape if isinstance(normalized_shape, int) else normalized_shape[-1]
                    self.weight = loader.array(np.ones((size,), dtype=np.float32))
                    self.bias = loader.array(np.zeros((size,), dtype=np.float32))
                    self.eps = eps

            class RMSNorm(Module):
                def __init__(self, normalized_shape, eps=1e-5, *args, **kwargs):
                    super().__init__()
                    size = normalized_shape if isinstance(normalized_shape, int) else normalized_shape[-1]
                    self.weight = loader.array(np.ones((size,), dtype=np.float32))
                    self.eps = eps

            class Dropout(Module):
                pass

            class Tanh(Module):
                pass

            m.Module = Module
            m.Linear = Linear
            m.Embedding = Embedding
            m.LayerNorm = LayerNorm
            m.RMSNorm = RMSNorm
            m.Dropout = Dropout
            m.Tanh = Tanh
            sys.modules[spec.name] = m
            return m

        if spec.name.startswith("mlx_lm.models.cache"):
            m = MockModule(spec.name)

            class _BaseCache:
                def __init__(self, *a, **k):
                    pass

                @property
                def meta_state(self):
                    return ""

                @meta_state.setter
                def meta_state(self, value):
                    pass

            class KVCache(_BaseCache):
                def __init__(self, *args, **kwargs):
                    self.keys = self.values = loader.array(np.zeros((1, 32, 0, 128)))
                    self.bits = 4.0
                    self.max_size = kwargs.get("max_size", 0)
                    self.keep = kwargs.get("keep", 0)
                    self.offset = kwargs.get("offset", 0)
                    self._idx = kwargs.get("idx", 0)

                def update_and_fetch(self, k, v):
                    self.keys, self.values = k, v
                    self.offset = k.shape[2] if hasattr(k, "shape") and len(k.shape) > 2 else self.offset
                    self._idx = self.keys.shape[2] if self.keys is not None else 0
                    return k, v

                @property
                def state(self):
                    return (self.keys, self.values)

                @state.setter
                def state(self, s):
                    self.keys, self.values = s[:2]

                def size(self):
                    return self.keys.shape[2] if self.keys is not None else 0

                def merge(self, caches):
                    """Merge singleton caches into a batched cache. Returns self for mock."""
                    return self

                def extend(self, *a, **k):
                    raise NotImplementedError(
                        f"{type(self).__name__}.extend requires batched conversion first"
                    )

                @classmethod
                def from_state(cls, state, meta_state=""):
                    inst = cls()
                    inst.state = state
                    if isinstance(meta_state, (list, tuple)) and meta_state:
                        try:
                            inst.offset = int(meta_state[0])
                        except Exception:
                            pass
                    return inst

            class RotatingKVCache(KVCache):
                def size(self):
                    if self.keys is None:
                        return 0
                    return min(int(self.offset), int(self.max_size or self.keys.shape[2]))

                def empty(self):
                    return self.keys is None

                def _temporal_order(self, value):
                    return value

                @property
                def meta_state(self):
                    return (self.keep, self.max_size, self.offset, self._idx)

                @meta_state.setter
                def meta_state(self, value):
                    if value and len(value) >= 4:
                        self.keep, self.max_size, self.offset, self._idx = map(int, value[:4])

            class BatchRotatingKVCache(RotatingKVCache):
                def __init__(self, max_size, left_padding):
                    super().__init__(max_size=max_size, keep=0)
                    self.left_padding = loader.array(left_padding)
                    self.offset = loader.array([0 for _ in left_padding]) if len(left_padding) > 1 else 0

                @classmethod
                def merge(cls, caches):
                    if not caches:
                        return cls(0, [])
                    non_empty = next((c for c in caches if getattr(c, "keys", None) is not None), None)
                    max_size = max(int(getattr(c, "max_size", 0)) for c in caches)
                    lengths = [c.size() if hasattr(c, "size") else 0 for c in caches]
                    for c, length in zip(caches, lengths):
                        if getattr(c, "keys", None) is not None and length > c.keys.shape[2]:
                            raise ValueError("oversized rotating cache buffer")
                    max_len = max(lengths) if lengths else 0
                    batch = cls(max_size, [max_len - l for l in lengths])
                    if non_empty is None:
                        batch.keys = batch.values = None
                        return batch
                    B, H, D = len(caches), non_empty.keys.shape[1], non_empty.keys.shape[3]
                    dtype = non_empty.keys._data.dtype
                    batch.keys = loader.array(np.zeros((B, H, max_len, D), dtype=dtype))
                    batch.values = loader.array(np.zeros((B, H, max_len, D), dtype=dtype))
                    offsets = []
                    for i, (c, length) in enumerate(zip(caches, lengths)):
                        pad = max_len - length
                        offsets.append(int(getattr(c, "offset", length) if not hasattr(getattr(c, "offset", None), "tolist") else loader.array(c.offset[i]).item() if hasattr(c.offset, "shape") else length))
                        if length > 0 and getattr(c, "keys", None) is not None:
                            ordered_k = c._temporal_order(c.keys) if hasattr(c, "_temporal_order") else c.keys
                            ordered_v = c._temporal_order(c.values) if hasattr(c, "_temporal_order") else c.values
                            batch.keys._data[i, :, pad:pad+length, :] = loader.array(ordered_k)._data[..., -length:, :]
                            batch.values._data[i, :, pad:pad+length, :] = loader.array(ordered_v)._data[..., -length:, :]
                    batch.left_padding = loader.array([max_len - l for l in lengths])
                    batch.offset = loader.array(offsets) if len(offsets) > 1 else offsets[0]
                    return batch

            class CacheList(list):
                def __init__(self, *args):
                    vals = args[0] if len(args) == 1 and isinstance(args[0], (list, tuple)) else args
                    super().__init__(vals)
                    self.caches = tuple(self)

                @classmethod
                def from_state(cls, sub_states, meta):
                    class_names, sub_meta_states = meta
                    cache_mod = sys.modules.get("mlx_lm.models.cache")
                    subs = []
                    for st, name, sub_meta in zip(sub_states, class_names, sub_meta_states):
                        cache_cls = getattr(cache_mod, name)
                        if name in ("KVCache", "RotatingKVCache"):
                            subs.append(cache_cls.from_state(st, sub_meta))
                        elif name == "ArraysCache":
                            c = cache_cls()
                            c.cache = list(st)
                            subs.append(c)
                        else:
                            c = cache_cls()
                            if hasattr(c, "state"):
                                c.state = tuple(st) if isinstance(st, (list, tuple)) else st
                            if sub_meta not in ("", None, ()): 
                                try:
                                    c.meta_state = sub_meta
                                except Exception:
                                    pass
                            subs.append(c)
                    return cls(*subs)

            class ArraysCache(KVCache):
                def __init__(self, *a, **k):
                    size = int(k.pop("size", 0) or 0)
                    super().__init__(*a, **k)
                    self.cache = [None] * size

                @property
                def state(self):
                    return tuple(self.cache)

                @state.setter
                def state(self, s):
                    self.cache = list(s) if s is not None else []

                def __setitem__(self, idx, value):
                    idx = int(idx)
                    while len(self.cache) <= idx:
                        self.cache.append(None)
                    self.cache[idx] = value

                def __getitem__(self, idx):
                    return self.cache[int(idx)]

                def extend(self, *a, **k):
                    raise NotImplementedError(
                        f"{type(self).__name__}.extend requires batched conversion first"
                    )

            class PoolingCache(_BaseCache):
                def __init__(self, ratio=1, *a, **k):
                    self.ratio = ratio
                    self.buf_kv = None
                    self.buf_gate = None
                    self.remainder = 0
                    self.pooled = None

                @property
                def offset(self):
                    return 0 if self.pooled is None else self.pooled.shape[1]

                @property
                def state(self):
                    return (self.buf_kv, self.buf_gate, self.pooled)

                @state.setter
                def state(self, s):
                    if s is None:
                        return
                    vals = list(s)
                    while len(vals) < 3:
                        vals.append(None)
                    self.buf_kv, self.buf_gate, self.pooled = vals[:3]

                def empty(self):
                    return self.offset == 0 and self.remainder == 0

                def size(self):
                    return self.offset

                @property
                def meta_state(self):
                    return self.ratio

                @meta_state.setter
                def meta_state(self, v):
                    self.ratio = int(v) if v not in (None, "") else 1

            class BatchPoolingCache(PoolingCache):
                def __init__(self, ratio=1, left_padding=None):
                    super().__init__(ratio)
                    left_padding = left_padding or [0]
                    self.left_padding = loader.array(left_padding)
                    self.remainder = [0] * len(left_padding)
                    self._pool_lengths = [0] * len(left_padding)
                    self._processed = [0] * len(left_padding)

                @property
                def meta_state(self):
                    return (self.ratio, self.remainder, self._pool_lengths, self._processed)

                @meta_state.setter
                def meta_state(self, v):
                    if v and len(v) >= 4:
                        self.ratio, self.remainder, self._pool_lengths, self._processed = v

            class BatchKVCache(KVCache):
                def __init__(self, left_padding):
                    super().__init__()
                    self.left_padding = loader.array(left_padding)
                    self.offset = loader.array([-l for l in left_padding]) if len(left_padding) > 1 else -left_padding[0]

                def make_mask(self, N, return_array=False, window_size=None):
                    cache_mod = sys.modules.get("mlx_lm.models.cache")
                    create_fn = getattr(cache_mod, "create_causal_mask")
                    offset = self.offset
                    phys = offset.max().item() if hasattr(offset, "max") else offset
                    return create_fn(N, offset=phys, window_size=window_size, left_padding=self.left_padding)

            for cls in (_BaseCache, KVCache, RotatingKVCache, BatchRotatingKVCache, CacheList, ArraysCache, PoolingCache, BatchPoolingCache, BatchKVCache):
                cls.__module__ = spec.name

            m._BaseCache = _BaseCache
            m.KVCache = KVCache
            m.RotatingKVCache = RotatingKVCache
            m.BatchRotatingKVCache = BatchRotatingKVCache
            m.ArraysCache = ArraysCache
            m.CacheList = CacheList
            m.PoolingCache = PoolingCache
            m.BatchPoolingCache = BatchPoolingCache
            m.BatchKVCache = BatchKVCache
            m.make_prompt_cache = lambda *a, **k: []
            m.create_attention_mask = lambda *a, **k: loader.array(np.zeros((1, 1, 1, 1)))
            m.create_causal_mask = lambda *a, **k: loader.array(np.zeros((1, 1, 1, 1)))
            m.dynamic_roll = lambda a, shift, axis=-2: loader.array(
                np.roll(loader.array(a)._data, shift, axis=axis)
            )
            sys.modules[spec.name] = m
            return m

        if spec.name in ("mlx_lm.models.base", "mlx_vlm.models.base"):
            m = MockModule(spec.name)
            m.scaled_dot_product_attention = lambda queries, *a, **k: loader.array(
                np.zeros(loader.array(queries).shape, dtype=loader.array(queries)._data.dtype)
            )
            sys.modules[spec.name] = m
            return m

        if spec.name == "mlx_lm.models.qwen3_5":
            m = MockModule(spec.name)
            TextModel_cls = type(
                "TextModel",
                (),
                {
                    "__init__": lambda self, *a, **k: None,
                    "__call__": lambda self, *a, **k: loader.array(np.zeros((1, 1))),
                    "norm": None,
                },
            )
            m.TextModel = TextModel_cls
            m.TextModelArgs = type(
                "TextModelArgs",
                (),
                {
                    "from_dict": classmethod(lambda cls, d: cls()),
                    "__init__": lambda self, *a, **k: None,
                },
            )
            m.Qwen3_5TextModel = type(
                "Qwen3_5TextModel",
                (),
                {
                    "__init__": lambda self, *a, **k: None,
                    "__call__": lambda self, *a, **k: loader.array(np.zeros((1, 1))),
                },
            )
            sys.modules[spec.name] = m
            return m

        if spec.name == "mlx_lm.utils":
            m = MockModule(spec.name)
            def _get_classes(config, *a, **k):
                model_type = config.get("model_type", "mock") if isinstance(config, dict) else getattr(config, "model_type", "mock")
                mod_name = f"mlx_lm.models.{model_type}"
                mod = sys.modules.get(mod_name)
                if mod is None:
                    mod = loader.create_module(importlib.machinery.ModuleSpec(mod_name, loader))
                model_cls = getattr(mod, "Model", None) or getattr(mod, "TextModel", None) or type("Model", (), {})
                args_cls = getattr(mod, "ModelArgs", None) or getattr(mod, "TextModelArgs", None) or type("Args", (), {})
                return model_cls, args_cls
            m._get_classes = _get_classes
            m.load_config = lambda *a, **k: {"model_type": "test"}
            sys.modules[spec.name] = m
            return m

        if spec.name == "mlx_vlm.utils":
            m = MockModule(spec.name)

            def _load_config(path, **kwargs):
                p = Path(path)
                cfg = p / "config.json"
                if cfg.exists():
                    return json.loads(cfg.read_text())
                return {"model_type": "test"}

            m.load_config = _load_config
            m.prepare_inputs = lambda *a, **k: {
                "input_ids": loader.array([[1]]),
                "pixel_values": None,
            }

            def _load_audio(path_or_bytes, sample_rate=16000):
                if isinstance(path_or_bytes, bytes):
                    return loader.array(np.zeros(16000, dtype=np.float32))
                if hasattr(path_or_bytes, "read"):
                    return loader.array(np.zeros(16000, dtype=np.float32))
                return loader.array(np.zeros(16000, dtype=np.float32))

            m.load_audio = _load_audio
            sys.modules[spec.name] = m
            return m

        if spec.name == "mlx_vlm.prompt_utils":
            m = MockModule(spec.name)
            m.extract_text_from_content = _extract_text_from_content
            m.get_message_json = _get_message_json
            m.apply_chat_template = lambda *a, **k: a[0] if a else []
            sys.modules[spec.name] = m
            return m

        if spec.name == "mlx.tokenizers._utils":
            m = MockModule(spec.name)

            def _is_spm_decoder(decoder):
                if isinstance(decoder, dict):
                    return decoder.get("type", "") == "Sentencepiece"
                decoder_type = getattr(decoder, "type", "")
                return decoder_type == "Sentencepiece"

            def _is_spm_decoder_no_space(decoder):
                return _is_spm_decoder(decoder)

            m._is_spm_decoder = _is_spm_decoder
            m._is_spm_decoder_no_space = _is_spm_decoder_no_space
            sys.modules[spec.name] = m
            return m

        if spec.name == "mlx_vlm.tool_parsers":
            m = MockModule(spec.name)
            m._infer_tool_parser = _infer_tool_parser
            m.load_tool_module = lambda *a, **k: _make_tool_module()
            sys.modules[spec.name] = m
            return m

        if spec.name == "mlx_lm.generate":
            m = MockModule(spec.name)

            class GenerationBatch:
                def __init__(self, *a, **k):
                    self.uids = [0]

                def prompt(self, *a, **k):
                    return None

                def _step(self, *a, **k):
                    pass

                def next(self, *a, **k):
                    return None

                def filter(self, *a, **k):
                    pass

                def extend(self, *a, **k):
                    pass

            class PromptProcessingBatch:
                def __init__(self, *a, **k):
                    self.uids = [0]
                    self.model = None
                    self.prompt_cache = []

                def prompt(self, *a, **k):
                    return None

                def _step(self, *a, **k):
                    pass

                def next(self, *a, **k):
                    return None

                def filter(self, *a, **k):
                    pass

                def split(self, *a, **k):
                    return self

                def extend(self, *a, **k):
                    pass

            class SequenceStateMachine:
                def __init__(self, transitions=None, initial="normal"):
                    self._initial = initial
                    self.state = initial
                    self._states = {}
                    for state_name, rules in (transitions or {}).items():
                        trie = {}
                        for seq, value in rules:
                            node = trie
                            for tok in seq:
                                node = node.setdefault(tok, {})
                            node["__match__"] = value
                        self._states[state_name] = [trie, None]

                def reset(self):
                    self.state = self._initial

                def __call__(self, token_id):
                    return self.state

            m.GenerationBatch = GenerationBatch
            m.PromptProcessingBatch = PromptProcessingBatch
            m.SequenceStateMachine = SequenceStateMachine
            m.BatchGenerator = type("BatchGenerator", (), {
                "__init__": lambda *a, **k: None,
                "_next": lambda *a, **k: None,
            })
            if not hasattr(loader, "_generation_stream"):
                loader._generation_stream = type("Stream", (), {})()
            m.generation_stream = loader._generation_stream
            sys.modules[spec.name] = m
            return m

        if spec.name == "mlx_lm.tokenizer_utils":
            m = MockModule(spec.name)

            class NaiveStreamingDetokenizer:
                def __init__(self, tokenizer):
                    self.tokenizer = tokenizer
                    self._tokens = []
                    self.last_subword_token = None
                    self.last_segment = ""
                    self.text = ""

                def add_token(self, token):
                    self._tokens.append(token)
                    self.last_subword_token = token

                def reset(self):
                    self._tokens = []
                    self.last_subword_token = None
                    self.last_segment = ""
                    self.text = ""

                def finalize(self):
                    result = "".join(chr(t) if t < 128 else "?" for t in self._tokens)
                    self.text = result
                    self.last_segment = result
                    return result

                def result(self):
                    return ""

                def decode(self, tokens):
                    return "mocked"

            class SPMStreamingDetokenizer(NaiveStreamingDetokenizer):
                pass

            class BPEStreamingDetokenizer(NaiveStreamingDetokenizer):
                pass

            def _is_spm_decoder(decoder):
                if isinstance(decoder, dict):
                    return decoder.get("type", "") == "Sentencepiece"
                decoder_type = getattr(decoder, "type", "")
                return decoder_type == "Sentencepiece"

            def _is_spm_decoder_no_space(decoder):
                return _is_spm_decoder(decoder)

            def _is_bpe_decoder(decoder):
                if isinstance(decoder, dict):
                    return decoder.get("type", "") in ("BPE", "ByteLevel")
                decoder_type = getattr(decoder, "type", "")
                return decoder_type in ("BPE", "ByteLevel")

            m.NaiveStreamingDetokenizer = NaiveStreamingDetokenizer
            m.SPMStreamingDetokenizer = SPMStreamingDetokenizer
            m.BPEStreamingDetokenizer = BPEStreamingDetokenizer
            m._is_spm_decoder = _is_spm_decoder
            m._is_spm_decoder_no_space = _is_spm_decoder_no_space
            m._is_bpe_decoder = _is_bpe_decoder
            sys.modules[spec.name] = m
            return m

        if spec.name == "mlx_vlm.turboquant":
            m = MockModule(spec.name)

            class TurboQuantMSEState:
                def __init__(self, norms, indices):
                    self.norms = norms
                    self.indices = indices

            class TurboQuantProdState:
                def __init__(self, norms, mse_indices, residual_norms, qjl_signs):
                    self.norms = norms
                    self.mse_indices = mse_indices
                    self.residual_norms = residual_norms
                    self.qjl_signs = qjl_signs

            class TurboQuantPolarState:
                def __init__(self, radii, level_indices):
                    self.radii = radii
                    self.level_indices = level_indices

            class TurboQuantPolarProdState:
                def __init__(self, norms, polar_state, residual_norms, qjl_signs):
                    self.norms = norms
                    self.polar_state = polar_state
                    self.residual_norms = residual_norms
                    self.qjl_signs = qjl_signs

            class TurboQuantSplitState:
                def __init__(self, low, high):
                    self.low = low
                    self.high = high

            class _QuantizedStateProxy:
                def __init__(self, state):
                    self._state = state

            def _state_length(state):
                if isinstance(state, TurboQuantMSEState):
                    return state.norms.shape[2]
                if isinstance(state, TurboQuantProdState):
                    return state.norms.shape[2]
                if isinstance(state, TurboQuantPolarState):
                    return state.radii.shape[2]
                if isinstance(state, TurboQuantPolarProdState):
                    return state.norms.shape[2]
                if isinstance(state, TurboQuantSplitState):
                    return _state_length(state.low)
                return 0

            def _packed_width(state, bits):
                dim = state.shape[-1]
                return max(1, math.ceil(dim * bits / 32))

            class _Codec:
                def __init__(self, head_dim, bits, mode="mse", seed=0):
                    self.head_dim = head_dim
                    self.bits = bits
                    self.mode = mode
                    self.seed = seed

                def quantize(self, arr):
                    arr = loader.array(arr)
                    packed = _packed_width(arr, max(int(math.ceil(self.bits)), 1))
                    dummy = loader.array(np.zeros(arr.shape[:-1] + (packed,), dtype=np.uint32))
                    if self.mode == "mse":
                        return TurboQuantMSEState(arr, dummy)
                    return TurboQuantProdState(arr, dummy, loader.array(np.zeros(arr.shape[:-1] + (1,))), loader.array(np.zeros(arr.shape[:-1] + (1,), dtype=np.uint32)))

                def dequantize(self, state):
                    if isinstance(state, TurboQuantMSEState):
                        return state.norms
                    if isinstance(state, TurboQuantProdState):
                        return state.norms
                    return loader.array(np.zeros((1, 1, 1, self.head_dim)))

            def _build_codec(arr, bits, mode="mse", seed=0):
                head_dim = loader.array(arr).shape[-1]
                return _Codec(head_dim, bits, mode=mode, seed=seed)

            def _slice_state(state, length):
                return _slice_state_range(state, 0, length)

            def _slice_state_range(state, start, end):
                if isinstance(state, TurboQuantMSEState):
                    return TurboQuantMSEState(
                        state.norms[..., start:end, :],
                        state.indices[..., start:end, :],
                    )
                if isinstance(state, TurboQuantProdState):
                    return TurboQuantProdState(
                        state.norms[..., start:end, :],
                        state.mse_indices[..., start:end, :],
                        state.residual_norms[..., start:end, :],
                        state.qjl_signs[..., start:end, :],
                    )
                if isinstance(state, TurboQuantPolarState):
                    return TurboQuantPolarState(
                        state.radii[..., start:end, :],
                        tuple(level[..., start:end, :] for level in state.level_indices),
                    )
                if isinstance(state, TurboQuantPolarProdState):
                    return TurboQuantPolarProdState(
                        state.norms[..., start:end, :],
                        _slice_state_range(state.polar_state, start, end),
                        state.residual_norms[..., start:end, :],
                        state.qjl_signs[..., start:end, :],
                    )
                if isinstance(state, TurboQuantSplitState):
                    return TurboQuantSplitState(
                        _slice_state_range(state.low, start, end),
                        _slice_state_range(state.high, start, end),
                    )
                return state

            def _concat_state(a, b):
                if isinstance(a, TurboQuantMSEState):
                    return TurboQuantMSEState(
                        loader.create_module(importlib.machinery.ModuleSpec("mlx.core", loader)).concatenate([a.norms, b.norms], axis=2),
                        loader.create_module(importlib.machinery.ModuleSpec("mlx.core", loader)).concatenate([a.indices, b.indices], axis=2),
                    )
                if isinstance(a, TurboQuantProdState):
                    mx_core = loader.create_module(importlib.machinery.ModuleSpec("mlx.core", loader))
                    return TurboQuantProdState(
                        mx_core.concatenate([a.norms, b.norms], axis=2),
                        mx_core.concatenate([a.mse_indices, b.mse_indices], axis=2),
                        mx_core.concatenate([a.residual_norms, b.residual_norms], axis=2),
                        mx_core.concatenate([a.qjl_signs, b.qjl_signs], axis=2),
                    )
                return b

            def _allocate_state_like(state, length):
                if isinstance(state, TurboQuantMSEState):
                    packed = state.indices.shape[-1]
                    return TurboQuantMSEState(
                        loader.array(np.zeros(state.norms.shape[:2] + (length, state.norms.shape[-1]))),
                        loader.array(np.zeros(state.indices.shape[:2] + (length, packed), dtype=np.uint32)),
                    )
                if isinstance(state, TurboQuantProdState):
                    packed = state.mse_indices.shape[-1]
                    shape = state.norms.shape[:2] + (length, state.norms.shape[-1])
                    return TurboQuantProdState(
                        loader.array(np.zeros(shape)),
                        loader.array(np.zeros(state.mse_indices.shape[:2] + (length, packed), dtype=np.uint32)),
                        loader.array(np.zeros(shape[:-1] + (1,))),
                        loader.array(np.zeros(shape[:-1] + (1,), dtype=np.uint32)),
                    )
                return state

            def _state_nbytes(state):
                if isinstance(state, TurboQuantMSEState):
                    return state.norms.nbytes + state.indices.nbytes
                if isinstance(state, TurboQuantProdState):
                    return state.norms.nbytes + state.mse_indices.nbytes
                return 0

            def _write_state(dst, start, src):
                return src

            def _reserve_state_capacity(state, capacity):
                return state

            def _validate_bits(bits):
                return bits

            def turboquant_enabled(bits):
                return not float(bits).is_integer()

            class TurboQuantKVCache:
                def __init__(self, bits=4.0, seed=0):
                    self.bits = bits
                    self.seed = seed
                    self.keys = None
                    self.values = None
                    self.offset = 0
                    self.key_codec = None
                    self.value_codec = None
                    self._cached_state = None
                    self._cached_state_offset = -1

                @property
                def nbytes(self):
                    return _state_nbytes(self.keys) + _state_nbytes(self.values)

                def _ensure_codecs(self, keys, values):
                    if self.key_codec is None:
                        key_bits = int(math.floor(self.bits) if not math.isclose(self.bits, round(self.bits), abs_tol=1e-6) else self.bits)
                        val_bits = int(math.ceil(self.bits) if not math.isclose(self.bits, round(self.bits), abs_tol=1e-6) else self.bits)
                        self.key_codec = _build_codec(keys, key_bits, mode="mse", seed=self.seed)
                        self.value_codec = _build_codec(values, val_bits, mode="mse", seed=self.seed + 1)

                def update_and_fetch(self, keys, values):
                    keys = loader.array(keys)
                    values = loader.array(values)
                    self._ensure_codecs(keys, values)
                    if self.keys is None:
                        fp_keys, fp_values = keys, values
                    else:
                        old_k, old_v = self.dequantize()
                        fp_keys = loader.create_module(importlib.machinery.ModuleSpec("mlx.core", loader)).concatenate([old_k, keys], axis=2)
                        fp_values = loader.create_module(importlib.machinery.ModuleSpec("mlx.core", loader)).concatenate([old_v, values], axis=2)
                    self.keys = self.key_codec.quantize(fp_keys)
                    self.values = self.value_codec.quantize(fp_values)
                    self.offset = fp_keys.shape[2]
                    return keys, values

                @classmethod
                def from_cache(cls, cache, bits=4.0, seed=0):
                    inst = cls(bits=bits, seed=seed)
                    if getattr(cache, "keys", None) is not None and getattr(cache, "values", None) is not None:
                        inst.update_and_fetch(cache.keys, cache.values)
                    return inst

                @property
                def state(self):
                    if self.keys is None:
                        return (None, None)
                    return (_slice_state(self.keys, self.offset), _slice_state(self.values, self.offset))

                @state.setter
                def state(self, value):
                    self.keys, self.values = value[:2]

                @property
                def meta_state(self):
                    return (str(self.bits), str(self.seed))

                @meta_state.setter
                def meta_state(self, value):
                    if value and len(value) >= 2:
                        self.bits = float(value[0])
                        self.seed = int(value[1])

                def dequantize(self, keys=None, values=None):
                    if keys is not None and values is not None:
                        return self.key_codec.dequantize(keys), self.value_codec.dequantize(values)
                    if self.keys is None:
                        empty = loader.array(np.zeros((1, 1, 0, 1)))
                        return empty, empty
                    return self.key_codec.dequantize(self.keys), self.value_codec.dequantize(self.values)

                def decode_attention(self, queries, **kwargs):
                    return loader.array(np.zeros(loader.array(queries).shape))

                def prefill_attention(self, queries, **kwargs):
                    return None

            m.TurboQuantKVCache = TurboQuantKVCache
            m.TurboQuantMSEState = TurboQuantMSEState
            m.TurboQuantProdState = TurboQuantProdState
            m.TurboQuantPolarState = TurboQuantPolarState
            m.TurboQuantPolarProdState = TurboQuantPolarProdState
            m.TurboQuantSplitState = TurboQuantSplitState
            m._build_codec = _build_codec
            m._concat_state = _concat_state
            m._slice_state = _slice_state
            m._slice_state_range = _slice_state_range
            m._state_length = _state_length
            m._state_nbytes = _state_nbytes
            m._allocate_state_like = _allocate_state_like
            m._write_state = _write_state
            m._reserve_state_capacity = _reserve_state_capacity
            m._QuantizedStateProxy = _QuantizedStateProxy
            m._validate_bits = _validate_bits
            m.turboquant_enabled = turboquant_enabled
            sys.modules[spec.name] = m
            return m

        if spec.name in ("dflash_mlx.engine.target_qwen_gdn", "dflash_mlx.engine.target_gemma4"):
            m = MockModule(spec.name)
            m._install_speculative_linear_cache_hook = lambda linear_attn: None
            m._install_split_full_attention_hook = lambda linear_attn: None
            m._install_full_attention_gqa_hook = lambda linear_attn: None
            sys.modules[spec.name] = m
            return m

        if spec.name == "dflash_mlx.runtime.config":
            m = MockModule(spec.name)
            from types import SimpleNamespace
            def _runtime_config_from_defaults(**kwargs):
                defaults = {"draft_window_size": 1024, "draft_sink_size": 64, "verify_mode": "adaptive"}
                defaults.update({k: v for k, v in kwargs.items() if v is not None})
                return SimpleNamespace(**defaults)
            m.runtime_config_from_defaults = _runtime_config_from_defaults
            sys.modules[spec.name] = m
            return m

        if spec.name == "dflash_mlx.runtime.context":
            m = MockModule(spec.name)
            from types import SimpleNamespace
            m.build_runtime_context = lambda runtime: SimpleNamespace(runtime=runtime)
            sys.modules[spec.name] = m
            return m

        if spec.name in ("mlx.core.random", "mlx.core.linalg", "mlx.core.distributed", "mlx.core.fast"):
            m = MockModule(spec.name)
            if spec.name == "mlx.core.random":
                m._rng = np.random.default_rng(0)
                m.state = loader.array([0], dtype="uint32")

                def _advance_state():
                    m.state = loader.array([(int(loader.array(m.state).item()) + 1) & 0xFFFFFFFF], dtype="uint32")

                def _resolve_shape(*a, **k):
                    if "shape" in k:
                        return k["shape"]
                    if len(a) >= 2 and all(isinstance(x, (int, float)) for x in a[:2]):
                        return ()
                    return a[0] if a else (1,)

                def _uniform(*a, **k):
                    low = a[0] if len(a) > 0 and isinstance(a[0], (int, float)) else 0.0
                    high = a[1] if len(a) > 1 and isinstance(a[1], (int, float)) else 1.0
                    out = loader.array(m._rng.uniform(low, high, size=_resolve_shape(*a, **k)))
                    _advance_state()
                    return out

                def _normal(*a, **k):
                    out = loader.array(m._rng.normal(size=_resolve_shape(*a, **k)))
                    _advance_state()
                    return out

                def _categorical(logits):
                    arr = loader.array(logits)._data
                    if arr.ndim == 1:
                        arr = arr.reshape(1, -1)
                    shifted = arr - np.max(arr, axis=-1, keepdims=True)
                    probs = np.exp(shifted)
                    probs = probs / np.sum(probs, axis=-1, keepdims=True)
                    samples = [m._rng.choice(arr.shape[-1], p=row) for row in probs]
                    _advance_state()
                    return loader.array(samples, dtype="uint32")

                def _seed(s):
                    seed = int(s)
                    m._rng = np.random.default_rng(seed)
                    m.state = loader.array([seed & 0xFFFFFFFF], dtype="uint32")

                m.uniform = _uniform
                m.normal = _normal
                m.categorical = _categorical
                m.seed = _seed
            if spec.name == "mlx.core.linalg":
                m.norm = lambda a, **k: loader.array(
                    np.linalg.norm(loader.array(a)._data, **k)
                )
            if spec.name == "mlx.core.fast":
                m.scaled_dot_product_attention = lambda queries, *a, **k: loader.array(
                    np.zeros(loader.array(queries).shape, dtype=loader.array(queries)._data.dtype)
                )
            sys.modules[spec.name] = m
            return m

        m = MockModule(spec.name)
        sys.modules[spec.name] = m
        return m

    def exec_module(self, module):
        if module.__name__ == "mlx.core":
            for x in [
                "float32",
                "float16",
                "bfloat16",
                "int8",
                "int32",
                "int64",
                "uint8",
                "uint16",
                "uint32",
                "bool_",
                "floating",
            ]:
                setattr(module, x, x)
            module.inf = float("inf")
            module.array = self.array
            module.eval = lambda *a: None


class MockMLXFinder(importlib.abc.MetaPathFinder):
    _dflash_mlx_checked = False
    _dflash_mlx_available = False

    def find_spec(self, fullname, path, target=None):
        # Only intercept dflash_mlx if it's actually importable.
        # On Linux/CI, dflash_mlx requires mlx (macOS-only), so the real
        # import should fail with ImportError and let tests skip gracefully.
        if fullname.startswith("dflash_mlx"):
            if not self._dflash_mlx_checked:
                self._dflash_mlx_checked = True
                try:
                    __import__("dflash_mlx")
                    self._dflash_mlx_available = True
                except ImportError:
                    self._dflash_mlx_available = False
            if not self._dflash_mlx_available:
                return None  # let real import machinery handle it

        if any(
            fullname == m or fullname.startswith(m + ".")
            for m in [
                "mlx",
                "mlx_lm",
                "mlx_vlm",
                "mlx_embeddings",
                "openai_harmony",
            ]
        ):
            return importlib.machinery.ModuleSpec(fullname, MockMLXLoader())
        return None


# =============================================================================
# openai_harmony mock — minimal encoding/parser shim for test suites
# =============================================================================

# Mapping of known Harmony special tokens → deterministic token IDs
_HARMONY_SPECIAL_TOKENS = {
    "<|start|>": 200000,
    "<|end|>": 200007,
    "<|channel|>": 200005,
    "<|message|>": 200008,
    "<|return|>": 200002,
    "<|call|>": 200012,
    "<|constrain|>": 200013,
    "assistant": 200001,
}


def _hash_token(text: str) -> int:
    """Deterministic token ID from text (avoids collisions with special tokens)."""
    return 30000 + (hash(text) % 10000)


class _MockHarmonyEncoding:
    """Minimal mock of openai_harmony.HarmonyEncoding."""

    def __init__(self, name: str = "HarmonyGptOss"):
        self.name = name

    def encode(self, text: str, allowed_special: str = "all") -> list[int]:
        """Encode text → list of token IDs.

        Splits on known special tokens and encodes remaining text in word
        chunks (max 3 tokens per chunk, deterministic via hash).
        """
        import re

        # Split on <|...|> tokens, then interleave with text chunks
        special_pattern = r"<\|[^>]+\|>"
        specials = re.findall(special_pattern, text)
        parts = re.split(special_pattern, text)

        tokens: list[int] = []
        for i, part in enumerate(parts):
            if part:  # non-empty text chunk
                words = part.split()
                for j in range(0, max(len(words), 1), 3):
                    chunk = " ".join(words[j : j + 3])
                    tokens.append(_hash_token(chunk))
            if i < len(specials):
                spec = specials[i]
                if spec in _HARMONY_SPECIAL_TOKENS:
                    tokens.append(_HARMONY_SPECIAL_TOKENS[spec])
                else:
                    tokens.append(_hash_token(spec))
        return tokens

    def decode(self, token_ids: list[int] | Any, **kwargs: Any) -> str:
        """Decode token IDs → text (reverse of encode, approximate)."""
        if hasattr(token_ids, "_data"):
            token_ids = token_ids._data.tolist()
        id_to_token = {v: k for k, v in _HARMONY_SPECIAL_TOKENS.items()}
        parts: list[str] = []
        for tid in token_ids:
            if tid in id_to_token:
                parts.append(id_to_token[tid])
            else:
                # Reverse hash → approximate text (not perfect, but good enough for tests)
                word_idx = tid - 30000
                parts.append(f"tok{word_idx}")
        return " ".join(parts)

    def stop_tokens_for_assistant_actions(self) -> list[int]:
        """Return stop token IDs for assistant actions."""
        return [200002, 200012]  # <|return|>, <|call|>


class _MockStreamableParser:
    """Minimal mock of openai_harmony.StreamableParser."""

    def __init__(self, encoding: Any, tokenizer: Any, strict: bool = False):
        self._encoding = encoding
        self._tokenizer = tokenizer
        self._strict = strict
        self._current_channel: str | None = None
        self._messages: list[dict[str, Any]] = []
        self._current_recipient: str | None = None
        self._in_analysis: bool = False

    @property
    def current_channel(self) -> str | None:
        """Return the current channel name."""
        return self._current_channel

    @property
    def messages(self) -> list[dict[str, Any]]:
        """Return accumulated messages."""
        return self._messages

    @property
    def current_recipient(self) -> str | None:
        """Return the current recipient."""
        return self._current_recipient

    def process(self, token: int) -> None:
        """Consume a token (no-op in mock)."""
        pass

    def process_eos(self) -> None:
        """Process end-of-stream (no-op in mock)."""
        pass

    def process_token(self, token: int) -> tuple[str, int | None, int | None, bool]:
        """Process a single token → (control_text, stream_token, visible_token, is_stop)."""
        id_to_token = {v: k for k, v in _HARMONY_SPECIAL_TOKENS.items()}
        token_str = id_to_token.get(token, "")

        control_text = ""
        stream_token: int | None = token
        visible_token: int | None = None
        is_stop = False

        if token_str == "<|channel|>":
            pass  # next token determines channel
        elif token_str == "final":
            self._current_channel = "final"
            self._in_analysis = False
        elif token_str == "analysis":
            self._current_channel = "analysis"
            self._in_analysis = True
        elif token_str == "commentary":
            self._current_channel = "commentary"
        elif token_str == "<|message|>":
            pass  # next tokens are content
        elif token_str == "<|return|>":
            is_stop = True
            stream_token = None
        elif token_str == "<|call|>":
            is_stop = True
            stream_token = None
        elif token_str == "<|end|>":
            self._current_channel = None
            self._in_analysis = False

        # Determine stream/visible based on channel
        if token_str not in _HARMONY_SPECIAL_TOKENS.values() or token_str == "<|message|>":
            # Check if this is a content token (not a special token)
            if token not in _HARMONY_SPECIAL_TOKENS.values():
                # Content token
                if self._current_channel == "final":
                    visible_token = token
                elif self._current_channel == "analysis":
                    # Analysis content streams but isn't visible
                    pass  # stream_token stays as token, visible stays None
                elif self._current_channel is None:
                    # Before any channel, still stream
                    pass

        return control_text, stream_token, visible_token, is_stop


class _MockRole:
    """Minimal mock of openai_harmony.Role enum."""
    system = "system"
    user = "user"
    assistant = "assistant"
    tool = "tool"


class _MockHarmonyMessage:
    """Minimal mock of openai_harmony.HarmonyMessage."""

    def __init__(self, role: str = "assistant", content: str = ""):
        self.role = role
        self.content = content


def _mock_load_harmony_encoding(name: str = "HarmonyGptOss") -> _MockHarmonyEncoding:
    """Load a mock Harmony encoding."""
    return _MockHarmonyEncoding(name)


def install_mock():
    import platform
    import sys

    if platform.system() != "Darwin":
        if not any(isinstance(f, MockMLXFinder) for f in sys.meta_path):
            sys.meta_path.insert(0, MockMLXFinder())
            for m in list(sys.modules.keys()):
                if any(
                    m == x or m.startswith(x + ".")
                    for x in [
                        "mlx",
                        "mlx_lm",
                        "mlx_vlm",
                        "mlx_embeddings",
                        "openai_harmony",
                        "dflash_mlx",
                    ]
                ):
                    del sys.modules[m]
