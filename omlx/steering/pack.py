# SPDX-License-Identifier: Apache-2.0
"""Steering pack: one unit vector per layer plus a manifest."""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

PACK_FORMAT = "omlx-steering-pack-v1"
DS4_FORMAT = "ds4-directional-steering-v1"
VECTOR_FILE = "direction.f32"
MANIFEST_FILE = "manifest.json"


class SteeringPackError(ValueError):
    """Pack is missing, corrupt or incompatible with the model."""


@dataclass
class SteeringPack:
    name: str
    vectors: np.ndarray  # (L, E) float32, unit-norm rows
    manifest: dict[str, Any] = field(default_factory=dict)
    path: Path | None = None

    @property
    def n_layers(self) -> int:
        return int(self.vectors.shape[0])

    @property
    def hidden_size(self) -> int:
        return int(self.vectors.shape[1])

    @property
    def sha256(self) -> str:
        return hashlib.sha256(
            np.ascontiguousarray(self.vectors, dtype="<f4").tobytes()
        ).hexdigest()

    @property
    def component(self) -> str:
        return (self.manifest.get("hook") or {}).get("component", "ffn")

    @property
    def mode(self) -> str:
        return (self.manifest.get("hook") or {}).get("mode", "block")

    @property
    def usage(self) -> dict[str, Any]:
        return self.manifest.get("usage") or {}

    @property
    def steer(self) -> dict[str, Any]:
        return self.manifest.get("steer") or {}


def get_steering_dir(base_path: Path | None = None) -> Path:
    if base_path is None:
        from ..settings import GlobalSettings

        base_path = Path(GlobalSettings.load().base_path)
    return Path(base_path) / "steering"


def model_shape_from_config(config: dict[str, Any]) -> dict[str, Any]:
    """model_type / hidden_size / n_layers from a model's config.json dict."""
    text = config.get("text_config") or {}
    return {
        "model_type": config.get("model_type") or text.get("model_type"),
        "hidden_size": text.get("hidden_size", config.get("hidden_size")),
        "n_layers": text.get("num_hidden_layers", config.get("num_hidden_layers")),
    }


def pack_fits(pack: SteeringPack, shape: dict[str, Any]) -> bool:
    """Whether ``pack`` can install on a model of ``shape`` (unknown fields pass)."""
    built_for = (pack.manifest.get("model") or {}).get("model_type")
    if shape.get("model_type") and built_for and built_for != shape["model_type"]:
        return False
    if shape.get("hidden_size") and pack.hidden_size != shape["hidden_size"]:
        return False
    return not (shape.get("n_layers") and pack.n_layers != shape["n_layers"])


def list_packs(
    base_path: Path | None = None,
) -> list[tuple[str, SteeringPack | None, str | None]]:
    """Every pack directory under the steering dir: (name, pack, error)."""
    sdir = get_steering_dir(base_path)
    if not sdir.is_dir():
        return []
    out: list[tuple[str, SteeringPack | None, str | None]] = []
    for entry in sorted(sdir.iterdir()):
        if not entry.is_dir():
            continue
        try:
            out.append((entry.name, load_pack(entry), None))
        except (SteeringPackError, OSError, ValueError) as exc:
            out.append((entry.name, None, str(exc)))
    return out


def _normalize_rows(v: np.ndarray) -> tuple[np.ndarray, bool]:
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    off = bool(np.any(np.abs(norms - 1.0) > 1e-3))
    if off:
        v = v / np.maximum(norms, 1e-12)
    return v.astype(np.float32, copy=False), off


def _read_f32(path: Path, shape: tuple[int, int]) -> np.ndarray:
    expected = shape[0] * shape[1] * 4
    size = path.stat().st_size
    if size != expected:
        raise SteeringPackError(
            f"{path}: size {size} != expected {expected} for shape {shape}"
        )
    return np.fromfile(path, dtype="<f4").reshape(shape).astype(np.float32)


def load_pack(path: Path | str) -> SteeringPack:
    """Load and validate a pack directory (the checks that need no model)."""
    root = Path(path)
    mpath = root / MANIFEST_FILE
    if not mpath.is_file():
        raise SteeringPackError(f"{root}: no {MANIFEST_FILE}")
    manifest = json.loads(mpath.read_text())
    fmt = str(manifest.get("format", ""))
    if not fmt.startswith("omlx-steering-pack-v1"):
        raise SteeringPackError(f"{root}: unknown pack format {fmt!r}")
    vec = manifest.get("vector") or {}
    shape = tuple(vec.get("shape") or ())
    if len(shape) != 2:
        raise SteeringPackError(f"{root}: manifest vector.shape must be [L, E]")
    vpath = root / vec.get("file", VECTOR_FILE)
    if not vpath.is_file():
        raise SteeringPackError(f"{vpath}: missing")
    v = _read_f32(vpath, (int(shape[0]), int(shape[1])))
    want = vec.get("sha256")
    if want:
        got = hashlib.sha256(vpath.read_bytes()).hexdigest()
        if got != want:
            raise SteeringPackError(f"{vpath}: sha256 mismatch (corrupt or tampered)")
    if not np.all(np.isfinite(v)):
        raise SteeringPackError(f"{vpath}: contains non-finite values")
    v, renorm = _normalize_rows(v)
    if renorm:
        logger.warning("steering pack %s: rows were not unit-norm, renormalised", root)
    return SteeringPack(
        name=str(manifest.get("name") or root.name),
        vectors=v,
        manifest=manifest,
        path=root,
    )


def save_pack(pack: SteeringPack, dest: Path | str) -> Path:
    """Write ``direction.f32`` + ``manifest.json`` (sha256 and shape refreshed)."""
    root = Path(dest)
    root.mkdir(parents=True, exist_ok=True)
    data = np.ascontiguousarray(pack.vectors, dtype="<f4").tobytes()
    (root / VECTOR_FILE).write_bytes(data)
    m = dict(pack.manifest)
    m["format"] = PACK_FORMAT
    m["ds4_format"] = DS4_FORMAT
    m["name"] = pack.name
    m["vector"] = {
        "file": VECTOR_FILE,
        "shape": [pack.n_layers, pack.hidden_size],
        "dtype": "f32",
        "sha256": hashlib.sha256(data).hexdigest(),
    }
    (root / MANIFEST_FILE).write_text(json.dumps(m, indent=2))
    return root


def resolve_pack_ref(
    ref: str, *, base_path: Path | None = None, shape: tuple[int, int] | None = None
) -> SteeringPack:
    """Resolve a ``steering_pack`` setting: name, dir, or bare .f32."""
    named = get_steering_dir(base_path) / ref
    if named.is_dir():
        return load_pack(named)
    p = Path(ref).expanduser()
    if p.is_dir():
        return load_pack(p)
    if p.is_file() and p.suffix == ".f32":
        if shape is None:
            raise SteeringPackError(f"{p}: bare .f32 needs the model shape")
        v, _ = _normalize_rows(_read_f32(p, shape))
        return SteeringPack(
            name=p.stem,
            vectors=v,
            manifest={
                "hook": {"component": "ffn", "mode": "block"},
                "vector": {"shape": list(shape)},
            },
            path=p,
        )
    raise SteeringPackError(f"steering pack {ref!r} not found")
