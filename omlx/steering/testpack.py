# SPDX-License-Identifier: Apache-2.0
"""Random-direction pack for plumbing tests (NOT a meaningful steering vector).

    python -m omlx.steering.testpack MODEL_DIR NAME [--component ffn|attn] [--seed N]

Writes ``<base>/steering/NAME``. The direction is random, so it proves hooks,
scopes, cache salt and coverage work on the real model; it will not steer a
behaviour. Real vectors come from `omlx steer build` or a DS4 ``.f32``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .pack import SteeringPack, get_steering_dir, save_pack


def make_test_pack(model_dir: Path, name: str, component="ffn", seed=0, dest=None):
    cfg = json.loads((Path(model_dir) / "config.json").read_text())
    cfg = cfg.get("text_config", cfg)
    layers, hidden = int(cfg["num_hidden_layers"]), int(cfg["hidden_size"])
    v = np.random.default_rng(seed).standard_normal((layers, hidden)).astype(np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    # No hook mode: the pack then installs on whichever adapter the target
    # model resolves to (a qwen4_exp target refuses mode-tagged block packs,
    # and the shape checks below catch cross-model use).
    manifest = {
        "description": "random test direction (plumbing only)",
        "model": {"model_type": cfg.get("model_type"), "hidden_size": hidden, "n_layers": layers},
        "hook": {"component": component},
        "usage": {"scope": "prefill+k", "decode_tokens": 32},
    }
    return save_pack(SteeringPack(name, v, manifest), dest or get_steering_dir() / name)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("name")
    ap.add_argument("--component", default="ffn", choices=["ffn", "attn"])
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    print(make_test_pack(Path(a.model_dir), a.name, a.component, a.seed))


if __name__ == "__main__":
    main()
