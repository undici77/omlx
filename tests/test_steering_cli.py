# SPDX-License-Identifier: Apache-2.0
"""Fast tests for the `omlx steer` CLI."""

import json
import shutil
import subprocess
import sys
import zipfile

import numpy as np
import pytest

from omlx.steering.pack import SteeringPack, load_pack, save_pack

ACTIONS = ("build", "sweep", "verify", "set", "show", "list", "import", "export")


def test_steer_help_is_exposed():
    result = subprocess.run(
        [sys.executable, "-m", "omlx.cli", "steer", "--help"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0
    for action in ACTIONS:
        assert action in result.stdout


@pytest.mark.parametrize("action", ACTIONS)
def test_steer_action_help(action):
    result = subprocess.run(
        [sys.executable, "-m", "omlx.cli", "steer", action, "--help"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


def _run_main(argv, monkeypatch, capsys):
    from omlx.cli import main

    monkeypatch.setattr(sys, "argv", ["omlx", *argv])
    with pytest.raises(SystemExit) as excinfo:
        main()
    captured = capsys.readouterr()
    return excinfo.value.code, captured.out, captured.err


@pytest.fixture()
def pack_dir(tmp_path, monkeypatch):
    """Point get_steering_dir() at a temp base and return the dir."""
    from omlx.steering import pack as pack_mod

    sdir = tmp_path / "base" / "steering"
    sdir.mkdir(parents=True)
    monkeypatch.setattr(pack_mod, "get_steering_dir", lambda base_path=None: sdir)
    return sdir


def _make_pack(sdir, name="terse", n_layers=4, hidden=8, model_type="qwen3_next"):
    vectors = np.zeros((n_layers, hidden), dtype=np.float32)
    vectors[:, 0] = 1.0
    manifest = {
        "description": "test pack",
        "model": {
            "model_type": model_type,
            "hidden_size": hidden,
            "n_layers": n_layers,
        },
        "hook": {"component": "ffn", "mode": "block"},
        "usage": {"scope": "prefill+k", "decode_tokens": 32},
    }
    return save_pack(SteeringPack(name, vectors, manifest), sdir / name)


def _make_model_dir(tmp_path, model_type="qwen3_next", hidden=8, layers=4):
    model_dir = tmp_path / "model"
    model_dir.mkdir(exist_ok=True)
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "model_type": model_type,
                "hidden_size": hidden,
                "num_hidden_layers": layers,
            }
        )
    )
    return model_dir


def test_steer_read_prompts_skips_comments_and_blanks(tmp_path):
    from omlx.cli import _steer_read_prompts

    f = tmp_path / "p.txt"
    f.write_text("# comment\n\nprompt one\n  prompt two  \n")
    assert _steer_read_prompts(str(f)) == ["prompt one", "prompt two"]


def test_steer_set_without_flags_fails_before_contacting_server(
    pack_dir, monkeypatch, capsys
):
    code, out, err = _run_main(["steer", "set", "m1"], monkeypatch, capsys)
    assert code == 1
    assert "Nothing to set" in err


def test_steer_list_marks_compatibility(pack_dir, tmp_path, monkeypatch, capsys):
    _make_pack(pack_dir)
    model_dir = _make_model_dir(tmp_path)
    code, out, err = _run_main(
        ["steer", "list", "--model", str(model_dir)], monkeypatch, capsys
    )
    assert code == 0, err
    assert "terse" in out
    assert "[compatible]" in out


def test_steer_list_marks_incompatible(pack_dir, tmp_path, monkeypatch, capsys):
    _make_pack(pack_dir)
    model_dir = _make_model_dir(tmp_path, model_type="llama")
    code, out, err = _run_main(
        ["steer", "list", "--model", str(model_dir)], monkeypatch, capsys
    )
    assert code == 0, err
    assert "[incompatible]" in out


def test_steer_list_without_model_dir(pack_dir, monkeypatch, capsys):
    _make_pack(pack_dir)
    code, out, err = _run_main(["steer", "list"], monkeypatch, capsys)
    assert code == 0, err
    assert "terse" in out
    assert "compatible" not in out


def test_steer_list_missing_dir_reports_empty(pack_dir, monkeypatch, capsys):
    shutil.rmtree(pack_dir)
    code, out, err = _run_main(["steer", "list"], monkeypatch, capsys)
    assert code == 0
    assert "No steering directory" in out


def test_steer_export_import_round_trip(pack_dir, tmp_path, monkeypatch, capsys):
    _make_pack(pack_dir, name="round")
    zpath = tmp_path / "round.zip"
    code, out, err = _run_main(
        ["steer", "export", "round", "-o", str(zpath)], monkeypatch, capsys
    )
    assert code == 0, err
    assert zpath.is_file()
    shutil.rmtree(pack_dir / "round")
    code, out, err = _run_main(["steer", "import", str(zpath)], monkeypatch, capsys)
    assert code == 0, err
    assert (pack_dir / "round" / "manifest.json").is_file()
    assert load_pack(pack_dir / "round").name == "round"


def test_steer_import_bare_f32_wraps_ds4(pack_dir, tmp_path, monkeypatch, capsys):
    model_dir = _make_model_dir(tmp_path)
    f32 = tmp_path / "ds4-verb.f32"
    np.arange(4 * 8, dtype="<f4").tofile(f32)
    manifest = tmp_path / "base.json"
    manifest.write_text(json.dumps({"description": "ds4 verbosity"}))
    code, out, err = _run_main(
        [
            "steer",
            "import",
            str(f32),
            "--json",
            str(manifest),
            "--model",
            str(model_dir),
            "--name",
            "ds4-verb",
        ],
        monkeypatch,
        capsys,
    )
    assert code == 0, err
    pack = load_pack(pack_dir / "ds4-verb")
    assert pack.n_layers == 4 and pack.hidden_size == 8
    assert pack.component == "ffn" and pack.mode == "block"


def test_steer_import_bare_f32_requires_json_and_model(
    pack_dir, tmp_path, monkeypatch, capsys
):
    f32 = tmp_path / "x.f32"
    f32.write_bytes(b"\0" * 128)
    code, out, err = _run_main(["steer", "import", str(f32)], monkeypatch, capsys)
    assert code == 1
    assert "--json" in err


def test_steer_import_refuses_zip_traversal(pack_dir, tmp_path, monkeypatch, capsys):
    zpath = tmp_path / "evil.zip"
    with zipfile.ZipFile(zpath, "w") as zf:
        zf.writestr("../evil.txt", "x")
    code, out, err = _run_main(["steer", "import", str(zpath)], monkeypatch, capsys)
    assert code == 1
    assert "unsafe zip member" in err
    assert not (pack_dir.parent / "evil.txt").exists()


def test_steer_import_rejects_corrupt_pack(pack_dir, tmp_path, monkeypatch, capsys):
    _make_pack(pack_dir, name="broken")
    manifest_path = pack_dir / "broken" / "manifest.json"
    # Tamper with the vector in place so the sha256 check refuses it.
    vec = pack_dir / "broken" / "direction.f32"
    data = bytearray(vec.read_bytes())
    data[0] ^= 0xFF
    vec.write_bytes(bytes(data))
    zpath = tmp_path / "broken.zip"
    with zipfile.ZipFile(zpath, "w") as zf:
        zf.write(manifest_path, "manifest.json")
        zf.write(vec, "direction.f32")
    code, out, err = _run_main(["steer", "import", str(zpath)], monkeypatch, capsys)
    assert code == 1
    assert "Import failed" in err


def test_steering_data_is_declared_package_data():
    """verify, --preset and sweep read omlx/steering/data at runtime; a wheel
    without the package-data entry would drop every file there."""
    import tomllib
    from fnmatch import fnmatch
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    pyproject = root / "pyproject.toml"
    if not pyproject.is_file():
        pytest.skip("pyproject.toml not available")
    globs = tomllib.loads(pyproject.read_text())["tool"]["setuptools"][
        "package-data"
    ]["omlx.steering"]
    data = root / "omlx" / "steering" / "data"
    for needed in (
        "ds4/succinct.txt",
        "ds4/verbose.txt",
        "ds4/eval_prompts.txt",
        "quality/gen_prompts.txt",
        "neutral/para_001_050.txt",
    ):
        assert (data / needed).is_file(), needed
        assert any(fnmatch(f"data/{needed}", g) for g in globs), needed


def test_build_preset_uses_bundled_prompts(monkeypatch):
    from omlx import cli

    good, bad = cli._STEER_PRESETS["verbosity"]
    assert len(cli._steer_read_prompts(str(cli._steer_data_dir() / good))) == 100
    assert len(cli._steer_read_prompts(str(cli._steer_data_dir() / bad))) == 100
