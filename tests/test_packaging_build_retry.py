# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the venvstacks build retry in packaging/build.py.

`venvstacks build` installs the locked layer requirements with `--find-links`
but no `--no-index`, and `_wheels/` only carries the locally built git-pinned
and sdist-only packages, so ~150 of the 155 wheels are pulled live from PyPI.
A connectivity blip there aborted the whole bundle build (observed as
`[Errno 9] Bad file descriptor` against pypi.org on the mlx-base layer) while
leaving the already-completed lock intact. These tests pin the retry behaviour:
transient index errors retry, real dependency errors fail fast.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

_BUILD_SCRIPT = Path(__file__).resolve().parents[1] / "packaging" / "build.py"


def _load_build_module():
    spec = importlib.util.spec_from_file_location("omlx_packaging_build", _BUILD_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def build_module():
    if not _BUILD_SCRIPT.is_file():
        pytest.skip("packaging/build.py not available (installed without source tree)")
    return _load_build_module()


_NETWORK_ERROR = (
    "WARNING: Retrying (Retry(total=0, connect=None, read=None, "
    "redirect=None, status=None)) after connection broken by "
    '"NewConnectionError(...): Failed to establish a new connection: '
    "[Errno 9] Bad file descriptor\"': /simple/addict/\n"
    "ERROR: No matching distribution found for addict==2.4.0\n"
)

_DEPENDENCY_ERROR = "ERROR: Cannot install mlx==0.32.3 and mlx-lm==0.31.3\n"

# venvstacks echoes the failing pip command, so the requirements path that
# names the layer is present in real install output.
_FAILED_LAYER_OUTPUT = (
    _NETWORK_ERROR
    + "CalledProcessError: Command '['pip', 'install', '-r', "
    "'/Users/u/Projects/omlx/packaging/requirements/framework-mlx-base/"
    "requirements-framework-mlx-base-macosx_arm64.txt']' "
    "returned non-zero exit status 1.\n"
)


def _write_flaky_stub(path: Path, fail_times: int) -> None:
    """Stub failing `fail_times` runs, echoing argv[2] as its error output."""
    path.write_text(
        "import pathlib, sys\n"
        "counter = pathlib.Path(sys.argv[1])\n"
        "attempts = int(counter.read_text()) if counter.exists() else 0\n"
        "counter.write_text(str(attempts + 1))\n"
        f"if attempts < {fail_times}:\n"
        "    sys.stdout.write(sys.argv[2])\n"
        "    sys.exit(1)\n"
        "sys.exit(0)\n",
        encoding="utf-8",
    )


@pytest.fixture
def stub(build_module, tmp_path, monkeypatch):
    counter = tmp_path / "attempts"
    script = tmp_path / "stub.py"

    def make(fail_times: int, error_text: str) -> tuple[list, Path]:
        _write_flaky_stub(script, fail_times)
        monkeypatch.setattr(build_module.time, "sleep", lambda _s: None)
        return [sys.executable, str(script), str(counter), error_text], counter

    return make


def test_transient_index_failure_is_retried(build_module, stub):
    cmd, counter = stub(1, _NETWORK_ERROR)

    build_module._build_with_network_retry(cmd)

    assert int(counter.read_text()) == 2, (
        "a transient index failure must retry the build once, then succeed"
    )


def test_non_network_failure_aborts_immediately(build_module, stub):
    cmd, counter = stub(5, _DEPENDENCY_ERROR)

    with pytest.raises(SystemExit):
        build_module._build_with_network_retry(cmd)

    assert int(counter.read_text()) == 1, (
        "a resolution/conflict failure is not connectivity and must not retry"
    )


def test_persistent_network_failure_gives_up(build_module, stub):
    cmd, counter = stub(99, _NETWORK_ERROR)

    with pytest.raises(SystemExit):
        build_module._build_with_network_retry(cmd, max_retries=3)

    assert int(counter.read_text()) == 3, (
        "retries must be bounded by max_retries, then exit non-zero"
    )


def test_purge_removes_only_the_failed_layer(build_module, tmp_path, monkeypatch):
    failed = tmp_path / "framework-mlx-base"
    completed = tmp_path / "cpython-3.11"
    failed.mkdir()
    completed.mkdir()
    (failed / "lib").mkdir()
    monkeypatch.setattr(build_module, "BUILD_DIR", tmp_path)

    build_module._purge_partial_layer(_FAILED_LAYER_OUTPUT)

    assert not failed.exists(), (
        "the half-installed layer must be gone so the retry reinstalls "
        "the full locked set instead of shipping a partial layer"
    )
    assert completed.exists(), (
        "the completed runtime layer must survive: wiping it re-downloads CPython"
    )


def test_purge_is_a_noop_when_no_layer_is_named(build_module, tmp_path, monkeypatch):
    kept = tmp_path / "framework-mlx-base"
    kept.mkdir()
    monkeypatch.setattr(build_module, "BUILD_DIR", tmp_path)

    build_module._purge_partial_layer("some unrelated failure output")

    assert kept.exists(), "an unidentifiable layer must not be wiped blindly"


def test_signatures_cover_the_observed_failure(build_module):
    assert any(sig in _NETWORK_ERROR for sig in build_module._TRANSIENT_NETWORK_SIGS), (
        "the pypi.org Errno 9 failure that motivated this retry must be matched"
    )
    assert not any(
        sig in _DEPENDENCY_ERROR for sig in build_module._TRANSIENT_NETWORK_SIGS
    ), "a resolver conflict must not look like connectivity"
