"""Dependency-free setup tests: no packages are installed or models downloaded."""
import importlib.util
import os
from pathlib import Path
from unittest.mock import patch

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "setup_notebook.py"
spec = importlib.util.spec_from_file_location("notebook_setup_under_test", SCRIPT)
setup_script = importlib.util.module_from_spec(spec)
spec.loader.exec_module(setup_script)


@pytest.mark.parametrize("version", [(3, 10), (3, 11), (3, 12), (3, 13)])
def test_python_supported(version):
    assert setup_script.validate_python(version) == version


@pytest.mark.parametrize("version", [(3, 9), (3, 14)])
def test_python_rejected(version):
    with pytest.raises(RuntimeError):
        setup_script.validate_python(version)


def test_disable_unused_frameworks():
    with patch.dict(os.environ, {"USE_TF": "1", "USE_FLAX": "1"}):
        setup_script.configure_backends()
        assert os.environ["USE_TF"] == os.environ["USE_FLAX"] == "0"


@pytest.mark.parametrize("mode,expected", [
    ("base", "."), ("preflight", ".[preflight]"),
    ("train", ".[train]"), ("train,repairs", ".[train,repairs]")
])
def test_install_mode_does_not_force_protobuf(tmp_path, mode, expected):
    (tmp_path / "pyproject.toml").write_text("[project]\nname='placeholder'\n")
    with patch.object(setup_script.subprocess, "run") as run:
        setup_script.setup(tmp_path, extras=mode)
    assert run.call_count == 2
    install, smoke = (call.args[0] for call in run.call_args_list)
    assert install[-2:] == ["-e", expected]
    assert "protobuf" not in " ".join(install).lower()
    assert smoke[-3:] == ["--verify-only", "--extras", mode]
    assert smoke[0] == setup_script.sys.executable


def test_cuda_check_only_for_training(tmp_path):
    (tmp_path / "pyproject.toml").write_text("[project]\nname='placeholder'\n")
    with pytest.raises(ValueError):
        setup_script.setup(tmp_path, extras="base", require_cuda=True)
    with patch.object(setup_script.subprocess, "run") as run:
        setup_script.setup(tmp_path, extras="train", require_cuda=True)
    assert run.call_count == 2
    assert run.call_args_list[-1].args[0][-1] == "--require-cuda"


def test_verify_only_mode_skips_installation():
    with patch.object(setup_script, "verify_stack") as verify, \
         patch.object(setup_script.subprocess, "run") as run, \
         patch.object(setup_script.sys, "argv", ["setup_notebook.py", "--verify-only", "--extras", "base"]):
        assert setup_script.main() == 0
    run.assert_not_called()
    assert verify.call_args.kwargs == {"require_cuda": False, "extras": "base"}
