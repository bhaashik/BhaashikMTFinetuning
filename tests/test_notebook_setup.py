"""Setup must stop on unsupported runtimes and preserve the real install failure."""
import importlib.util
from pathlib import Path
import subprocess
import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('setup_notebook',ROOT/'scripts/setup_notebook.py')
setup=importlib.util.module_from_spec(spec);spec.loader.exec_module(setup)


def test_unsupported_python_stops_before_package_install(monkeypatch):
    monkeypatch.setattr(setup.sys,'version_info',(3,14,0))
    calls=[]
    monkeypatch.setattr(setup.subprocess,'run',lambda *a,**k:calls.append(a))
    with pytest.raises(RuntimeError,match='supported 3.10–3.13'):
        setup.setup(ROOT)
    assert not calls
    assert setup.validate_python((3,13,5))==(3,13)


def test_install_failure_is_not_retried_as_a_compiler_problem(monkeypatch):
    calls=[]
    def fail(command,**kwargs):
        calls.append(command)
        raise subprocess.CalledProcessError(17,command)
    monkeypatch.setattr(setup.subprocess,'run',fail)
    with pytest.raises(subprocess.CalledProcessError) as error:
        setup.setup(ROOT)
    assert error.value.returncode==17
    assert len(calls)==1
    assert '--only-binary=sentencepiece,tokenizers' in calls[0]
    assert not any('sudo' in command or 'apt-get' in command for command in calls)
