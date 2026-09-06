import shlex
import sys
import time
from pathlib import Path

import pytest

import tiny_tensor_compiler.native as native_module
from tiny_tensor_compiler import NativeCompilationTimeout, compile_native, lower_to_cpu, lower_to_loops
from tiny_tensor_compiler.frontend import GraphBuilder


def _loop_program():
    builder = GraphBuilder()
    value = builder.tensor([1, -2, 3], dtype="int32")
    return lower_to_loops(lower_to_cpu(builder.finish(value.relu())))


def _python_compiler_command() -> str:
    return shlex.join([sys.executable])


def test_compiler_timeout_terminates_spawned_child_process_tree(tmp_path, monkeypatch):
    started_marker = tmp_path / "child-started.txt"
    survived_marker = tmp_path / "child-survived.txt"
    child_code = (
        "import pathlib,sys,time; "
        "pathlib.Path(sys.argv[1]).write_text('started', encoding='utf-8'); "
        "time.sleep(0.8); "
        "pathlib.Path(sys.argv[2]).write_text('survived', encoding='utf-8')"
    )
    parent_code = (
        "import pathlib,subprocess,sys,time; "
        "started=pathlib.Path(sys.argv[1]); "
        "subprocess.Popen([sys.executable,'-c',sys.argv[3],sys.argv[1],sys.argv[2]]); "
        "deadline=time.monotonic()+2.0; "
        "\nwhile not started.exists() and time.monotonic() < deadline: time.sleep(0.005); "
        "\ntime.sleep(10)"
    )
    command = [
        sys.executable,
        "-c",
        parent_code,
        str(started_marker),
        str(survived_marker),
        child_code,
    ]
    monkeypatch.setattr(native_module, "_build_compile_command", lambda *args: command)

    with pytest.raises(NativeCompilationTimeout):
        compile_native(
            _loop_program(),
            compiler=_python_compiler_command(),
            compiler_timeout=0.35,
        )

    assert started_marker.is_file(), "the descendant must start before timeout cancellation"
    time.sleep(1.0)
    assert not survived_marker.exists(), "a timed-out compiler descendant must not outlive the tree"
