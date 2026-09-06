import shlex
import sys
import time
from pathlib import Path

import pytest

import tiny_tensor_compiler.native as native_module
from tiny_tensor_compiler import (
    GraphBuilder,
    NativeCompilationTimeout,
    clear_native_cache,
    compile_native,
    lower_to_cpu,
    lower_to_loops,
)


@pytest.fixture(autouse=True)
def _clear_native_artifact_cache():
    clear_native_cache()
    yield
    clear_native_cache()


def _loop_program():
    builder = GraphBuilder()
    value = builder.tensor([1, -2, 3], dtype="int32")
    return lower_to_loops(lower_to_cpu(builder.finish(value.relu())))


def _python_compiler_command() -> str:
    return shlex.join([sys.executable])


def test_compiler_timeout_terminates_spawned_descendant(tmp_path, monkeypatch):
    spawned_marker = tmp_path / "descendant-spawned.txt"
    survived_marker = tmp_path / "descendant-survived.txt"
    child_code = (
        "import pathlib,time;"
        "time.sleep(1.0);"
        f"pathlib.Path({str(survived_marker)!r}).write_text('survived', encoding='utf-8')"
    )
    parent_code = (
        "import pathlib,subprocess,sys,time;"
        f"subprocess.Popen([sys.executable, '-c', {child_code!r}]);"
        f"pathlib.Path({str(spawned_marker)!r}).write_text('spawned', encoding='utf-8');"
        "time.sleep(10)"
    )
    command = [sys.executable, "-c", parent_code]
    monkeypatch.setattr(native_module, "_build_compile_command", lambda *args: command)

    with pytest.raises(NativeCompilationTimeout) as caught:
        compile_native(
            _loop_program(),
            compiler=_python_compiler_command(),
            compiler_timeout=0.4,
        )

    assert caught.value.command == tuple(command)
    assert spawned_marker.read_text(encoding="utf-8") == "spawned"
    time.sleep(1.2)
    assert not survived_marker.exists()
