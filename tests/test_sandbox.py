"""Tests for the RLVR/TIR sandbox executor (lithos/posttrain/sandbox.py, E1a)."""

import sys

import pytest
from lithos.posttrain.sandbox import (
    DEFAULT_MEMORY_MB,
    MAX_OUTPUT_CHARS,
    octave_available,
    run_octave,
    run_python,
    run_tool,
)

pytestmark = pytest.mark.skipif(
    not sys.platform.startswith(("linux", "darwin")), reason="POSIX-only sandbox"
)


def test_runs_and_captures_stdout():
    r = run_python("print(6 * 7)")
    assert r.ok
    assert r.stdout.strip() == "42"
    assert r.returncode == 0
    assert not r.timed_out
    assert r.output.strip() == "42"


def test_setup_shares_namespace():
    r = run_python("print(area(2))", setup="import math\ndef area(r): return math.pi * r * r")
    assert r.ok
    assert r.stdout.startswith("12.56")


def test_error_is_not_ok_and_output_is_traceback():
    r = run_python("raise ValueError('boom')")
    assert not r.ok
    assert r.returncode != 0
    assert "ValueError" in r.stderr
    assert "boom" in r.output  # output falls back to stderr on failure


def test_timeout_kills_and_flags():
    r = run_python("while True:\n    pass", timeout_s=1.0)
    assert not r.ok
    assert r.timed_out
    assert r.returncode is None
    assert r.duration_s < 5.0  # actually stopped, didn't hang


def test_timeout_reaps_spawned_grandchild(tmp_path):
    # A gaming/buggy tool call could fork a process to outlive the timeout; the
    # process-group kill must reap it, not just SIGKILL the direct child.
    import os
    import signal
    import time

    marker = tmp_path / "gc.pid"
    code = f"""
import subprocess, sys, time
subprocess.Popen([sys.executable, "-c",
  "import os,time; open({str(marker)!r},'w').write(str(os.getpid())); time.sleep(30)"])
time.sleep(30)
"""
    r = run_python(code, timeout_s=2.0)
    assert r.timed_out
    time.sleep(1.0)  # let the marker land
    if not marker.exists():
        return  # grandchild never started; nothing to assert
    pid = int(marker.read_text())
    try:
        os.kill(pid, 0)  # exists?
        os.kill(pid, signal.SIGKILL)  # cleanup
        raise AssertionError(f"grandchild {pid} survived the timeout — process group not reaped")
    except ProcessLookupError:
        pass  # correctly reaped


def test_isolation_between_runs():
    run_python("x = 123")
    r = run_python("print(x)")  # x must not leak across processes
    assert not r.ok
    assert "NameError" in r.stderr


def test_output_is_truncated():
    r = run_python(f"print('a' * {MAX_OUTPUT_CHARS * 2})")
    assert r.ok
    assert len(r.stdout) <= MAX_OUTPUT_CHARS + 32
    assert "truncated" in r.stdout


def test_sympy_available_in_sandbox():
    r = run_python("import sympy; print(sympy.simplify('x**2 - x**2'))")
    assert r.ok
    assert r.stdout.strip() == "0"


def test_run_tool_dispatch():
    r = run_tool("python", "print('hi')")
    assert r.ok and r.stdout.strip() == "hi"
    with pytest.raises(ValueError, match="unknown runtime"):
        run_tool("ruby", "puts 1")


def test_octave_degrades_gracefully_when_absent():
    r = run_octave("disp(1 + 1)")
    if octave_available():
        assert r.ok
        assert "2" in r.stdout
    else:
        assert not r.ok
        assert "octave not installed" in r.stderr


def test_default_memory_is_generous_enough_for_numpy():
    # A too-tight RLIMIT_AS breaks scientific imports; the default must not.
    assert DEFAULT_MEMORY_MB >= 1024
    r = run_python("import numpy as np; print(int(np.arange(1000).sum()))")
    assert r.ok
    assert r.stdout.strip() == "499500"
