"""Checks for the shell scripts in scripts/. Run: python -m pytest tests -q
The library test needs the envs and nccl-tests build from scripts/01_setup.sh; it is skipped (with a reason) if they
are missing, so the syntax checks still run on any machine."""
import os
import pathlib
import subprocess

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPTS = sorted((ROOT / "scripts").glob("*.sh"))


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_bash_syntax(script):
    r = subprocess.run(["bash", "-n", str(script)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def _nccl_env():
    r = subprocess.run(["bash", str(ROOT / "scripts/02_nccl.sh"), "--print-env"], capture_output=True, text=True,
                       cwd=ROOT)
    if r.returncode != 0:
        pytest.skip(f"02_nccl.sh --print-env failed (envs not set up?): {r.stderr.strip()[-200:]}")
    env = dict(line.split("=", 1) for line in r.stdout.strip().splitlines())
    if not os.path.exists(env["BIN"]):
        pytest.skip(f"nccl-tests not built at {env['BIN']}")
    return env


def test_nccl_binary_resolves_all_libraries():
    """The environment 02_nccl.sh exports must let all_reduce_perf load every shared library
    (the first M1 run failed with 'libcudart.so.13: cannot open shared object file')."""
    env = _nccl_env()
    r = subprocess.run(["ldd", env["BIN"]], capture_output=True, text=True,
                       env={**os.environ, "LD_LIBRARY_PATH": env["LD_LIBRARY_PATH"]})
    missing = [l.strip() for l in r.stdout.splitlines() if "not found" in l]
    assert not missing, f"unresolved libraries: {missing}"
    assert "libnccl.so" in r.stdout and "libcudart.so.13" in r.stdout


BENCH_FAILED = "Successful requests:                     0         \nFailed requests:                         128       \n"
BENCH_OK = "Successful requests:                     128       \nFailed requests:                         0         \n"
BENCH_OK_NO_FAILED_LINE = "Successful requests:                     128       \n"
BENCH_ZERO_NO_FAILED_LINE = "Successful requests:                     0         \n"
BENCH_PARTIAL = "Successful requests:                     120       \nFailed requests:                         8         \n"


@pytest.mark.parametrize("text,ok", [(BENCH_FAILED, False), (BENCH_OK, True), (BENCH_OK_NO_FAILED_LINE, True),
                                     (BENCH_PARTIAL, False), (BENCH_ZERO_NO_FAILED_LINE, False),
                                     ("Traceback: server died\n", False)],
                         ids=["all-failed", "ok", "ok-no-failed-line", "partial", "zero-no-failed-line", "no-summary"])
def test_sweep_bench_ok(tmp_path, text, ok):
    """03_sweep_vllm.sh must not count a benchmark with failed requests as a result (M2 run 2: 128 x 500 errors
    were reported as 0.00 tok/s)."""
    log = tmp_path / "bench.log"
    log.write_text(text)
    r = subprocess.run(["bash", str(ROOT / "scripts/03_sweep_vllm.sh"), "--check-bench-log", str(log)])
    assert (r.returncode == 0) == ok
