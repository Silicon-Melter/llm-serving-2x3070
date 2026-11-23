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


# Real lines from results/vllm/m3-581eb68/8b-awq_tp2.try-auto.log (TP=2: each rank reports its own sizes)
KV_RANK1 = ("Replace gpu_memory_utilization config with `--kv-cache-memory=4557176832` (4.24 GiB) to fit into requested "
            "memory, or `--kv-cache-memory=4185915392` (3.9 GiB) to fully utilize gpu memory.\n"
            "Memory after warmup: 4.06 GiB free; 512 KV cache blocks (2.25 GiB) fit within the requested 7.66 GiB.\n")
KV_RANK0 = ("Replace gpu_memory_utilization config with `--kv-cache-memory=4548263936` (4.24 GiB) to fit into requested "
            "memory, or `--kv-cache-memory=3607560192` (3.36 GiB) to fully utilize gpu memory.\n"
            "Memory after warmup: 3.52 GiB free; 384 KV cache blocks (1.69 GiB) fit within the requested 7.65 GiB.\n")


@pytest.mark.parametrize("text,expected", [
    (KV_RANK0 + KV_RANK1, "3607560192 1814623682"),
    (KV_RANK1 + KV_RANK0, "3607560192 1814623682"),   # rank order in the log must not matter
    ("Traceback: no memory report\n", " 0"),
], ids=["rank0-first", "rank1-first", "no-report"])
def test_sweep_kv_parse_takes_smallest_rank(tmp_path, text, expected):
    """The KV rule must size from the tightest GPU: GPU0 also drives the display, so its numbers are smaller.
    Taking the last line in the log (as 581eb68 did for the automatic size) depends on rank print order."""
    log = tmp_path / "server.log"
    log.write_text(text)
    r = subprocess.run(["bash", str(ROOT / "scripts/03_sweep_vllm.sh"), "--parse-kv-log", str(log)],
                       capture_output=True, text=True)
    assert r.returncode == 0 and r.stdout.strip("\n") == expected, r.stdout + r.stderr


def _stage_done(d, stage):
    return subprocess.run(["bash", str(ROOT / "scripts/06_quality.sh"), "--stage-done", str(d), stage]).returncode == 0


def test_quality_stage_done_rejects_files_cut_off_by_a_reset(tmp_path):
    """After the 2026-11-23 reset, lm-eval's sample files were 0 bytes: such a stage must be redone, not skipped."""
    import gzip
    d = tmp_path / "cfg"; lm = d / "lm_eval" / "m"; lm.mkdir(parents=True)
    assert not any(_stage_done(d, s) for s in ("kld", "gsm8k", "mmlu"))            # nothing there yet
    (lm / "samples_gsm8k_x.jsonl").write_text("")                                  # 0 bytes, as after the reset
    assert not _stage_done(d, "gsm8k")
    (lm / "samples_gsm8k_x.jsonl").write_text('{"doc_id": 0}\n')
    assert _stage_done(d, "gsm8k")
    good = gzip.compress(b'{"i": 0}\n' * 2000)
    (d / "logprobs.jsonl.gz").write_bytes(good[: len(good) // 2])                  # cut off mid-write
    assert not _stage_done(d, "kld")
    (d / "logprobs.jsonl.gz").write_bytes(good)
    assert _stage_done(d, "kld")
    for i in range(56):
        (lm / f"samples_mmlu_s{i}_x.jsonl").write_text('{"doc_id": 0}\n')
    assert not _stage_done(d, "mmlu")                                              # 56 of 57 subjects
    (lm / "samples_mmlu_s56_x.jsonl").write_text('{"doc_id": 0}\n')
    assert _stage_done(d, "mmlu")
