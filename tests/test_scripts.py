"""Checks for the shell scripts in scripts/. Run: python -m pytest tests -q
The library test needs the envs and nccl-tests build from scripts/01_setup.sh; it is skipped (with a reason) if they
are missing, so the syntax checks still run on any machine."""
import gzip
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


def test_quality_stage_done_reads_gzipped_samples(tmp_path):
    """Committed results keep lm-eval samples gzipped: a resumed run must still see them as done."""
    import gzip
    d = tmp_path / "cfg"; lm = d / "lm_eval" / "m"; lm.mkdir(parents=True)
    (lm / "samples_gsm8k_x.jsonl.gz").write_bytes(gzip.compress(b""))              # empty inside
    assert not _stage_done(d, "gsm8k")
    (lm / "samples_gsm8k_x.jsonl.gz").write_bytes(gzip.compress(b'{"doc_id": 0}\n'))
    assert _stage_done(d, "gsm8k")
    for i in range(29):                                                            # 29 subjects, each plain + gz
        (lm / f"samples_mmlu_s{i}_x.jsonl").write_text('{"doc_id": 0}\n')
        (lm / f"samples_mmlu_s{i}_x.jsonl.gz").write_bytes(gzip.compress(b'{"doc_id": 0}\n'))
    assert not _stage_done(d, "mmlu")                                              # 58 files, 29 subjects
    for i in range(29, 57):
        (lm / f"samples_mmlu_s{i}_x.jsonl.gz").write_bytes(gzip.compress(b'{"doc_id": 0}\n'))
    assert _stage_done(d, "mmlu")


def test_sweep_sets_the_flashinfer_build_environment_only_for_fp8_kv():
    """FP8 KV on sm_86 goes through FlashInfer's JIT: CUDA_HOME must be the CUDA 13.0 toolkit 01b builds, ninja on PATH."""
    run = lambda extra: subprocess.run(["bash", str(ROOT / "scripts/03_sweep_vllm.sh"), "--kv-env", extra],  # noqa: E731
                                       capture_output=True, text=True, env={**os.environ, "BULK": "/b"}).stdout.split()
    env = dict(kv.split("=", 1) for kv in run("--tensor-parallel-size 2 --kv-cache-dtype fp8"))
    home = subprocess.run(["bash", str(ROOT / "scripts/01b_jit_toolkit.sh"), "--home"], capture_output=True, text=True,
                          env={**os.environ, "BULK": "/b"}).stdout.strip()
    assert env["CUDA_HOME"] == home == "/b/cuda-jit-13.0/nvidia/cu13"
    assert env["PATH"].split(":")[:2] == ["/b/envs/serve-vllm/bin", env["CUDA_HOME"] + "/bin"]
    assert env["FLASHINFER_CUDA_ARCH_LIST"] == "8.6"
    assert run("--tensor-parallel-size 2") == [] and run("") == []
    # the BF16-KV control on FlashInfer needs the same JIT environment
    assert run("--tensor-parallel-size 2 --attention-backend FLASHINFER") == run("--kv-cache-dtype fp8")


def test_quality_and_sweep_use_the_same_fp8_kv_environment():
    out = lambda script, extra: subprocess.run(["bash", str(ROOT / "scripts" / script), "--kv-env", extra],  # noqa: E731
                                               capture_output=True, text=True, env={**os.environ, "BULK": "/b"}).stdout
    for extra in ("--tensor-parallel-size 2 --kv-cache-dtype fp8 --kv-cache-memory-bytes 1", "--tensor-parallel-size 2",
                  "--tensor-parallel-size 2 --attention-backend FLASHINFER"):
        assert out("06_quality.sh", extra) == out("03_sweep_vllm.sh", extra)
    assert out("06_quality.sh", "--kv-cache-dtype fp8").startswith("CUDA_HOME=")


def _quality_run(tmp_path, **env):
    """06_quality.sh in a scratch folder: a vllm that exits at once, an nvidia-smi that reports idle GPUs."""
    fake = tmp_path / "bin"; fake.mkdir(exist_ok=True)
    (fake / "nvidia-smi").write_text("#!/bin/sh\necho 0\n"); (fake / "nvidia-smi").chmod(0o755)
    venv = tmp_path / "bulk/envs/serve-vllm/bin"; venv.mkdir(parents=True, exist_ok=True)
    (venv / "vllm").write_text("#!/bin/sh\necho 'ValueError: fake server' >&2\nexit 1\n"); (venv / "vllm").chmod(0o755)
    (tmp_path / "results/quality").mkdir(parents=True, exist_ok=True); (tmp_path / "results/quality/corpus.json").write_text("{}")
    base = {**os.environ, "PATH": f"{fake}:{os.environ['PATH']}", "BULK": str(tmp_path / "bulk"), "PORT": "59997",
            "RUN_ID": "t", "ONLY": "4b-fp8_1gpu", "STAGES": "kld", "CPUSET": "0"}
    return subprocess.run(["bash", str(ROOT / "scripts/06_quality.sh")], cwd=tmp_path, capture_output=True, text=True,
                          env={**base, **env}, timeout=120)


def test_quality_exits_nonzero_when_a_stage_is_left_undone(tmp_path):
    r = _quality_run(tmp_path)
    assert "did not start" in r.stdout and "incomplete: 4b-fp8_1gpu:kld" in r.stdout
    assert r.returncode == 1


def test_quality_exits_zero_when_every_stage_is_done(tmp_path):
    d = tmp_path / "results/quality/t/4b-fp8_1gpu"; d.mkdir(parents=True)
    (d / "logprobs.jsonl.gz").write_bytes(gzip.compress(b'{"x": 1}\n'))
    r = _quality_run(tmp_path)
    assert "already done, skipped" in r.stdout and "incomplete" not in r.stdout
    assert r.returncode == 0


def _configs(script):
    """The CONFIGS array of a sweep script, evaluated by bash itself (the rest of the script is not run)."""
    cmd = f'eval "$(sed -n "/^CONFIGS=(/,/^)/p" "{script}")"; printf "%s\\n" "${{CONFIGS[@]}}"'
    out = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, check=True).stdout
    return [line.split("|") for line in out.splitlines()]


VARIANT_FLAG = {"_fp8kv": "--kv-cache-dtype fp8", "_fi": "--attention-backend FLASHINFER", "_s128": "--max-num-seqs 128"}


@pytest.mark.parametrize("script", ["03_sweep_vllm.sh", "06_quality.sh"])
def test_config_tables_change_one_thing_per_variant(script):
    """Every row has 4 fields and a unique name; a *_fp8kv / *_fi row is its twin plus exactly one flag."""
    rows = _configs(ROOT / "scripts" / script)
    assert len(rows) > 5 and all(len(r) == 4 for r in rows)
    by_name = {r[0]: r for r in rows}
    assert len(by_name) == len(rows)
    assert all(gpus in ("0", "1", "0,1") for _, gpus, _, _ in rows)
    variants = 0
    for name, gpus, model, extra in rows:
        for suffix, flag in VARIANT_FLAG.items():
            if name.endswith(suffix) and name[: -len(suffix)] in by_name:
                _, tgpus, tmodel, textra = by_name[name[: -len(suffix)]]
                assert flag in extra, name
                assert (gpus, model) == (tgpus, tmodel), name
                assert " ".join(extra.replace(flag, " ").split()) == " ".join(textra.split()), name
                variants += 1
    assert variants >= 2


def test_sweep_max_num_seqs_defaults_to_64_and_follows_the_config():
    seqs = lambda extra: subprocess.run(["bash", str(ROOT / "scripts/03_sweep_vllm.sh"), "--seqs", extra],  # noqa: E731
                                        capture_output=True, text=True).stdout.strip()
    assert seqs("--tensor-parallel-size 2") == "64" and seqs("") == "64"
    assert seqs("--tensor-parallel-size 2 --kv-cache-dtype fp8 --max-num-seqs 128") == "128"
    assert seqs("--max-num-seqs 32 --tensor-parallel-size 2") == "32"


def test_sweep_server_gets_max_num_seqs_exactly_once():
    """COMMON has no --max-num-seqs: 64 is added for plain configs, nothing when the config sets its own."""
    arg = lambda extra: subprocess.run(["bash", str(ROOT / "scripts/03_sweep_vllm.sh"), "--seq-arg", extra],  # noqa: E731
                                       capture_output=True, text=True).stdout.split()
    assert arg("--tensor-parallel-size 2 --kv-cache-dtype fp8") == ["--max-num-seqs", "64"]
    assert arg("") == ["--max-num-seqs", "64"]
    assert arg("--tensor-parallel-size 2 --max-num-seqs 128") == []
    common = subprocess.run(["bash", "-c", f'grep -E "^COMMON=" "{ROOT / "scripts/03_sweep_vllm.sh"}"'],
                            capture_output=True, text=True).stdout
    assert "--max-model-len 4096" in common and "--max-num-seqs" not in common


def test_sweep_starts_each_server_with_the_right_max_num_seqs(tmp_path):
    """The real sweep loop against a fake vllm: `serve` answers /health (python http.server), `bench` logs its
    arguments and fails. Checks the server command line and the concurrency of the KV smoke test, per config."""
    fake = tmp_path / "bin"; fake.mkdir()
    (fake / "nvidia-smi").write_text("#!/bin/sh\necho 0\n"); (fake / "nvidia-smi").chmod(0o755)
    www = tmp_path / "www"; www.mkdir(); (www / "health").write_text("ok")
    # bash starts background jobs with SIGINT ignored; the fake server restores it so stop_server's INT works
    (tmp_path / "server.py").write_text(
        "import functools, http.server, signal\n"
        "signal.signal(signal.SIGINT, signal.default_int_handler)\n"
        f"h = functools.partial(http.server.SimpleHTTPRequestHandler, directory={str(www)!r})\n"
        "http.server.ThreadingHTTPServer(('127.0.0.1', 59996), h).serve_forever()\n")
    venv = tmp_path / "bulk/envs/serve-vllm/bin"; venv.mkdir(parents=True)
    (venv / "vllm").write_text(f"""#!/bin/sh
echo "ARGS $*"
[ "$1" = serve ] && exec python3 -I {tmp_path / "server.py"} 2>/dev/null
exit 1
""")
    (venv / "vllm").chmod(0o755)
    for tool in ("hf", "huggingface-cli"):
        (venv / tool).write_text("#!/bin/sh\nexit 0\n"); (venv / tool).chmod(0o755)
    env = {**os.environ, "PATH": f"{fake}:{os.environ['PATH']}", "BULK": str(tmp_path / "bulk"), "PORT": "59996",
           "RUN_ID": "t", "ONLY": "8b-awq_tp2 8b-awq_tp2_fp8kv_s128", "CPUSET": "0", "KV_STEPS": "0.95"}
    r = subprocess.run(["bash", str(ROOT / "scripts/03_sweep_vllm.sh")], cwd=tmp_path, capture_output=True, text=True,
                       env=env, timeout=180)
    assert "failed the c=64 smoke test" in r.stdout and "failed the c=128 smoke test" in r.stdout, r.stdout
    out = tmp_path / "results/vllm/t"
    words = lambda f: (out / f).read_text().split()  # noqa: E731
    after = lambda w, flag: [w[i + 1] for i, x in enumerate(w) if x == flag]  # noqa: E731
    assert after(words("8b-awq_tp2.try-auto.log"), "--max-num-seqs") == ["64"]
    assert after(words("8b-awq_tp2_fp8kv_s128.try-auto.log"), "--max-num-seqs") == ["128"]
    assert after(words("8b-awq_tp2.try-auto.smoke.log"), "--max-concurrency") == ["64"]
    assert after(words("8b-awq_tp2_fp8kv_s128.try-auto.smoke.log"), "--max-concurrency") == ["128"]
