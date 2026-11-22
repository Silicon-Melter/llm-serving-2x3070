"""Tests for scripts/05_analyse.py on synthetic vLLM 0.34 result files."""
import importlib.util
import json
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("analyse", ROOT / "scripts/05_analyse.py")
analyse = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analyse)


def write(run_dir, name, tput, failed=0, completed=128):
    d = {"completed": completed, "failed": failed, "output_throughput": tput, "request_throughput": 1.0,
         "median_ttft_ms": 10.0, "p99_ttft_ms": 20.0, "median_tpot_ms": 5.0, "p99_tpot_ms": 9.0,
         "median_itl_ms": 4.0, "p99_itl_ms": 8.0}
    (run_dir / name).write_text(json.dumps(d))


@pytest.fixture
def results(tmp_path):
    run = tmp_path / "vllm" / "runA"
    run.mkdir(parents=True)
    (run / "8b-awq_1gpu.server.txt").write_text("GPU KV cache size: 6,848 tokens, Maximum concurrency ...\n")
    write(run, "8b-awq_1gpu_c8_r1.json", 300.0)
    write(run, "8b-awq_1gpu_c16_r1.json", 360.0)
    write(run, "8b-awq_1gpu_c16_r2.json", 340.0)
    write(run, "8b-awq_1gpu_c16_r3.json", 0.0, failed=128, completed=0)   # failed run: must be dropped
    write(run, "8b-awq_dp2_c16_r1.json", 660.0)
    (run / "notes.json").write_text("{}")                                  # not a result file
    return tmp_path


def test_load_filters_and_parses(results):
    raw = analyse.load(str(results))
    assert len(raw) == 4                       # failed repeat and non-matching file skipped
    assert set(raw.repeat) == {1, 2}
    one = raw[raw.config == "8b-awq_1gpu"]
    assert (one.kv_tokens == 6848).all()


def test_summarise_repeats_and_efficiency(results):
    df = analyse.summarise(analyse.load(str(results)), usd_per_gpu_hour=0.15).set_index(["config", "concurrency"])
    c16 = df.loc[("8b-awq_1gpu", 16)]
    assert c16.n == 2 and c16.out_tok_s == pytest.approx(350.0)
    assert (c16.out_tok_s_min, c16.out_tok_s_max) == (340.0, 360.0)
    assert c16.usd_per_1M_out_tok == pytest.approx(0.15 / (350.0 * 3600) * 1e6)
    dp = df.loc[("8b-awq_dp2", 16)]
    assert dp.eff_same_conc == pytest.approx(660.0 / (2 * 350.0))
    assert dp.eff_same_load_per_gpu == pytest.approx(660.0 / (2 * 300.0))   # vs 1 GPU at c=8
