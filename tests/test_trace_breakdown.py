"""Tests for scripts/11_trace_breakdown.py on a synthetic Chrome trace."""
import gzip
import importlib.util
import json
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("tb", ROOT / "scripts/11_trace_breakdown.py")
tb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tb)


@pytest.mark.parametrize("name,cls", [
    ("ncclDevKernel_AllReduce_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<4096ul>)", "allreduce"),
    ("void marlin::Marlin<(vllm::ScalarTypeId)5, 256, 4, 8, 4, 8>(...)", "gemm"),
    ("ampere_fp16_s16816gemm_fp16_128x128_ldg8_f2f_stages_32x5_tn", "gemm"),
    ("void flash::flash_fwd_splitkv_kernel<...>", "attention"),
    ("triton_red_fused_add_rms_norm_0", "other"),
])
def test_classify(name, cls):
    assert tb.classify(name) == cls


def test_breakdown_shares_and_idle(tmp_path):
    k = lambda name, ts, dur: {"ph": "X", "cat": "kernel", "name": name, "ts": ts, "dur": dur}  # noqa: E731
    events = [k("marlin_gemm", 0, 40), k("ncclDevKernel_AllReduce", 40, 20), k("flash_attn", 60, 10),
              k("rms_norm", 70, 10), k("marlin_gemm", 90, 10),  # gap 80-90 = idle
              {"ph": "X", "cat": "cpu_op", "name": "aten::mm", "ts": 0, "dur": 100}]  # CPU events ignored
    p = tmp_path / "rank0.pt.trace.json.gz"
    with gzip.open(p, "wt") as f:
        json.dump({"traceEvents": events}, f)
    b = tb.breakdown(tb.load_events(str(p)))
    assert b["window_ms"] == pytest.approx(0.1)
    assert b["allreduce_share"] == pytest.approx(0.2) and b["gemm_share"] == pytest.approx(0.5)
    assert b["attention_share"] == pytest.approx(0.1) and b["other_share"] == pytest.approx(0.1)
    assert b["idle_share"] == pytest.approx(0.1)


def test_overlapping_kernels_are_not_double_counted():
    assert tb.union_length([(0, 10), (5, 15), (20, 30)]) == 25
