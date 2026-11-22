"""H3: where the GPU time of a decode step goes, from vLLM torch-profiler traces (Chrome trace JSON, one per rank).

Kernels are put into classes by name: all-reduce (NCCL / custom all-reduce), GEMM (Marlin, cutlass, cuBLAS),
attention, and everything else. Over the profiled window, per rank:
  busy   = union of kernel intervals (GPU doing something)
  idle   = window - busy (launch gaps, CPU scheduling, waiting for the other GPU outside NCCL)
and each class's share of the window. The all-reduce share is H3's measure (predicted 15-35 %; falsified < 10 % or > 50 %).

  python scripts/11_trace_breakdown.py <trace dir or files...> [--json out.json]
"""
import argparse
import glob
import gzip
import json
import os
import re

CLASSES = [  # first match wins
    ("allreduce", re.compile(r"nccl|allreduce|all_reduce|cross_device_reduce", re.I)),
    ("gemm", re.compile(r"marlin|gemm|cutlass|sm\d+_xmma|ampere_.*(h|s)\d+gemm|cublas|matmul|scaled_mm", re.I)),
    ("attention", re.compile(r"flash|attn|attention|paged|unified_attention|reshape_and_cache", re.I)),
]


def classify(name):
    for cls, rx in CLASSES:
        if rx.search(name):
            return cls
    return "other"


def load_events(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as f:
        data = json.load(f)
    evs = data["traceEvents"] if isinstance(data, dict) else data
    return [e for e in evs if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")
            and "dur" in e]


def union_length(intervals):
    total, end = 0.0, None
    for s, e in sorted(intervals):
        if end is None or s > end:
            total += e - s
            end = e
        elif e > end:
            total += e - end
            end = e
    return total


def breakdown(events):
    if not events:
        return None
    start = min(e["ts"] for e in events)
    stop = max(e["ts"] + e["dur"] for e in events)
    window = stop - start
    per = {}
    for e in events:
        c = classify(e.get("name", ""))
        per.setdefault(c, []).append((e["ts"], e["ts"] + e["dur"]))
    busy = union_length([iv for ivs in per.values() for iv in ivs])
    out = {"window_ms": window / 1e3, "busy_share": busy / window, "idle_share": 1 - busy / window,
           "kernels": len(events)}
    for c in [k for k, _ in CLASSES] + ["other"]:
        out[f"{c}_share"] = union_length(per.get(c, [])) / window
        out[f"{c}_ms"] = union_length(per.get(c, [])) / 1e3
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+"); ap.add_argument("--json")
    a = ap.parse_args()
    files = []
    for p in a.paths:
        files += sorted(glob.glob(os.path.join(p, "**", "*.json*"), recursive=True)) if os.path.isdir(p) else [p]
    res = {}
    for f in files:
        b = breakdown(load_events(f))
        if b:
            res[os.path.basename(f)] = b
            print(f"{os.path.basename(f)[:60]:60s} window {b['window_ms']:8.1f} ms  all-reduce {b['allreduce_share']:6.1%}"
                  f"  gemm {b['gemm_share']:6.1%}  attention {b['attention_share']:6.1%}  other {b['other_share']:6.1%}"
                  f"  idle {b['idle_share']:6.1%}")
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)
