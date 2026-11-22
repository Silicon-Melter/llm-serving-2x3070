"""H3: where the GPU time of a decode step goes, from vLLM torch-profiler traces (Chrome trace JSON, one per rank).

Kernels are put into classes by name: all-reduce (every NCCL collective: the per-layer all-reduces plus the one
logits all-gather per step; custom all-reduce), GEMM (Marlin, cutlass, cuBLAS),
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


def prefill_windows(events, long_us=100, join_us=300e3, pad_us=20e3):
    """Time windows that hold a prefill: clusters of long all-reduces (a 512-token prefill sums ~4 MB per all-reduce,
    ~0.5 ms; a decode step sums 8 KB, ~11 us). Only valid on the rank that does not wait for the other inside NCCL;
    pick it with the most clusters (see __main__)."""
    longs = sorted((e for e in events if "AllReduce" in e.get("name", "") and e["dur"] >= long_us), key=lambda e: e["ts"])
    cl = []
    for e in longs:
        if cl and e["ts"] - cl[-1][1] < join_us:
            cl[-1][1] = e["ts"] + e["dur"]
        else:
            cl.append([e["ts"], e["ts"] + e["dur"]])
    return [(a - pad_us, b + pad_us) for a, b in cl]


def breakdown(events, exclude=()):
    """Shares of the profiled window; `exclude` = time windows (e.g. prefills) left out of both window and kernels."""
    if exclude:
        events = [e for e in events if not any(a <= e["ts"] <= b for a, b in exclude)]
    if not events:
        return None
    start = min(e["ts"] for e in events)
    stop = max(e["ts"] + e["dur"] for e in events)
    window = stop - start - sum(min(b, stop) - max(a, start) for a, b in exclude if b > start and a < stop)
    per = {}
    for e in events:
        c = classify(e.get("name", ""))
        per.setdefault(c, []).append((e["ts"], e["ts"] + e["dur"]))
    busy = union_length([iv for ivs in per.values() for iv in ivs])
    out = {"window_ms": window / 1e3, "busy_share": busy / window, "idle_share": 1 - busy / window,
           "kernels": len(events), "steps": sum(1 for e in events if "AllGather" in e.get("name", ""))}
    for c in [k for k, _ in CLASSES] + ["other"]:
        out[f"{c}_share"] = union_length(per.get(c, [])) / window
        out[f"{c}_ms"] = union_length(per.get(c, [])) / 1e3
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+"); ap.add_argument("--json")
    ap.add_argument("--decode-only", action="store_true", help="leave out prefill windows (single-user runs)")
    a = ap.parse_args()
    files = []
    for p in a.paths:
        files += sorted(glob.glob(os.path.join(p, "**", "*.json*"), recursive=True)) if os.path.isdir(p) else [p]
    res, loaded, windows = {}, {}, ()
    for f in files:
        loaded[f] = load_events(f)
    if a.decode_only:  # prefill windows from the rank that shows them cleanly (most clusters); same host clock
        windows = max((prefill_windows(ev) for ev in loaded.values()), key=len)
        print(f"decode-only: {len(windows)} prefill windows left out")
    for f in files:
        b = breakdown(loaded.pop(f), exclude=windows)
        if b:
            res[os.path.basename(f)] = b
            print(f"{os.path.basename(f)[:60]:60s} window {b['window_ms']:8.1f} ms  all-reduce {b['allreduce_share']:6.1%}"
                  f"  gemm {b['gemm_share']:6.1%}  attention {b['attention_share']:6.1%}  other {b['other_share']:6.1%}"
                  f"  idle {b['idle_share']:6.1%}  steps {b['steps']}")
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)
