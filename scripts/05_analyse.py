"""Collect vLLM bench JSONs into one table, compute scaling efficiency and cost, draw the charts.
Usage: python scripts/05_analyse.py results/ [--usd-per-gpu-hour 0.15] [--runs RUN_ID ...]
Reads results/vllm/<run-id>/<config>_c<C>_r<k>.json (written by 03_sweep_vllm.sh, vLLM 0.34 key names) and the
matching <config>.server.txt for the KV-cache size. Repeats are averaged; min and max are kept as the spread."""
import argparse
import glob
import json
import os
import re

import pandas as pd

KEYS = {  # column -> vLLM 0.34 result key
    "out_tok_s": "output_throughput", "req_s": "request_throughput",
    "ttft_p50": "median_ttft_ms", "ttft_p99": "p99_ttft_ms",
    "tpot_p50": "median_tpot_ms", "tpot_p99": "p99_tpot_ms", "itl_p50": "median_itl_ms", "itl_p99": "p99_itl_ms",
}


def kv_tokens(run_dir, config):
    try:
        txt = open(os.path.join(run_dir, f"{config}.server.txt")).read()
    except OSError:
        return None
    m = re.search(r"GPU KV cache size: ([\d,]+) tokens", txt)
    return int(m.group(1).replace(",", "")) if m else None


def load(results, runs=None):
    rows = []
    for f in sorted(glob.glob(os.path.join(results, "vllm", "*", "*.json"))):
        run = os.path.basename(os.path.dirname(f))
        m = re.fullmatch(r"(.+)_c(\d+)_r(\d+)\.json", os.path.basename(f))
        if not m or (runs and run not in runs):
            continue
        d = json.load(open(f))
        if not d.get("completed") or d.get("failed"):
            continue  # a run with failed requests is not a measurement
        cfg, conc, rep = m.group(1), int(m.group(2)), int(m.group(3))
        row = dict(engine="vllm", run=run, config=cfg, concurrency=conc, repeat=rep,
                   gpus=1 if "1gpu" in cfg else 2, kv_tokens=kv_tokens(os.path.dirname(f), cfg))
        row.update({k: d.get(v) for k, v in KEYS.items()})
        rows.append(row)
    return pd.DataFrame(rows)


def summarise(raw, usd_per_gpu_hour):
    g = raw.groupby(["engine", "config", "concurrency", "gpus"], as_index=False)
    df = g.agg(n=("repeat", "count"), kv_tokens=("kv_tokens", "max"), out_tok_s=("out_tok_s", "mean"),
               out_tok_s_min=("out_tok_s", "min"), out_tok_s_max=("out_tok_s", "max"),
               **{c: (c, "mean") for c in KEYS if c != "out_tok_s"})
    df["usd_per_1M_out_tok"] = usd_per_gpu_hour * df.gpus / (df.out_tok_s * 3600) * 1e6
    base = df[df.config == "8b-awq_1gpu"].set_index("concurrency").out_tok_s
    for cfg in ["8b-awq_tp2", "8b-awq_pp2", "8b-awq_dp2"]:
        sel = df.config == cfg
        # same total concurrency: 2-GPU throughput / (2 x 1-GPU throughput)
        df.loc[sel, "eff_same_conc"] = df[sel].apply(lambda r: r.out_tok_s / (2 * base.get(r.concurrency, float("nan"))), axis=1)
        # same load per GPU: compare with 1 GPU at half the concurrency (what each DP replica actually sees)
        df.loc[sel, "eff_same_load_per_gpu"] = df[sel].apply(
            lambda r: r.out_tok_s / (2 * base.get(r.concurrency // 2, float("nan"))), axis=1)
    return df.sort_values(["config", "concurrency"])


def charts(df, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(out_dir, exist_ok=True)
    for metric, ylabel in [("out_tok_s", "output tokens/s"), ("tpot_p50", "TPOT p50 (ms)"),
                           ("usd_per_1M_out_tok", "USD per 1M output tokens")]:
        plt.figure(figsize=(7, 4.5))
        for cfg, g in df.groupby("config"):
            plt.plot(g.concurrency, g[metric], marker="o", label=cfg)
        plt.xscale("log", base=2); plt.xlabel("concurrent requests"); plt.ylabel(ylabel); plt.grid(alpha=.3)
        plt.legend(fontsize=7); plt.tight_layout(); plt.savefig(os.path.join(out_dir, f"{metric}.png"), dpi=150)
        plt.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--usd-per-gpu-hour", type=float, default=0.15, help="assumed price of one RTX 3070 hour")
    ap.add_argument("--runs", nargs="*", help="only these run ids (folders under results/vllm/)")
    ap.add_argument("--no-charts", action="store_true")
    a = ap.parse_args()
    raw = load(a.results, a.runs)
    if raw.empty:
        raise SystemExit("no successful vLLM results found")
    df = summarise(raw, a.usd_per_gpu_hour)
    df.to_csv(os.path.join(a.results, "summary.csv"), index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(df.to_string(index=False, float_format=lambda x: f"{x:.3g}"))
    if not a.no_charts:
        charts(df, os.path.join(a.results, "charts"))
