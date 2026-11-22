"""Collect vLLM bench JSONs into one table, compute scaling efficiency and cost, draw the charts.
Usage: python scripts/05_analyse.py results/ [--usd-per-gpu-hour 0.15]
vLLM result keys used: output_throughput, mean/median/p99 ttft_ms and tpot_ms. Check one JSON first; key names can
change between versions."""
import argparse, glob, json, os, re
import pandas as pd
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("results")
ap.add_argument("--usd-per-gpu-hour", type=float, default=0.15, help="assumed price of one RTX 3070 hour")
a = ap.parse_args()

rows = []
for f in glob.glob(os.path.join(a.results, "vllm", "*.json")):
    m = re.match(r"(.+)_c(\d+)\.json", os.path.basename(f))
    if not m:
        continue
    d = json.load(open(f))
    name, conc = m.group(1), int(m.group(2))
    gpus = 1 if "1gpu" in name else 2
    tput = d.get("output_throughput")
    rows.append(dict(engine="vllm", config=name, concurrency=conc, gpus=gpus, out_tok_s=tput,
                     ttft_p50=d.get("median_ttft_ms"), ttft_p99=d.get("p99_ttft_ms"),
                     tpot_p50=d.get("median_tpot_ms"), tpot_p99=d.get("p99_tpot_ms")))
df = pd.DataFrame(rows).sort_values(["config", "concurrency"])
df["usd_per_1M_out_tok"] = a.usd_per_gpu_hour * df.gpus / (df.out_tok_s * 3600) * 1e6

base = df[df.config == "8b-awq_1gpu"].set_index("concurrency").out_tok_s
for cfg in ["8b-awq_tp2", "8b-awq_pp2", "8b-awq_dp2"]:
    sel = df.config == cfg
    df.loc[sel, "scaling_eff_vs_1gpu"] = df[sel].apply(lambda r: r.out_tok_s / (2 * base.get(r.concurrency, float("nan"))), axis=1)

os.makedirs(os.path.join(a.results, "charts"), exist_ok=True)
df.to_csv(os.path.join(a.results, "summary.csv"), index=False)
print(df.to_string(index=False))

for metric, ylabel in [("out_tok_s", "output tokens/s"), ("tpot_p50", "TPOT p50 (ms)"), ("usd_per_1M_out_tok", "USD per 1M output tokens")]:
    plt.figure(figsize=(7, 4.5))
    for cfg, g in df.groupby("config"):
        plt.plot(g.concurrency, g[metric], marker="o", label=cfg)
    plt.xscale("log", base=2); plt.xlabel("concurrent requests"); plt.ylabel(ylabel); plt.grid(alpha=.3)
    plt.legend(fontsize=7); plt.tight_layout(); plt.savefig(os.path.join(a.results, "charts", f"{metric}.png"), dpi=150)
