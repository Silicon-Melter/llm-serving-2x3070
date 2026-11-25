"""Theoretical ceiling for every measured point, so results are judged against what the hardware allows.

The ceiling uses only physics and the workload: peak memory bandwidth, peak tensor maths, and 8 GiB of VRAM per card.
It ignores software (CUDA context, activations, display, scheduling, all-reduce time), so measured / ceiling says how
much of the card the software stack delivers. Per request, the ceiling time is
    prefill  = (2 x params x prompt + attention FLOPs) / peak maths          (compute-bound)
    decode   = answer tokens x step(B) / B,  step(B) = max(bytes / bandwidth, 2 x params x B / peak maths)
             bytes = weights read per step + B x average context x KV bytes per token
with the batch B = min(users per engine, requests whose KV fits in 8 GiB minus weights). TP and PP split the bytes
and FLOPs over 2 cards (PP with a single user cannot overlap its stages); 2 copies (dp2) are two 1-card engines.

  python scripts/09_roofline.py results/summary.csv [--out results/roofline.csv]
"""
import argparse
import math
import re

import pandas as pd

GiB = 2 ** 30
HW = {"bw": 448e9, "vram": 8 * GiB, "fp16": 40.6e12, "int8": 162.6e12}  # RTX 3070: FP16 tensor with FP32 accumulate
# Other cards (for judging community results against their own hardware) are passed in as `hw` dicts of the same shape.
MODELS = {  # Qwen3 configs from the Hugging Face config.json files
    "4b": dict(L=36, h=2560, hq=32, kvh=8, hd=128, inter=9728, vocab=151936, tied=True),
    "8b": dict(L=36, h=4096, hq=32, kvh=8, hd=128, inter=12288, vocab=151936, tied=False),
    "14b": dict(L=40, h=5120, hq=40, kvh=8, hd=128, inter=17408, vocab=151936, tied=False),
    "llama8b": dict(L=32, h=4096, hq=32, kvh=8, hd=128, inter=14336, vocab=128256, tied=False),  # Llama-3.1-8B
}
FMT = {"bf16": (2.0, "fp16"), "fp16": (2.0, "fp16"), "fp8": (1.0, "fp16"), "awq": (0.52, "fp16"), "w8a8": (1.0, "int8"),
       "nvfp4": (0.5625, "fp4")}  # bytes per weight, maths used (NVFP4: 4 bits + one FP8 scale per 16 weights)


def model_numbers(size, fmt, hw=HW):
    m = MODELS[size]
    plin = m["L"] * (2 * m["h"] * m["hq"] * m["hd"] + 2 * m["h"] * m["kvh"] * m["hd"] + 3 * m["h"] * m["inter"])
    pemb = m["vocab"] * m["h"]
    b, maths = FMT[fmt]
    return dict(m, plin=plin, pemb=pemb,
                w_mem=plin * b + pemb * 2 * (1 if m["tied"] else 2),  # bytes held (embedding and output layer BF16)
                w_read=plin * b + pemb * 2,                             # bytes read per decode step
                p_flop=plin + pemb, peak=hw[maths], kv_tok=2 * m["L"] * m["kvh"] * m["hd"] * 2)


def ceiling(size, fmt, layout, users, prompt=512, answer=256, hw=HW):
    """Ceiling output tokens/s for `users` concurrent requests of prompt -> answer tokens, and the batch it assumes."""
    x = model_numbers(size, fmt, hw)
    split = 2 if layout in ("tp2", "pp2") else 1
    engines = 2 if layout == "dp2" else 1
    per_engine = [math.ceil(users / 2), users // 2] if engines == 2 else [users]
    ctx = prompt + answer / 2
    total, batches = 0.0, []
    for u in per_engine:
        if u == 0:
            continue
        kv_bytes = hw["vram"] - x["w_mem"] / split           # per card (PP: stages hold about half each)
        fit = int(kv_bytes / (x["kv_tok"] / split) // (prompt + answer))
        if fit < 1:
            return float("nan"), 0
        B = min(u, fit)
        par = split if not (layout == "pp2" and B == 1) else 1  # one request cannot fill both pipeline stages
        step = max((x["w_read"] + B * ctx * x["kv_tok"]) / par / hw["bw"], 2 * x["p_flop"] * B / par / x["peak"])
        pre_flop = 2 * x["plin"] * prompt + 2 * x["L"] * x["hq"] * x["hd"] * prompt ** 2
        prefill = pre_flop / (split if layout != "pp2" or B > 1 else 1) / x["peak"]
        req_per_s = B / (B * prefill + answer * step)
        total += req_per_s * answer
        batches.append(B)
    return total, sum(batches)


def decode_ceiling_single(size, fmt, layout, hw=HW):
    """Single-user decode tokens/s: bandwidth / bytes read per token (no prefill); compare with 1 / TPOT."""
    x = model_numbers(size, fmt, hw)
    par = 2 if layout == "tp2" else 1
    return hw["bw"] * par / x["w_read"]


def parse(config):
    m = re.fullmatch(r"(\d+b)-([a-z0-9]+)_(1gpu|tp2|pp2|dp2)(?:_.*)?", config)
    return m.groups() if m else None


def table(summary):
    rows = []
    for r in summary.itertuples():
        p = parse(r.config)
        if not p:
            continue
        size, fmt, layout = p
        ceil, B = ceiling(size, fmt, layout, int(r.concurrency))
        row = dict(config=r.config, users=int(r.concurrency), measured=r.out_tok_s, ceiling=ceil, batch_ceiling=B,
                   share=r.out_tok_s / ceil)
        if r.concurrency == 1:
            dc = decode_ceiling_single(size, fmt, layout)
            row.update(decode_measured=1000 / r.tpot_p50, decode_ceiling=dc, decode_share=1000 / r.tpot_p50 / dc)
        rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("summary"); ap.add_argument("--out")
    a = ap.parse_args()
    df = table(pd.read_csv(a.summary))
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(df.to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    if a.out:
        df.to_csv(a.out, index=False)
