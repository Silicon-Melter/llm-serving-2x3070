"""Community results judged against their own hardware's ceiling, next to ours against the RTX 3070's.

Raw tok/s across different cards mostly measures the card. Dividing each result by the ceiling of the hardware it
ran on (the same physics as scripts/09_roofline.py: peak bandwidth, peak tensor maths, VRAM, the workload) says how
much of that hardware the software delivered, which is comparable across cards.

Single stream (decode only, short context):  ceiling = cards in the tensor split x bandwidth / bytes read per token.
    GGUF files: bytes read = file size - token_embd.weight (looked up one row per token, not streamed).
    Layer split runs the cards one after the other, so its ceiling is one card's.
Batched: scripts/09_roofline.ceiling() with the source's card, prompt and answer lengths and concurrency.

  python scripts/12_frontier_share.py [--roofline results/roofline.csv] [--out results/frontier_share.csv]
"""
import argparse
import importlib.util
import pathlib

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("roof", ROOT / "scripts/09_roofline.py")
roof = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(roof)
GiB = roof.GiB

# Peak DRAM bandwidth and dense tensor maths (no sparsity; FP16 = FP16 inputs with FP32 accumulate, the mode
# vLLM and llama.cpp use). Sources are listed in results/frontier_sources.md.
GPUS = {
    "RTX 3070 8GB": roof.HW,
    "RTX 5060 Ti 16GB": dict(bw=448e9, vram=16 * GiB, fp16=47.4e12, int8=189.5e12, fp4=379.0e12),
    "RTX 4060 Ti 8GB": dict(bw=288e9, vram=8 * GiB, fp16=44.2e12, int8=176.9e12),
    "RTX 4090 24GB": dict(bw=1008e9, vram=24 * GiB, fp16=165.2e12, int8=660.6e12),
}

# Bytes streamed per decode step = GGUF tensor bytes - token_embd.weight, from the files' own headers:
# JaaackXD/Llama-3-8B-GGUF (the XiongjieDai file), Qwen/Qwen3-14B-GGUF, Qwen/Qwen3-32B-GGUF, Q4_K_M.
LLAMA3_8B_Q4KM = 4_912_898_048 - 295_501_824
QWEN3_14B_Q4KM = 8_995_793_920 - 437_575_680
QWEN3_32B_Q4KM = 19_756_174_336 - 437_575_680

# Single stream, decode only. read = bytes streamed per token. measured = decode tok/s as posted.
SINGLE = [
    dict(source="XiongjieDai 2024 (llama-bench tg512)", gpu="RTX 3070 8GB", cards=1, engine="llama.cpp",
         model="Llama-3-8B Q4_K_M", read=LLAMA3_8B_Q4KM, measured=72.79),
    dict(source="XDA 2026-11-17", gpu="RTX 5060 Ti 16GB", cards=1, engine="llama.cpp, 1 card",
         model="Qwen3-14B Q4_K_M", read=QWEN3_14B_Q4KM, measured=46.31),
    dict(source="XDA 2026-11-17", gpu="RTX 5060 Ti 16GB", cards=1, engine="llama.cpp, layer split x2",
         model="Qwen3-14B Q4_K_M", read=QWEN3_14B_Q4KM, measured=46.37),
    dict(source="XDA 2026-11-17", gpu="RTX 5060 Ti 16GB", cards=2, engine="llama.cpp, tensor split x2",
         model="Qwen3-14B Q4_K_M", read=QWEN3_14B_Q4KM, measured=79.64),
    dict(source="XDA 2026-11-17", gpu="RTX 5060 Ti 16GB", cards=1, engine="llama.cpp, layer split x2",
         model="Qwen3-32B Q4_K_M", read=QWEN3_32B_Q4KM, measured=20.82),
    dict(source="XDA 2026-11-17", gpu="RTX 5060 Ti 16GB", cards=2, engine="llama.cpp, tensor split x2",
         model="Qwen3-32B Q4_K_M", read=QWEN3_32B_Q4KM, measured=37.16),
]

# Batched serving: the source's workload. answer = output length (cap if outputs were not forced).
BATCHED = [
    dict(source="arXiv 2601.09527 (vLLM 0.12)", gpu="RTX 5060 Ti 16GB", layout="1gpu", size="8b", fmt="nvfp4",
         users=64, prompt=256, answer=256, measured=1797.5),
    dict(source="arXiv 2601.09527 (vLLM 0.12)", gpu="RTX 5060 Ti 16GB", layout="tp2", size="8b", fmt="nvfp4",
         users=64, prompt=256, answer=256, measured=1646.4),
    dict(source="arXiv 2601.09527 (vLLM 0.12)", gpu="RTX 5060 Ti 16GB", layout="1gpu", size="8b", fmt="nvfp4",
         users=128, prompt=256, answer=256, measured=2113.9),
    dict(source="arXiv 2601.09527 (vLLM 0.12)", gpu="RTX 5060 Ti 16GB", layout="tp2", size="8b", fmt="nvfp4",
         users=128, prompt=256, answer=256, measured=2682.4),
    dict(source="charlesmorin1 field notes (vLLM 0.28)", gpu="RTX 4060 Ti 8GB", layout="1gpu", size="4b", fmt="awq",
         users=1, prompt=120, answer=256, measured=73.78),
    dict(source="charlesmorin1 field notes (vLLM 0.28)", gpu="RTX 4060 Ti 8GB", layout="1gpu", size="4b", fmt="awq",
         users=32, prompt=120, answer=256, measured=1340.93),
    dict(source="DatabaseMart (vLLM)", gpu="RTX 4090 24GB", layout="tp2", size="llama8b", fmt="fp16",
         users=100, prompt=100, answer=545, measured=3194.74),
]


def community():
    rows = []
    for r in SINGLE:
        hw = GPUS[r["gpu"]]
        ceil = r["cards"] * hw["bw"] / r["read"]
        rows.append(dict(side="community", regime="single stream", source=r["source"], hardware=r["gpu"],
                         setup=r["engine"], model=r["model"], users=1, measured=r["measured"], ceiling=ceil,
                         share=r["measured"] / ceil))
    for r in BATCHED:
        hw = GPUS[r["gpu"]]
        ceil, _ = roof.ceiling(r["size"], r["fmt"], r["layout"], r["users"], r["prompt"], r["answer"], hw)
        rows.append(dict(side="community", regime="batched", source=r["source"], hardware=r["gpu"],
                         setup=r["layout"], model=f'{r["size"]} {r["fmt"]}, {r["prompt"]} in / {r["answer"]} out',
                         users=r["users"], measured=r["measured"], ceiling=ceil, share=r["measured"] / ceil))
    return pd.DataFrame(rows)


def ours(roofline_csv):
    """Our points from 09_roofline's table: single-stream decode (1 / TPOT p50) and batched (512 in / 256 out)."""
    df = pd.read_csv(roofline_csv)
    rows = []
    for r in df.itertuples():
        if r.users == 1 and r.decode_ceiling == r.decode_ceiling:
            rows.append(dict(side="ours", regime="single stream", source="this study (vLLM 0.34)",
                             hardware="RTX 3070 8GB", setup=r.config, model=r.config.split("_")[0], users=1,
                             measured=r.decode_measured, ceiling=r.decode_ceiling, share=r.decode_share))
        rows.append(dict(side="ours", regime="batched", source="this study (vLLM 0.34)", hardware="RTX 3070 8GB",
                         setup=r.config, model=f"{r.config.split('_')[0]}, 512 in / 256 out", users=r.users,
                         measured=r.measured, ceiling=r.ceiling, share=r.share))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--roofline", default=str(ROOT / "results/roofline.csv"))
    ap.add_argument("--out")
    a = ap.parse_args()
    df = pd.concat([community(), ours(a.roofline)], ignore_index=True)
    with pd.option_context("display.width", 250, "display.max_columns", 20, "display.max_colwidth", 45):
        print(df.to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    if a.out:
        df.to_csv(a.out, index=False)
