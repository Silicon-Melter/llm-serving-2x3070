# Inputs for results/frontier_share.csv (scripts/12_frontier_share.py)

Collected 2026-11-25 by two research subagents, read-only web. V = read on the page, V* = read from the file's own
GGUF header, D = derived (arithmetic shown), S = secondary source, U = inferred.

## Hardware (dense tensor maths, FP16 inputs with FP32 accumulate unless noted)
| Card | Bandwidth | FP16 | INT8 | FP4 | Source |
|---|---|---|---|---|---|
| RTX 3070 8 GB | 448 GB/s V | 40.6 TFLOPS V | 162.6 TOPS V | — | Ampere GA102 whitepaper v2, Table 10 |
| RTX 5060 Ti 16 GB | 448 GB/s V | 47.4 D | 189.5 D | 379.0 D | nvidia.com compare page; per-SM rates from the RTX Blackwell whitepaper App. C (5070 row) × 36 SMs × 2.57 GHz; FP4 matches 759 sparse AI TOPS ÷ 2 |
| RTX 4060 Ti 8 GB | 288 GB/s S | 44.2 D | 176.9 D | — | Ada per-SM rates (whitepaper v2.1 App. A) × 34 SMs × 2.54 GHz; bandwidth from Wikipedia |
| RTX 4090 24 GB | 1008 GB/s V | 165.2 V | 660.6 V | — | Ada whitepaper v2.1, App. A, Table 2 |

GeForce Blackwell runs FP16/FP8 with FP32 accumulate at half rate (V, 5070 row), as Ampere and Ada do.

## Model bytes streamed per decode step
| File | Tensor bytes (V*) | token_embd.weight (V*) | Streamed (D) |
|---|---|---|---|
| JaaackXD/Llama-3-8B-GGUF ggml-model-Q4_K_M.gguf (XiongjieDai's file, "4.58 GB" = GiB) | 4,912,898,048 | Q4_K 295,501,824 | 4,617,396,224 |
| Qwen/Qwen3-14B-GGUF Qwen3-14B-Q4_K_M.gguf | 8,995,793,920 | Q4_K 437,575,680 | 8,558,218,240 |
| Qwen/Qwen3-32B-GGUF Qwen3-32B-Q4_K_M.gguf | 19,756,174,336 | Q4_K 437,575,680 | 19,318,598,656 |
vLLM checkpoints use scripts/09_roofline.py's shapes; NVFP4 = 4.5 bits per weight (E2M1 + one E4M3 scale per 16, NVIDIA
blog "Introducing NVFP4"), output layer BF16. Llama-3.1-8B shape from deepseek-ai/DeepSeek-R1-Distill-Llama-8B config.json.

## Workloads
| Source | Workload used | Notes |
|---|---|---|
| [XiongjieDai](https://github.com/XiongjieDai/GPU-Benchmarks-on-LLM-Inference) | llama-bench tg512, 72.79 tok/s | V; flash attention off by default then (U) |
| [XDA 2026-11-17](https://www.xda-developers.com/two-cheap-16gb-graphics-cards-run-bigger-local-models-than-one-expensive-24gb-card/) | decode, depth not stated | treated as short context; x8/x8 from the CPU (V) |
| [arXiv 2601.09527](https://arxiv.org/html/2601.09527v1) Table 16 | 128–512 in (median 256), ≤ 256 out, output tok/s, FP16 KV (U), Poisson arrivals under a concurrency cap | outputs not stated as forced and real concurrency can sit below the cap, so the ceiling is high and the share a lower bound |
| [charlesmorin1 field notes](https://github.com/charlesmorin1/vllm-consumer-gpu-field-notes) | ~120 in (U: 70-word paragraph + chat template), 256 out (thinking on), closed batches of n × 3 reps, output tok/s incl. TTFT | c=1 73.78, c=32 1340.93 aggregate |
| [DatabaseMart 2×4090](https://www.databasemart.com/blog/vllm-gpu-benchmark-dual-rtx4090) | 100 requests at once, 100 in, 545 out on average (54,514 tokens generated), FP16, TP=2 | 3194.74 output tok/s; one closed batch, so its tail lowers the measured rate |

Left out: ai-muninn 2×2080 Ti (Qwen3.8-27B shape and decode depth not stated), MTP / speculative rows (more than one
token per weight read, so the plain ceiling does not apply), Reddit rows (not verifiable).
