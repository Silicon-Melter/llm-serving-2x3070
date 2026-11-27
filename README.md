# Serving LLMs across 2 × RTX 3070 (8 GB, PCIe): a starter kit

Proof project for the TokenMe inference internship pitch. Question: **how do you serve a model cleanly across two
consumer GPUs with no NVLink, and when is it worth it?** Compare tensor parallel (TP), pipeline parallel (PP) and two
independent replicas (DP) in vLLM and SGLang, measure scaling efficiency, find the interconnect bottleneck, and show
the quantisation trade-off.

## Hardware facts to record first (`scripts/00_hw_facts.sh`)
- 2 × RTX 3070 8 GB, Ampere sm_86. **No native FP8 compute**: vLLM runs FP8 checkpoints as W8A16 (Marlin), so FP8
  saves memory but not maths. INT4 AWQ/GPTQ use Marlin kernels too.
- No NVLink. GeForce drivers usually disable PCIe peer-to-peer, so NCCL traffic goes through host memory. Check the
  PCIe link width of each card (x16/x4 or x8/x8 changes everything) with `nvidia-smi topo -m` and
  `nvidia-smi -q | grep -A3 "Link Width"`.

## Experiment matrix
| Model (HF id) | Weights | 1 GPU | TP=2 | PP=2 | DP=2 (2 replicas) |
|---|---|---|---|---|---|
| `Qwen/Qwen3-8B-AWQ` (INT4) | ~5.7 GB | yes (small KV) | yes | yes | yes |
| `Qwen/Qwen3-8B-FP8` (W8A16 on Ampere) | ~8.5 GB | no | yes | yes | no |
| `Qwen/Qwen3-14B-AWQ` (INT4) | ~9.5 GB | no | yes | yes | no |
| `Qwen/Qwen3-4B` (BF16) | ~8 GB | no (no room for KV) | yes | yes | no |

Expected story (the hypothesis to test, not a result): for a model that fits on one card, two replicas beat TP over
PCIe on throughput; TP wins on single-request latency; PP needs many concurrent requests to fill its bubble; for models
that need both cards, TP vs PP depends on the all-reduce cost per decode step, which `nccl-tests` and an `nsys` trace
will show.

## Metrics
- Output tokens/s, TTFT and TPOT (p50, p99) at concurrency 1, 4, 16, 64; input 512, output 256 tokens (plus one
  ShareGPT run and one Thai prompt set).
- **Scaling efficiency** = throughput on 2 GPUs / (2 × throughput on 1 GPU), for the 8B-AWQ model.
- **Interconnect:** NCCL all-reduce bus bandwidth (GB/s) for 8 KB to 256 MB; share of a decode step spent in
  all-reduce (nsys); TP with `--disable-custom-all-reduce` on and off.
- **Quality:** lm-evaluation-harness GSM8K + MMLU subset per checkpoint, against the BF16 reference numbers on the
  model card (or a run of the BF16 model on a borrowed bigger GPU).
- **Cost:** USD per 1M output tokens at an assumed price per GPU-hour (write the price down).

## Steps
1. `bash scripts/00_hw_facts.sh > results/hw.txt`
2. `bash scripts/01_setup.sh` (conda env `serve`, vLLM, SGLang in its own env, nccl-tests). It ends with
   `scripts/01b_jit_toolkit.sh`, a CUDA 13.0 toolkit from NVIDIA's wheels that FlashInfer compiles FP8-KV kernels
   with on sm_86. The venv's own nvcc is 13.5 against 13.0 headers, which CCCL rejects.
3. `bash scripts/02_nccl.sh` → `results/nccl_allreduce.txt`.
4. `bash scripts/03_sweep_vllm.sh` (each config: start server, wait, sweep concurrency, stop) → `results/vllm/*.json`.
5. `bash scripts/04_sweep_sglang.sh` → `results/sglang/<run>/*.json`. It uses the same client (`vllm bench serve`) and
   workload as step 4, so `05_analyse.py` reads both engines (column `engine`). Single-stream llama.cpp (T1 / T2):
   `bash scripts/13_llamacpp.sh` builds llama.cpp for sm_86, downloads the community's GGUFs and runs llama-bench
   → `results/llamacpp/<run>/`.
6. `python scripts/05_analyse.py results/ --runs m3-581eb68 m3b-ab5bb8b` → `results/summary.csv` and the charts. Name the runs: other folders under `results/vllm/` are pre-checks and probes with different KV sizes (e.g. `m3-tpcheck*` used automatic sizing, 24,576 KV tokens instead of 46,464).
7. Tuning round on the best config: CUDA graphs on/off, `--max-num-seqs`, chunked prefill, `--kv-cache-dtype fp8`,
   PP with more in-flight requests. Change one thing at a time and log why.
8. Write-up: README with the table, 3–4 charts, the nsys screenshot of the all-reduce, and "what I'd do on NVLink /
   H100".

## Lab rules
On this PC, register every run with the GPU board, e.g.
`board-job run ad-pc.tokenme.vllm-sweep --res gpu0,gpu1 --task "vLLM TP/PP/DP sweep" -- bash scripts/03_sweep_vllm.sh`
(check the gpu-board skill for the exact `--res` names). Stop if T29 or a collector needs the GPUs.
