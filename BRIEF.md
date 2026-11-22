# Brief: serve an LLM across 2 × RTX 3070 (ad-pc)

| | |
|---|---|
| Written | 2026-11-22, for earth |
| Owner | a Claude Code agent on **ad-pc** (this PC: 2 × RTX 3070 8 GB, 62 GB RAM, 32 threads) |
| Works with | earth, **interactively**. Check in at every milestone below; don't run the whole plan unattended |
| Repo | a new public repo earth creates, e.g. `Silicon-Melter/llm-serving-2x3070` (ask earth for the name). This brief and the starter kit go in it as the first commit |
| Time | about 2 weeks: week 1 runs, week 2 tuning and write-up |

## Why
earth (third-year Robotics & AI, KMITL) is pitching inference-optimisation internships to two Bangkok companies:
**TokenMe** (an inference provider on its own GPU fleet) and **SIAM.AI Cloud** (GPU cloud with an inference service).
Both emails are on hold until this project has real numbers and a public link. The project has to show
TokenMe's own job description in action: *find where performance is lost, try ways to recover it, measure, and report
what the numbers show*, across the GPU, memory, kernels and serving stack.

So the deliverable is a **public repo with honest, reproducible measurements and a clear write-up**, not a
leaderboard. A negative result explained well (for example "TP over PCIe loses to two replicas, and here is the
all-reduce trace that shows why") is worth as much as a speedup.

## The question
With two consumer GPUs and no NVLink, **when is it worth splitting a model across both cards, and how?**
- For a model that fits on one card: does tensor parallel (TP) or two independent replicas (DP) give more throughput, and which gives lower latency?
- For models that need both cards: TP or pipeline parallel (PP)? Why?
- What does the PCIe link cost per decode step, and can tuning recover it?
- What does each quantisation (INT4 AWQ, FP8 as W8A16) trade in memory, speed and quality?

## Hardware facts (verify, don't assume)
- RTX 3070 = Ampere sm_86, 8 GB. **No native FP8 maths**: vLLM runs FP8 checkpoints as weight-only W8A16 (Marlin), so FP8 saves memory, not compute.
- No NVLink. GeForce drivers usually disable PCIe peer-to-peer, so NCCL may go through host memory. The PCIe link width of each slot (x16/x16, x8/x8 or x16/x4) matters a lot: record it.
- The GPUs are shared lab hardware. CLAUDE.md in ad-research lists T29 work (`lead_full` / `lead_2pool`) as a standing job on both GPUs. earth says the 3070s are free now, but **check `nvidia-smi` and the GPU board before every run**, and stop if a collector or T29 job needs them.

## Starter kit (in this folder; untested)
Written in a cloud session without access to this PC. The scripts pass `bash -n`, but **nothing has run yet, and flag names may differ in the vLLM/SGLang versions you install**. Read each script before running it. Check flags against `--help`, fix what's wrong, and commit the fix with a one-line reason.

| Script | Does |
|---|---|
| `scripts/00_hw_facts.sh` | GPUs, driver, PCIe gen and width, topology, P2P support |
| `scripts/01_setup.sh` | conda envs `serve-vllm` and `serve-sglang`, nccl-tests build, versions to `results/versions.txt` |
| `scripts/02_nccl.sh` | NCCL all-reduce bandwidth, P2P on and forced off |
| `scripts/03_sweep_vllm.sh` | for each config, starts `vllm serve`, sweeps concurrency 1/4/16/64 with `vllm bench serve`, then stops |
| `scripts/04_sweep_sglang.sh` | the same for SGLang (TP and DP) |
| `scripts/05_analyse.py` | `results/summary.csv`, scaling efficiency, cost per 1M tokens, charts |

Model matrix (in `README.md`): `Qwen/Qwen3-8B-AWQ` (fits on 1 GPU: the scaling baseline), `Qwen/Qwen3-8B-FP8`,
`Qwen/Qwen3-14B-AWQ` and `Qwen/Qwen3-4B` BF16 (these need both). A config that OOMs is a result: record it and why.

## Milestones (check in with earth after each)
**M0 — Set up and agree (about 30 min).**
1. Confirm the repo name with earth, create the local clone, and commit this kit as is.
2. Run `00_hw_facts.sh`. Show earth the PCIe width, P2P status and free VRAM, and agree the GPUs are free.
3. Register on the GPU board (gpu-board skill): `board-job run ad-pc.<repo>.<step> --res gpu0 …`, plus a booking for gpu1 if a run uses both (check the skill for how to hold two GPUs).

**M1 — Environment and interconnect (half a day).**
1. `01_setup.sh` (fix as needed), with versions recorded.
2. `02_nccl.sh`. Report the all-reduce bus bandwidth at 1 MB, 16 MB and 256 MB, with P2P on and off. *Check in:* is this PCIe-bound as expected?

**M2 — First end-to-end config (half a day).** Run only `8b-awq_1gpu`, at a single concurrency of 16. Confirm the server starts, the benchmark JSON is written and `05_analyse.py` parses it (fix the key names if they differ). *Check in* with the first numbers before sweeping.

**M3 — vLLM sweep (1 to 2 days, mostly unattended runs).** The full `03_sweep_vllm.sh` matrix, under board-job. While it runs, report progress on the board (`board-job say`), not by reading logs aloud. *Check in:* the table so far, plus anything surprising.

**M4 — Explain (1 day).**
- An `nsys profile` of one decode-heavy run each of TP=2 and 1 GPU. Find the share of a decode step spent in NCCL all-reduce, and screenshot the timeline.
- TP with and without `--disable-custom-all-reduce`.
- Quality: lm-evaluation-harness, GSM8K plus an MMLU subset (fixed seed and sample count) for each checkpoint served. Use the same subset everywhere.
*Check in:* the story in 3 bullets before tuning.

**M5 — SGLang and tuning (2 to 3 days).** `04_sweep_sglang.sh`. Then tune the best config **one change at a time**: CUDA graphs, `--max-num-seqs`, chunked prefill, `--kv-cache-dtype fp8`, PP with more in-flight requests. One row per change, with the reason and the result.

**M6 — Write-up (2 days).** A README with: the question; hardware facts; method (versions, workloads, the price assumed for cost); the results table; 3 to 4 charts (throughput vs concurrency, TPOT, cost per 1M tokens, scaling efficiency); the nsys screenshot; quality deltas; conclusions; and "what changes on NVLink / H100 / B200" (SIAM.AI sells those). Then a headline-numbers block earth can paste into emails: model, method, tokens/s, latency, speedup or efficiency vs 1 GPU, cost per 1M tokens.

## Rules for the numbers
- Every number names its model, precision, engine and version, parallel mode, concurrency, input/output lengths and GPU. Keep the raw JSONs in `results/` and commit them.
- At least 2 repeats for the headline configs. Report the spread, and don't cherry-pick the best run.
- Warm up before measuring (the first request after start-up compiles graphs).
- Write hypotheses down **before** a run, then say whether the numbers agreed.
- Never claim a speedup without the matching quality check.

## Done when
- The public repo has the scripts, raw results, `summary.csv`, charts, nsys screenshot and write-up, and earth has reviewed it.
- A headline-numbers block exists for the TokenMe and SIAM.AI emails.
- All board jobs are finished (`board-job finish`).

## Report (fill in at the end)
_empty_

## Start prompt (earth pastes this into Claude Code on ad-pc)
> Read `BRIEF.md` in this folder: it's your task. We'll work through it together. Start with M0. Show me the hardware facts and ask me for the repo name before running anything heavy. Check in with me after each milestone.
