## swe-verify — llm-serving-2x3070 main @ 5a5517b (base bfaf72d = origin/main), 2026-11-27, by Claude Opus 5.5 (ad-pc, session b4e95076)
Scope: `git diff --stat bfaf72d..5a5517b -- scripts tests`, 27 commits, 16 code files.
- **Scripts:** 01_setup.sh, 01b_jit_toolkit.sh (new), 03_sweep_vllm.sh, 06_quality.sh, 07_kld.py, 08_quality_compare.py, 09_roofline.py, 10_profile_tp.sh, 11_trace_breakdown.py, 12_frontier_share.py.
- **Tests:** test_frontier_share.py, test_kld.py, test_quality_compare.py, test_roofline.py, test_scripts.py, test_trace_breakdown.py.
- **Also changed, not code:** results/ (data), README.md, reports/.

| Layer | Verdict | Command | Evidence (pasted) |
|---|---|---|---|
| 1 Unit | PASS | `env -u LD_LIBRARY_PATH python3 -m pytest tests -q -rs` | `63 passed in 1.96s` (no skips: the NCCL library test ran) |
| 1 Audit | PASS | `git diff bfaf72d..5a5517b -- tests/` | **No test lines removed;** no new skip, xfail or mark other than one `parametrize`. **Tolerances** are `approx` default, `rel=1e-6` on exact maths, and `rel=0.02` with a comment on two physical estimates (4.93e9 B read, 5.71 GiB loaded vs the server log). **Expected values** are worked by hand in the tests, e.g. KL vs an exact sum and McNemar 2/1024. **No mock of the code under test:** the 06 exit tests fake only `vllm` (exits at once) and `nvidia-smi` (idle), and run the real script. |
| 1 Red-green | PASS | One mutation at a time in a detached worktree; run the module's tests; restore | **07:** tail-mass term dropped → `1 failed`. q_floor fallback → `lp` survived (`5 passed`); test added in 535f67b, after which it fails (`1 failed, 5 passed`). **08:** McNemar `2*tail`→`tail` → `1 failed`; filter check → `if False` → `3 failed`; gz files dropped → `1 failed`. **09:** batch not KV-capped → `2 failed`. **03:** KV pick `head`→`tail` → `2 failed`; kv_env CUDA_HOME back to the venv → `2 failed`; FLASHINFER pattern dropped → `2 failed`. **06:** kv_env differs from 03 → `1 failed`; FLASHINFER pattern dropped → `1 failed`; gz samples ignored → `1 failed`; MMLU `-ge 57`→`-ge 1` → `2 failed`; final exit dropped → `1 failed`; no check after "did not start" → `1 failed`; always exit 1 → `1 failed`. **01b:** toolkit dir renamed → `1 failed`. **11:** Inductor-elementwise rule off → `2 failed`; `^triton_tem_` dropped → `1 failed`; prefill split off → `1 failed`. GEMM before attention (the 15233b2 bug) → `FAILED test_classify[void flash::flash_fwd_splitkv_kernel<…cutlass::half_t…>-attention]`. The batch harness first reported that one as surviving because its swap script failed unchecked; rerun alone it fails as shown. **12:** batched ceiling on our card instead of the source card → `2 failed`; single-stream `cards×` dropped → `2 failed`. Every restore → module tests pass; full suite `63 passed`. |
| 2 E2E | PASS | GPU runs via `board-job run` (m5a-baseline, m5b2-w8a8-fp8kv, m5b3-fp8kv-mmlu-retry, m5c-w8a8-fi-control), plus CPU reruns on committed data | Details below the table. |
| 3 Perf | INFORMATIONAL | same GPU runs | No perf baseline file; the stored baseline is the M3 summary. m5a reproduces it within 0.5 %, except 8b-awq_tp2 c=64 at −2.0 % (KV 44,928 vs 46,464, from more display memory on GPU0). Repeats agree within 0.2 %. |
| 4 Numerics | n/a | | No CUDA or Triton op written here; FlashInfer kernels are third-party |
| 5 Metrics | PASS | `08_quality_compare.py --ref ref8b-bf16_tp2 --test 8b-w8a8_tp2_fp8kv`; `07_kld.py compare` | GSM8K −0.45 [−1.82, +0.91], MMLU +0.61 [−0.61, +1.75], `passes True`; mean KL 0.0391, top-1 90.6 % (W8A8 BF16 KV: 0.0338 / 91.2 %) |

**E2E evidence (layer 2):**
- **01b:** a fresh install from an empty directory printed `flashinfer jit toolkit …/cuda-jit-13.0/nvidia/cu13 (nvcc 13.0.88, cudart 13.0.96)`. Its file tree is `SAME-TREE` as the hand-built one, and a second run is a no-op.
- **FlashInfer builds with that toolkit:** sampling, batch decode and fa2 prefill (bf16 q, fp8_e4m3 KV, head dim 128) each print `BUILT`.
- **03 (sweep):**
  - m5a-6b95fd4: 4 configs `ok`, 2 repeats each.
  - m5b-d4821ef: `Using FLASHINFER attention backend` and `ok (0.80 …): GPU KV cache size: 35,776 tokens`; c=64 at 1086.0 tok/s (r1 1086.0, r2 1086.0).
  - m5c-f3d1502: `ok (0.80 …): 17,856 tokens`, c=64 at 792.6.
  - The earlier m5b-c37d57e failure (nvcc 13.5 vs CCCL) is what 01b fixes.
- **06 (quality):**
  - m5b2: `kld: collected (19 s)`; `gsm8k … 0.8749 (358 s)`; `mmlu: FAILED`, because the server hit a CUDA illegal memory access at 28 %.
  - m5b3, a resume with the same RUN_ID: `stages: mmlu` only, `mmlu … 0.7807 (813 s)`.
  - After gzipping the samples: stage_done stays true for kld, gsm8k and mmlu, and 08 gives identical output.
  - The exit-1 path is unit-tested only (fake server); the m5b2 job, run before 250d584, is the case it fixes.
- **07 / 08:** real compares as in layer 5; the 08 output is identical on plain and gzipped samples.
- **09 / 12 / 05:** rebuilt from the committed M3 runs, `summary.csv`, `roofline.csv` and `frontier_share.csv` are `identical` to the committed files.
- **11:** rerun on committed traces (h3b-18e63e5/c64 rank0+rank1, `--by-step`), `breakdown.json` is `identical`.
- **10_profile_tp.sh:** produced the committed h3-fa9897e and h3b-18e63e5 traces; it has no unit test beyond the syntax check.

**Verdict: PASS** — every changed script ran on ad-pc at this commit or produced the committed outputs it is responsible for. Every new check fails when the code it guards is broken. FP8 KV for W8A8 TP=2 passes the quality gate.

Gaps:
- 10_profile_tp.sh has only E2E coverage.
- The 06 exit-1 path has not run on a real failed server since the change.
- The CUDA illegal memory access during MMLU with FlashInfer FP8 KV happened once in two attempts. It is not reproduced or localised; it is a reliability finding, not covered by this verdict.
