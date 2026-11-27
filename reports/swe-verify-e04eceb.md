## swe-verify — llm-serving-2x3070 main @ e04eceb (base 2b1c1b1 = origin/main), 2026-11-27, by Claude Opus 5.5 (ad-pc, session b4e95076)
Scope: `git diff --stat 2b1c1b1..e04eceb -- scripts tests`:
- scripts/03_sweep_vllm.sh: +1 config line, `8b-awq_tp2_fp8kv`.
- tests/test_scripts.py: +30 lines, a new config-table test.
- Also results/vllm/m5d-009667d (data).

| Layer | Verdict | Command | Evidence (pasted) |
|---|---|---|---|
| 1 Unit | PASS | `env -u LD_LIBRARY_PATH python3 -m pytest tests -q` | `65 passed in 2.02s` |
| 1 Audit | PASS | `git diff 2b1c1b1..e04eceb -- tests/` | **Additions only.** No skip, xfail, tolerance or mock. **The config table is read by bash itself** (`eval` of the `CONFIGS=( … )` block), so no hand-written parser can drift from the script. **The test's expectations are written in the test:** every row has 4 fields, names are unique, GPUs are one of 0 / 1 / 0,1, and each `_fp8kv` / `_fi` row is its twin plus exactly one flag. |
| 1 Red-green | PASS | Mutate the new 03 line in a detached worktree, run `-k config_tables`, restore | Each mutation gives `FAILED test_config_tables_change_one_thing_per_variant[03_sweep_vllm.sh]` (1 failed, 1 passed): fp8 flag dropped; model changed to Qwen/Qwen3-8B; a second flag added (`--max-num-seqs 128`); a duplicate name (8b-w8a8_tp2_fi renamed to 8b-w8a8_tp2_fp8kv); a 3-field row. Restored: `65 passed`. |
| 2 E2E | PASS | `board-job run ad-pc.llm-serving-2x3070.m5d-awq-fp8kv -- env CPUSET=0-2,4-18 RUN_ID=m5d-009667d REPEAT=2 ONLY="8b-awq_tp2_fp8kv" bash scripts/03_sweep_vllm.sh` at 009667d (same 03 as e04eceb) | `ok (auto): GPU KV cache size: 30,720 tokens` → `did not start (0.95 …): torch.OutOfMemoryError` → `ok (0.90, --kv-cache-memory-bytes 3027045888 …): GPU KV cache size: 82,112 tokens`; `Using FLASHINFER attention backend`; 8 bench JSONs; c=64 r1/r2 1025.0 / 1025.5 tok/s; job `exit 0`. 05_analyse over m5a/m5b/m5c/m5d in a scratch dir: exit 0. |
| 3 Perf | INFORMATIONAL | same run | c=1 110.0 (AWQ BF16 KV 110.0), c=64 1025.2 (877.4); repeats within 0.05 %. A new config, so no baseline. |
| 4 Numerics | n/a | | no op code |
| 5 Metrics | n/a | | speed-only config; AWQ quality was measured in M4 (fails the gate), and FP8-KV quality is not claimed for AWQ |
**Verdict: PASS** — the new config runs end to end on ad-pc, and the new test fails on each way the table could break the one-change-at-a-time rule. Gaps: 06_quality's table is tested by the same check but not rerun here (unchanged).
