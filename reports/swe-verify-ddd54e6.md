## swe-verify — llm-serving-2x3070 main @ ddd54e6 (base d55ada8 = origin/main), 2026-11-27, by Claude Opus 5.5 (ad-pc, session b4e95076)
Scope: `git diff --stat d55ada8..ddd54e6 -- scripts tests`:
- scripts/03_sweep_vllm.sh: +3 config lines, `8b-w8a8_tp2_fp8kv_mbt512/1024/4096`.
- tests/test_scripts.py: VARIANT_FLAG gains the three `_mbtN` suffixes.
- Also results/vllm/m5f-0ad3ae2 and m5f-ctl-0ad3ae2 (data).

| Layer | Verdict | Command | Evidence (pasted) |
|---|---|---|---|
| 1 Unit | PASS | `env -u LD_LIBRARY_PATH python3 -m pytest tests -q` | `68 passed in 14.16s` |
| 1 Audit | PASS | `git diff d55ada8..ddd54e6 -- tests/` | The variant-flag map gains 3 entries, which widens the one-flag-per-variant rule; the docstring is updated. Nothing removed or loosened. |
| 1 Red-green | PASS | Mutate one new row at a time in a detached worktree at 0ad3ae2, run `-k config_tables`, restore | Each mutation gives `FAILED test_config_tables_change_one_thing_per_variant[03_sweep_vllm.sh]`: the 512 row carries `--max-num-batched-tokens 1024`; the 4096 row drops `--kv-cache-dtype fp8`; the 1024 row runs on GPU 1 alone. Restored: `68 passed`. |
| 2 E2E | PASS | `board-job run ad-pc.llm-serving-2x3070.m5f-prefill-chunk -- env CPUSET=0-2,4-18 RUN_ID=m5f-0ad3ae2 REPEAT=2 CONC="16 64" ONLY="8b-w8a8_tp2_fp8kv_mbt512 8b-w8a8_tp2_fp8kv_mbt1024 8b-w8a8_tp2_fp8kv_mbt4096" bash scripts/03_sweep_vllm.sh` at 0ad3ae2; the control `m5f-control` (default chunk) right after | All 3 configs `ok`, with KV 36,160 / 36,160 / 31,488 tokens; the 4096 row failed the c=64 smoke test twice (OOM) before `ok (0.75 …)`. vLLM's `non-default args` include `'max_num_batched_tokens': 512`. All 12 bench runs OK; job `exit 0`. Control: `ok (0.80 …): 35,776 tokens`, c=64 1089.46 / 1089.49. |
| 3 Perf | INFORMATIONAL | same runs | c=64 against the same-session control (1089.5): 1005.9 / 993.2 / 1000.3 tok/s for 512 / 1024 / 4096. A tuning result, not a code-speed claim. |
| 4 Numerics | n/a | | no op code |
| 5 Metrics | n/a | | speed-only configs |
**Verdict: PASS** — the three config rows run end to end and each differs from its twin by exactly the one flag the test requires. Gaps: none for this change.
