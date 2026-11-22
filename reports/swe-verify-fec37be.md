## swe-verify — llm-serving-2x3070 main @ fec37be (base ce10134), 2026-11-22, by Claude Opus 5.5 (ad-pc, session b4e95076)
Scope: `git diff --stat ce10134..fec37be` → scripts/03_sweep_vllm.sh, scripts/05_analyse.py, tests/test_scripts.py, tests/test_analyse.py (new), .gitignore, results/ (data).

| Layer | Verdict | Command | Evidence (pasted) |
|---|---|---|---|
| 1 Unit | PASS | `env -u LD_LIBRARY_PATH python3 -m pytest tests -q` | `14 passed in 0.27s` |
| 1 Audit | PASS | `git diff ce10134..fec37be -- tests/` | only additions: 6 bench-log cases, 2 analysis tests on synthetic files; no skips, no mocks of the code under test, no loosened asserts; expected values computed by hand in the test (350 = mean(340,360); 660/(2·300)) |
| 1 Red-green | PASS | (a) bench_ok `[1-9]`→`[0-9]`; (b) analyse failed-run filter → `if False:`; restore each | (a) `FAILED test_sweep_bench_ok[zero-no-failed-line]` (b) `FAILED test_load_filters_and_parses - assert 5 == 4`, `FAILED test_summarise_repeats_and_efficiency`; restored: `14 passed` |
| 2 E2E | PASS | `board-job run … m2-verify -- env ONLY=8b-awq_1gpu CONC=16 REPEAT=2 RUN_ID=m2-fec37be bash scripts/03_sweep_vllm.sh`; `python3 scripts/05_analyse.py results/ --no-charts` | auto sizing failed → 1.05 and 1.00 GiB failed the c=64 smoke test (OOM) → 0.94 GiB accepted; `c=16 r=1: 362.88`, `c=16 r=2: 362.53` tok/s; GPUs back to idle; analysis: n=3, kv_tokens 6848, out_tok_s 363 |
| 3 Perf | INFORMATIONAL | same run | 363 tok/s ±0.1 % (3 runs); no stored baseline yet: this run is the baseline |
| 4 Numerics | n/a | | no CUDA/Triton op |
| 5 Metrics | n/a | | serving config only; quality evals are M4 |
**Verdict: PASS** — sweep script and analysis work end to end on ad-pc at this commit; both new checks fail when broken. Gaps: TP/PP/DP configs and the "does not fit" path not yet exercised (M3).
