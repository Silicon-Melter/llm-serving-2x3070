## swe-verify — multigpu-serving-kit main @ 2aa61b3 (base fec37be), 2026-11-22, by ad-pc/retina b4e95076
Scope: scripts/03_sweep_vllm.sh (581eb68: uniform KV rule, try_start, smoke test, CPU pinning, new configs; 2aa61b3: KV log parsing as functions, automatic size = smallest over ranks), tests/test_scripts.py.

| Layer | Verdict | Command | Evidence (pasted) |
|---|---|---|---|
| 1 Unit | PASS | `env -u LD_LIBRARY_PATH python3 -m pytest tests -q` | `17 passed in 0.32s` (tests at 2aa61b3) |
| 1 Audit | PASS | `git diff 581eb68..2aa61b3 -- tests/` | 24 lines added, no skip/xfail, no assertion removed or loosened; fixtures are real log lines from results/vllm/m3-581eb68/8b-awq_tp2.try-auto.log |
| 1 Red-green | PASS | restore `tail -1` in kv_auto → test → restore | `FAILED tests/test_scripts.py::test_sweep_kv_parse_takes_smallest_rank[rank0-first]` / `17 passed` |
| 2 E2E | PASS | 581eb68: full M3 sweep (board job m3-sweep); 2aa61b3: `RUN_ID=e2e-2aa61b3 ONLY=8b-awq_tp2 CONC=1 bash scripts/03_sweep_vllm.sh` (board job e2e-2aa61b3) | M3: 11 of 13 configs measured, 2 recorded as "does not fit" with logs, `done: results/vllm/m3-581eb68`, exit 0. E2E 2aa61b3: same KV choice as M3 (`ok (0.95 …): GPU KV cache size: 46,464 tokens`), `c=1 r=1: Output token throughput (tok/s): 109.09` (M3: 110.19 / 110.22), exit 0, GPUs back to idle |
| 3 Perf | INFORMATIONAL | — | the script is the benchmark; no speed claim about the script itself |
| 4 Numerics | n/a | | no CUDA op |
| 5 Metrics | n/a | | no model output changes |

**Verdict: PASS** — the KV rule starts every config that fits, and the parser now sizes from the tightest GPU regardless of log order. Gaps: when the automatic attempt crashes before printing a KV suggestion (8b-awq_dp2 in M3), the rule has no fallback and declares "does not fit"; fix and rerun planned for H1.
