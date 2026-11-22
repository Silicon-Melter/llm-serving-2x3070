## swe-verify — llm-serving-2x3070 main @ c1c1319 (base 711e5d0), 2026-11-22, by Claude Opus 5.5 (ad-pc, session b4e95076)
Scope: `git diff --stat 711e5d0..c1c1319` → scripts/01_setup.sh, scripts/02_nccl.sh, tests/test_scripts.py (new), results/ (data).

| Layer | Verdict | Command | Evidence (pasted) |
|---|---|---|---|
| 1 Unit | PASS | `env -u LD_LIBRARY_PATH python3 -m pytest tests -q` | `6 passed in 0.03s` (bash -n on 5 scripts + library resolution of all_reduce_perf in 02_nccl.sh's environment) |
| 1 Audit | PASS | `git diff 711e5d0..c1c1319 -- tests/` | tests/ is new: no skips added except an explicit skip-with-reason when the envs are absent; no mocks; no loosened asserts; no oracle pasted from output |
| 1 Red-green | PASS | remove `$CUDART_LIB` from the export → pytest → restore → pytest | `AssertionError: unresolved libraries: ['libcudart.so.13 => not found']` / `6 passed in 0.03s` |
| 2 E2E | PASS | `board-job run … verify-setup` and `verify-setup-2` (after `make clean` of nccl-tests): `bash scripts/01_setup.sh`; `board-job run … m1-nccl-run2 -- bash scripts/02_nccl.sh` | setup: `exit 0`, versions `vllm 0.34.0 torch 2.14.0+cu130 cuda 13.0 nccl (2, 30, 7)` / `sglang 0.5.23 torch 2.14.1+cu130`, all_reduce_perf rebuilt from clean (7,075,128 B); nccl: `exit 0`, results/nccl/20261122-130707/, `8192 B 11.62 us`, `268435456 B busbw 7.63`, transport `via SHM/direct`, PCIe `4, 8` under load. Earlier red evidence on real runs: old 01_setup.sh → `unsupported GNU version! gcc versions later than 12`; old 02_nccl.sh → `libcudart.so.13: cannot open shared object file` |
| 3 Perf | n/a | | measurement scripts; no speed claim about this code |
| 4 Numerics | n/a | | no CUDA/Triton op |
| 5 Metrics | n/a | | no model output change |
**Verdict: PASS** — both scripts run end to end on ad-pc at this commit and the library fix is covered by a test that fails without it. Gaps: 01_setup.sh's first-time env creation path was exercised only once (in the original run at fc90309's parent), not from an empty SSD2.
