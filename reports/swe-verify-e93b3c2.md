## swe-verify — llm-serving-2x3070 main @ e93b3c2 (base d55ada8 = origin/main), 2026-11-27, by Claude Opus 5.5 (ad-pc, session b4e95076)
Scope: `git diff --stat d55ada8..e93b3c2 -- scripts tests`, 14 commits.
- **scripts/03_sweep_vllm.sh:** IN / OUTLEN can be overridden.
- **scripts/04_sweep_sglang.sh:** rewritten on 03's rules and client, then two fixes (the flag and the failure reason) and a nopg config.
- **scripts/05_analyse.py:** reads results/sglang/.
- **scripts/13_llamacpp.sh:** new; the build plus llama-bench for T1 / T2.
- **Tests:** tests/test_scripts.py and tests/test_analyse.py.
- **Data:** results for m5f-ctl, m5g, t1, m5h, m5i.

| Layer | Verdict | Command | Evidence (pasted) |
|---|---|---|---|
| 1 Unit | PASS | `env -u LD_LIBRARY_PATH python3 -m pytest tests -q` | `80 passed in 36.54s`, no skips |
| 1 Audit | PASS | `git diff d55ada8..e93b3c2 -- tests/` | **Removed lines:** the old body of the fake-vllm sweep test and the variant-flag map. Every removed assert is back in the new test via the shared `_fake_sweep` / `_after` helpers, and input-length checks were added. **One new skip** fires only if the SGLang env is missing, with its reason; it ran here. **No loosened tolerance.** **Fakes stand in only for external programs:** vllm, the SGLang python, llama-bench and nvidia-smi. The real scripts run end to end. **One test checks against the real installed SGLang's `--help`.** |
| 1 Red-green | PASS | One mutation at a time in a detached worktree at e93b3c2; run the module's tests; restore | **03:** IN fixed at 512 → `FAILED test_sweep_prompt_length_can_be_set`. **05:** vllm only / no SGLang KV regex / engine hard-coded → `FAILED test_sglang_runs_load_with_their_engine_and_kv_size`, each time. **04:** largest rank's KV → `FAILED test_sglang_kv_tokens_takes_the_smallest_rank`; bench_ok ignores failures → `FAILED test_sglang_bench_ok_matches_the_vllm_sweep[…]`; JIT toolkit = venv → `FAILED test_sglang_uses_the_same_jit_toolkit_as_vllm_fp8_kv`; fail_reason takes any error first → `FAILED test_sglang_fail_reason_skips_the_peer_access_warning`; radix cache on and bench backend vllm → `FAILED test_sglang_sweep_starts_servers_and_benches_like_the_vllm_sweep`; old `--cuda-graph-max-bs` → `FAILED test_sglang_flags_exist_in_the_installed_sglang` (`assert ['--cuda-graph-max-bs'] == []`). **13:** T2 on both GPUs / `-r 5` hard-coded / tensor row loses `-sm` → `FAILED test_llamacpp_bench_runs_each_split_on_the_right_cards`. 14 of 14 caught; restored → `80 passed`. |
| 2 E2E | PASS | GPU board jobs m5g-t4, t1-llamacpp2, m5h2-sglang, m5i-sglang-nopg at the commits in their run IDs | Details below the table. |
| 3 Perf | INFORMATIONAL | same runs | Engine and tuning results, not code-speed claims. Repeats agree within 0.1–1.9 %. |
| 4 Numerics | n/a | | no op code |
| 5 Metrics | n/a | | speed only. SGLang FP8-KV quality is **not measured**: same checkpoint and KV dtype as the vLLM run that passed, but a different kernel stack. |

**E2E evidence (layer 2):**
- **03 (m5g):**
  - run.txt says `in 120 out 256`, and the c=32 JSON has `total_input_tokens 30720` over `completed 256`, i.e. 120 per request.
  - The 512 run says `in 512`.
  - 8 bench runs, all OK.
- **13 (t1-c907781):**
  - The first build failed on `nvml.h`, so c907781 adds `cuda-nvml-dev`. The rebuild links `bin/llama-cli`, and `llama-bench --help` lists `<none|layer|row|tensor>`.
  - 7 of 9 benches OK, 5 repeats each, ±0.01–0.3 tok/s on tg.
  - `-sm row` fails to load in llama.cpp a38695013 ("device CUDA0 does not support split buffers"): a llama.cpp limitation, kept as a result.
- **04 (m5h-f572d6e):**
  - 3 configs started: AWQ `ok (0.78): 34361`, W8A8 `ok (0.75): 8676`, FP8 KV `ok (0.75): 17353`. SGLang logs `Using awq_marlin kernel`.
  - DP2 `does not fit`, with SGLang's own reason: "Raise --mem-fraction-static above 0.845".
  - 24 bench runs OK. The first try m5h-08879d0 died on the ambiguous flag, which is fixed and now tested.
- **04 (m5i-8309397):** `--disable-prefill-cuda-graph` → `did not start (0.85): torch.OutOfMemoryError …` (the new failure reason, not the peer-access warning) → `ok (0.82): 31021 KV tokens`; 6 bench runs OK.
- **05:** `05_analyse.py <scratch> --runs m5b-d4821ef m5h-f572d6e m5i-8309397` exits 0 with separate vllm / sglang rows per config.

**Verdict: PASS** — every changed script ran on ad-pc and produced the committed results. Every new check fails when the code it guards is broken. Gaps: no GPU run of 03 since the IN / OUTLEN change other than m5g; the 13 build step is checked by its real run only.
