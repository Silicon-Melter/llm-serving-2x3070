#!/usr/bin/env bash
# SGLang against the same workload as 03_sweep_vllm.sh, measured by the same client (`vllm bench serve`, OpenAI
# completions endpoint), so the two engines' JSONs, keys and analysis are identical. Results: results/sglang/<run>/.
#   RUN_ID=m5h ONLY="8b-awq_tp2" REPEAT=2 bash scripts/04_sweep_sglang.sh
# Same rules as 03: random 512 -> 256 tokens, greedy, ignore EOS, no prefix cache (--disable-radix-cache), at most
# 64 requests running (--max-running-requests 64), 127.0.0.1, server in its own session and killed as a group,
# VRAM checked back to idle after each config.
# KV rule: SGLang sizes weights + KV pool as a fraction of total VRAM (--mem-fraction-static). The largest fraction
# in MEM_STEPS that starts and survives a c=64 burst wins; the KV size it gives is recorded (max_total_num_tokens).
# FlashInfer (SGLang's attention and sampling kernels on sm_86) compiles on first use: same CUDA 13.0 toolkit as
# vLLM's FP8-KV path (scripts/01b_jit_toolkit.sh), for every config.
set -uo pipefail
BULK="${BULK:-/media/earth/SSD2/llm-serving}"; SGL="$BULK/envs/serve-sglang"; VLLM="$BULK/envs/serve-vllm"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$BULK/hf}"
bench_ok() {  # bench_ok <bench log>: at least one request succeeded and none failed (as in 03_sweep_vllm.sh)
  grep -qE "^Successful requests:\s+[1-9]" "$1" || return 1
  if grep -q "^Failed requests:" "$1"; then grep -qE "^Failed requests:\s+0\s*$" "$1"; fi
}
kv_tokens_of() {  # kv_tokens_of <server log>: smallest max_total_num_tokens over ranks / replicas
  grep -oE "max_total_num_tokens=[0-9]+" "$1" | cut -d= -f2 | sort -n | head -1; }
jit_env() { local c="$BULK/cuda-jit-13.0/nvidia/cu13"
  echo "CUDA_HOME=$c PATH=$SGL/bin:$c/bin:$PATH FLASHINFER_WORKSPACE_BASE=$BULK/flashinfer-ws-sgl FLASHINFER_CUDA_ARCH_LIST=8.6"; }
[ "${1:-}" = "--check-bench-log" ] && { bench_ok "$2"; exit $?; }   # hooks used by tests/test_scripts.py
[ "${1:-}" = "--kv-tokens" ] && { kv_tokens_of "$2"; exit 0; }
[ "${1:-}" = "--jit-env" ] && { jit_env; exit 0; }

PORT="${PORT:-30000}"; HOST=127.0.0.1
CONC="${CONC:-1 4 16 64}"; IN="${IN:-512}"; OUTLEN="${OUTLEN:-256}"; REPEAT="${REPEAT:-1}"; SEED=1234
RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"; OUT="results/sglang/$RUN_ID"; mkdir -p "$OUT"
MAX_RUN=64; SMOKE_C=64
MEM_STEPS="${MEM_STEPS:-0.85 0.82 0.80 0.78 0.75 0.72 0.70 0.65 0.60}"
COMMON="--context-length 4096 --max-running-requests $MAX_RUN --cuda-graph-max-bs $MAX_RUN --disable-radix-cache --random-seed $SEED --host $HOST --port $PORT"
CONFIGS=(   # name | GPUs | model | extra args   (names match 03_sweep_vllm.sh for the same setup)
  "8b-awq_tp2|0,1|Qwen/Qwen3-8B-AWQ|--tp-size 2"
  "8b-awq_dp2|0,1|Qwen/Qwen3-8B-AWQ|--dp-size 2"
  "8b-w8a8_tp2|0,1|nm-testing/Qwen3-8B-W8A8-INT8|--tp-size 2"
  "8b-w8a8_tp2_fp8kv|0,1|nm-testing/Qwen3-8B-W8A8-INT8|--tp-size 2 --kv-cache-dtype fp8_e4m3"
)
CPUSET="${CPUSET:-0-19}"

gpu_used() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$1" | tr -d ' '; }
declare -A IDLE; for g in 0 1; do IDLE[$g]=$(gpu_used $g); done
wait_idle() { local i g ok
  for i in $(seq 1 60); do ok=1; for g in 0 1; do [ "$(gpu_used $g)" -le $(( ${IDLE[$g]} + 300 )) ] || ok=0; done
    [ $ok = 1 ] && return 0; sleep 2; done
  echo "  GPU memory did not return to idle: $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' ')"; return 1; }
stop_server() {  # the server runs in its own session: kill the whole process group
  kill -INT -- "-$1" 2>/dev/null; for _ in $(seq 1 30); do kill -0 -- "-$1" 2>/dev/null || break; sleep 1; done
  kill -KILL -- "-$1" 2>/dev/null; wait "$1" 2>/dev/null; wait_idle; }
bench() {  # bench <log> <args...>
  local log="$1"; shift
  taskset -c "$CPUSET" "$VLLM/bin/vllm" bench serve --backend openai --endpoint /v1/completions --model "$MODEL" \
    --host $HOST --port $PORT --dataset-name random --random-range-ratio 0 --ignore-eos --temperature 0 "$@" > "$log" 2>&1 || return 1
  bench_ok "$log"; }
try_start() {  # try_start <mem fraction>: 0 = up and survived the smoke burst (left running)
  local F="$1" up=""
  LOG="$OUT/$NAME.try-$F.log"
  CUDA_VISIBLE_DEVICES=$GPUS setsid taskset -c "$CPUSET" env $(jit_env) "$SGL/bin/python" -m sglang.launch_server \
    --model-path "$MODEL" $COMMON --mem-fraction-static "$F" $EXTRA > "$LOG" 2>&1 &
  SPID=$!
  for _ in $(seq 1 480); do curl -sf "http://$HOST:$PORT/health" >/dev/null && { up=1; break; }
    kill -0 "$SPID" 2>/dev/null || break; sleep 5; done
  if [ -z "$up" ]; then
    echo "  did not start ($F): $(grep -m1 -E 'out of memory|Error|error' "$LOG" | sed 's/^.*\] //' | cut -c1-140)"
    stop_server "$SPID" || exit 1; return 1; fi
  if bench "$OUT/$NAME.try-$F.smoke.log" --random-input-len $IN --random-output-len 16 --num-prompts $((2 * SMOKE_C)) \
       --max-concurrency $SMOKE_C && curl -sf "http://$HOST:$PORT/health" >/dev/null; then
    echo "  ok ($F): $(kv_tokens_of "$LOG") KV tokens"; return 0; fi
  echo "  started but failed the c=$SMOKE_C smoke test ($F): $(grep -m1 -E 'out of memory|Error' "$LOG" | sed 's/^.*\] //' | cut -c1-140)"
  stop_server "$SPID" || exit 1; return 1; }

{ echo "run_id $RUN_ID"; echo "engine sglang $("$SGL/bin/python" -c 'import sglang; print(sglang.__version__)' 2>/dev/null)"
  echo "cpuset $CPUSET"; echo "idle_mib gpu0 ${IDLE[0]} gpu1 ${IDLE[1]}"; echo "common $COMMON"
  echo "in $IN out $OUTLEN seed $SEED"; echo "started $(date -Is) commit $(git rev-parse --short HEAD 2>/dev/null)"; } > "$OUT/run.txt"
for c in "${CONFIGS[@]}"; do
  IFS='|' read -r NAME GPUS MODEL EXTRA <<<"$c"
  [ -n "${ONLY:-}" ] && [[ " $ONLY " != *" $NAME "* ]] && continue
  echo "=== $NAME (GPUs $GPUS)"
  wait_idle || { echo "  GPUs busy before start, stopping"; exit 1; }
  STARTED=""
  for F in $MEM_STEPS; do try_start "$F" && { STARTED="$F"; break; }; done
  if [ -z "$STARTED" ]; then echo "  $NAME: does not fit (no --mem-fraction-static started and survived load), see $OUT/$NAME.try-*.log"; continue; fi
  cp "$OUT/$NAME.try-$STARTED.log" "$OUT/$NAME.log"
  { echo "mem_fraction_static $STARTED"; echo "extra $EXTRA"
    grep -hoE "max_total_num_tokens=[0-9]+|max_running_requests=[0-9]+|chunked_prefill_size=[0-9]+|attention_backend='?[a-z_0-9]+|quantization='?[a-z_0-9-]+|kv_cache_dtype='?[a-z_0-9]+" "$OUT/$NAME.log" | sort -u
  } > "$OUT/$NAME.server.txt"
  for C in $CONC; do
    N=$(( 8 * C > 32 ? 8 * C : 32 ))
    for R in $(seq 1 "$REPEAT"); do
      bench "$OUT/${NAME}_c${C}_r${R}.bench.log" --random-input-len $IN --random-output-len $OUTLEN \
        --num-prompts $N --max-concurrency "$C" --num-warmups "$C" --seed $((SEED + R)) \
        --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99 \
        --metadata engine=sglang config=$NAME gpus=$GPUS mem_fraction_static=$STARTED repeat=$R \
        --save-result --result-dir "$OUT" --result-filename "${NAME}_c${C}_r${R}.json" \
        && echo "  c=$C r=$R: $(grep -m1 -E 'Output token throughput' "$OUT/${NAME}_c${C}_r${R}.bench.log" | tr -s ' ')" \
        || echo "  c=$C r=$R: FAILED, see $OUT/${NAME}_c${C}_r${R}.bench.log"
    done
  done
  stop_server "$SPID" || exit 1
done
echo "done: $OUT"
