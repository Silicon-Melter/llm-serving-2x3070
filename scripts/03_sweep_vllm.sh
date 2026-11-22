#!/usr/bin/env bash
# For each config: start `vllm serve`, wait for /health, sweep concurrency with `vllm bench serve`, stop, and check the
# GPUs are back to idle before the next config. Flags checked against vLLM 0.34.0 (`vllm serve --help=all`).
#   ONLY="8b-awq_1gpu" CONC="16" bash scripts/03_sweep_vllm.sh     # one config, one concurrency (M2)
#   REPEAT=2 bash scripts/03_sweep_vllm.sh                         # full matrix, 2 repeats
# Output: results/vllm/<run-id>/<name>_c<C>_r<k>.json, <name>.server.txt (KV size, kernels), <name>.log
set -uo pipefail
bench_ok() {  # bench_ok <bench log>: at least one request succeeded and none failed
  grep -qE "^Successful requests:\s+[1-9]" "$1" || return 1
  if grep -q "^Failed requests:" "$1"; then grep -qE "^Failed requests:\s+0\s*$" "$1"; fi
}
[ "${1:-}" = "--check-bench-log" ] && { bench_ok "$2"; exit $?; }   # used by tests/test_scripts.py
BULK="${BULK:-/media/earth/SSD2/llm-serving}"; VENV="$BULK/envs/serve-vllm"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$BULK/hf}"
PORT=8000; HOST=127.0.0.1                     # 127.0.0.1, not localhost (name lookup adds latency)
CONC="${CONC:-1 4 16 64}"; IN=512; OUTLEN=256; REPEAT="${REPEAT:-1}"; SEED=1234
RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"; OUT="results/vllm/$RUN_ID"; mkdir -p "$OUT"
# Prefix caching off: random prompts must not hit a cache. Same max length everywhere so KV budgets compare.
# --max-num-seqs 64 = the highest concurrency tested; vLLM's default (256) reserves sampler memory for 256 requests.
COMMON="--max-model-len 4096 --max-num-seqs 64 --disable-prefix-caching --port $PORT --host $HOST"
# KV sizing (vLLM 0.34 extensible KV cache): try vLLM's automatic sizing first. On an 8 GB card it can fail
# ("0 KV cache blocks fit"); then step down from the log's "--kv-cache-memory=<X> ... to fully utilize" suggestion
# in 5 % steps until the server starts, and record the size used. (Explicit --gpu-memory-utilization is not used:
# M2 probes showed it double-counts with the extensible KV cache; results/vllm/m2-20261122/probe/.)
KV_STEPS="${KV_STEPS:-0.95 0.90 0.85 0.80 0.75 0.70 0.60 0.50}"
# name | GPUs | model | extra args       (1-GPU configs use GPU1: GPU0 also drives the display)
CONFIGS=(
  "8b-awq_1gpu|1|Qwen/Qwen3-8B-AWQ|"
  "8b-awq_tp2|0,1|Qwen/Qwen3-8B-AWQ|--tensor-parallel-size 2"
  "8b-awq_tp2_nocustomar|0,1|Qwen/Qwen3-8B-AWQ|--tensor-parallel-size 2 --disable-custom-all-reduce"
  "8b-awq_pp2|0,1|Qwen/Qwen3-8B-AWQ|--pipeline-parallel-size 2"
  "8b-awq_dp2|0,1|Qwen/Qwen3-8B-AWQ|--data-parallel-size 2"
  "8b-fp8_tp2|0,1|Qwen/Qwen3-8B-FP8|--tensor-parallel-size 2"
  "8b-fp8_pp2|0,1|Qwen/Qwen3-8B-FP8|--pipeline-parallel-size 2"
  "14b-awq_tp2|0,1|Qwen/Qwen3-14B-AWQ|--tensor-parallel-size 2"
  "14b-awq_pp2|0,1|Qwen/Qwen3-14B-AWQ|--pipeline-parallel-size 2"
  "4b-bf16_tp2|0,1|Qwen/Qwen3-4B|--tensor-parallel-size 2"
)

gpu_used() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$1" | tr -d ' '; }
declare -A IDLE; for g in 0 1; do IDLE[$g]=$(gpu_used $g); done
wait_idle() {  # every GPU within 300 MiB of its idle level measured at script start, or fail
  for i in $(seq 1 60); do
    local ok=1; for g in 0 1; do [ "$(gpu_used $g)" -le $(( ${IDLE[$g]} + 300 )) ] || ok=0; done
    [ $ok = 1 ] && return 0; sleep 2
  done
  echo "  GPU memory did not return to idle: $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' ')"; return 1
}
stop_server() {  # the server runs in its own session: kill the whole process group (engine workers included)
  kill -INT -- "-$1" 2>/dev/null; for i in $(seq 1 30); do kill -0 -- "-$1" 2>/dev/null || break; sleep 1; done
  kill -KILL -- "-$1" 2>/dev/null; wait "$1" 2>/dev/null; wait_idle
}

bench() {  # bench <logfile> <args...>: run vllm bench serve; succeed only if every request succeeded
  local log="$1"; shift
  "$VENV/bin/vllm" bench serve --backend vllm --model "$MODEL" --host $HOST --port $PORT --dataset-name random \
    --random-range-ratio 0 --ignore-eos --temperature 0 "$@" > "$log" 2>&1 || return 1
  bench_ok "$log"
}
SMOKE_C=64   # a KV size is accepted only if the server also survives a burst at the highest concurrency tested
{ echo "run_id $RUN_ID"; cat results/versions.txt 2>/dev/null; echo "common $COMMON"; echo "in $IN out $OUTLEN seed $SEED"; } > "$OUT/run.txt"
for c in "${CONFIGS[@]}"; do
  IFS='|' read -r NAME GPUS MODEL EXTRA <<<"$c"
  [ -n "${ONLY:-}" ] && [[ " $ONLY " != *" $NAME "* ]] && continue
  echo "=== $NAME (GPUs $GPUS)"
  "$VENV/bin/hf" download "$MODEL" >/dev/null 2>&1 || "$VENV/bin/huggingface-cli" download "$MODEL" >/dev/null   # outside the health timeout
  wait_idle || { echo "  GPUs busy before start, stopping"; exit 1; }
  STARTED=""; SUGGEST=""; ATTEMPTS="auto $KV_STEPS"
  for A in $ATTEMPTS; do
    if [ "$A" = auto ]; then KVARG=""; else
      [ -n "$SUGGEST" ] || break
      KVARG="--kv-cache-memory $(python3 -c "print(int($SUGGEST*$A))")"; fi
    LOG="$OUT/$NAME.try-$A.log"
    CUDA_VISIBLE_DEVICES=$GPUS setsid "$VENV/bin/vllm" serve "$MODEL" $COMMON $KVARG $EXTRA > "$LOG" 2>&1 &
    SPID=$!
    for i in $(seq 1 180); do
      curl -sf "http://$HOST:$PORT/health" >/dev/null && { STARTED="$A"; break; }
      kill -0 "$SPID" 2>/dev/null || break; sleep 5
    done
    if [ -n "$STARTED" ]; then   # started: now the load smoke test (prefill peak at SMOKE_C concurrent requests)
      bench "$OUT/$NAME.try-$A.smoke.log" --random-input-len $IN --random-output-len 16 --num-prompts $((2 * SMOKE_C)) \
        --max-concurrency $SMOKE_C && curl -sf "http://$HOST:$PORT/health" >/dev/null && break
      echo "  started but failed the c=$SMOKE_C smoke test ($A${KVARG:+, $KVARG}): $(grep -m1 -E 'out of memory|Error' "$LOG" | sed 's/^.*\] //' | cut -c1-140)"
      STARTED=""; SMOKEFAIL=1
    fi
    # step-down base: vLLM's "to fully utilize" suggestion, else the KV size it allocated in this attempt
    [ -z "$SUGGEST" ] && SUGGEST=$(grep -oE -- "--kv-cache-memory=[0-9]+\` \([0-9.]+ GiB\) to fully utilize" "$LOG" | grep -oE "[0-9]{6,}" | sort -n | head -1)
    [ -z "$SUGGEST" ] && SUGGEST=$(grep -oE "Available KV cache memory: [0-9.]+ GiB" "$LOG" | tail -1 | grep -oE "[0-9.]+" | python3 -c "import sys; v=sys.stdin.read().strip(); print(int(float(v)*2**30) if v else '')")
    [ -z "${SMOKEFAIL:-}" ] && echo "  did not start ($A${KVARG:+, $KVARG}): $(grep -m1 -E 'No available memory|out of memory|Error' "$LOG" | sed 's/^.*\] //' | cut -c1-140)"
    SMOKEFAIL=""; stop_server "$SPID" || exit 1
  done
  if [ -z "$STARTED" ]; then echo "  $NAME: does not fit (no KV size started), see $OUT/$NAME.try-*.log"; continue; fi
  mv "$OUT/$NAME.try-$STARTED.log" "$OUT/$NAME.log"
  { echo "kv_sizing $STARTED ${KVARG:-auto}"; echo "extra $EXTRA"
    grep -hE "GPU KV cache size|Maximum concurrency|Available KV cache memory|MarlinLinearKernel|Graph capturing finished|quantization|custom.?all.?reduce|CUDA graph|Model loading took" "$OUT/$NAME.log" | sed 's/^.*\] //' | sort -u
  } > "$OUT/$NAME.server.txt"
  for C in $CONC; do
    N=$(( 8 * C > 32 ? 8 * C : 32 ))
    for R in $(seq 1 "$REPEAT"); do
      bench "$OUT/${NAME}_c${C}_r${R}.bench.log" --random-input-len $IN --random-output-len $OUTLEN \
        --num-prompts $N --max-concurrency "$C" --num-warmups "$C" --seed $((SEED + R)) \
        --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99 \
        --metadata config=$NAME gpus=$GPUS kv_sizing=$STARTED repeat=$R \
        --save-result --result-dir "$OUT" --result-filename "${NAME}_c${C}_r${R}.json" \
        && echo "  c=$C r=$R: $(grep -m1 -E 'Output token throughput' "$OUT/${NAME}_c${C}_r${R}.bench.log" | tr -s ' ')" \
        || echo "  c=$C r=$R: FAILED ($(grep -m1 -E "^Failed requests" "$OUT/${NAME}_c${C}_r${R}.bench.log" | tr -s " ")), see $OUT/${NAME}_c${C}_r${R}.bench.log"
    done
  done
  stop_server "$SPID" || exit 1
done
echo "done: $OUT"
