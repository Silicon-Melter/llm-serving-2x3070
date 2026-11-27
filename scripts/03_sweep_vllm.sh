#!/usr/bin/env bash
# For each config: start `vllm serve`, wait for /health, sweep concurrency with `vllm bench serve`, stop, and check the
# GPUs are back to idle before the next config. Flags checked against vLLM 0.34.0 (`vllm serve --help=all`).
#   ONLY="8b-awq_1gpu" CONC="16" bash scripts/03_sweep_vllm.sh     # one config, one concurrency (M2)
#   REPEAT=2 bash scripts/03_sweep_vllm.sh                         # full matrix, 2 repeats
# Output: results/vllm/<run-id>/<name>_c<C>_r<k>.json, <name>.server.txt (KV size, kernels), <name>.log
set -uo pipefail
BULK="${BULK:-/media/earth/SSD2/llm-serving}"; VENV="$BULK/envs/serve-vllm"
bench_ok() {  # bench_ok <bench log>: at least one request succeeded and none failed
  grep -qE "^Successful requests:\s+[1-9]" "$1" || return 1
  if grep -q "^Failed requests:" "$1"; then grep -qE "^Failed requests:\s+0\s*$" "$1"; fi
}
kv_suggest() {  # kv_suggest <server log>: smallest "--kv-cache-memory=X ... to fully utilize" over all ranks (bytes)
  grep -oE -- "--kv-cache-memory=[0-9]+\` \([0-9.]+ GiB\) to fully utilize" "$1" | grep -oE "[0-9]{6,}" | sort -n | head -1
}
kv_auto() {  # kv_auto <server log>: KV size automatic sizing gave, smallest over all ranks (bytes; 0 if none)
  grep -oE "KV cache blocks \([0-9.]+ GiB\) fit" "$1" | grep -oE "[0-9.]+" | sort -g | head -1 |
    python3 -c "import sys; v=sys.stdin.read().strip(); print(int(float(v)*2**30) if v else 0)"
}
# kv_env <extra args>: environment for FP8 KV configs only. On sm_86 vLLM serves FP8 KV through FlashInfer, whose
# kernels are compiled on first use: it needs CUDA_HOME (a CUDA 13.0 toolkit matching torch's cudart, built by
# scripts/01b_jit_toolkit.sh; the venv's nvcc 13.5 fails CCCL's version check, /usr/local/cuda is 12.3) and ninja on
# PATH. Other configs run with the plain environment.
kv_env() {
  case " $1 " in *" --kv-cache-dtype fp8"*|*" --attention-backend FLASHINFER"*)
    local c="$BULK/cuda-jit-13.0/nvidia/cu13"   # built by scripts/01b_jit_toolkit.sh
    echo "CUDA_HOME=$c PATH=$VENV/bin:$c/bin:$PATH FLASHINFER_WORKSPACE_BASE=$BULK/flashinfer-ws FLASHINFER_CUDA_ARCH_LIST=8.6" ;;
  esac; }
[ "${1:-}" = "--check-bench-log" ] && { bench_ok "$2"; exit $?; }   # used by tests/test_scripts.py
[ "${1:-}" = "--parse-kv-log" ] && { echo "$(kv_suggest "$2") $(kv_auto "$2")"; exit 0; }
[ "${1:-}" = "--kv-env" ] && { kv_env "$2"; exit 0; }
# seqs_of <extra args>: the config's own --max-num-seqs, else 64 (= the highest concurrency in the default CONC)
seqs_of() { local s; s=$(grep -oE -- "--max-num-seqs [0-9]+" <<<"$1" | tail -1 | cut -d" " -f2); echo "${s:-64}"; }
# seq_arg <extra args>: what to add to COMMON: --max-num-seqs 64 unless the config sets its own
seq_arg() { [[ " $1 " == *" --max-num-seqs "* ]] || echo "--max-num-seqs $(seqs_of "$1")"; }
[ "${1:-}" = "--seqs" ] && { seqs_of "$2"; exit 0; }
[ "${1:-}" = "--seq-arg" ] && { seq_arg "$2"; exit 0; }
export HF_HUB_CACHE="${HF_HUB_CACHE:-$BULK/hf}"
PORT="${PORT:-8000}"; HOST=127.0.0.1                     # 127.0.0.1, not localhost (name lookup adds latency)
CONC="${CONC:-1 4 16 64}"; IN=512; OUTLEN=256; REPEAT="${REPEAT:-1}"; SEED=1234
RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"; OUT="results/vllm/$RUN_ID"; mkdir -p "$OUT"
# Prefix caching off: random prompts must not hit a cache. Same max length everywhere so KV budgets compare.
# --max-num-seqs 64 = the highest concurrency tested; vLLM's default (256) reserves sampler memory for 256 requests.
# A config may set its own --max-num-seqs (the *_s128 configs); the smoke test then runs at that concurrency.
COMMON="--max-model-len 4096 --disable-prefix-caching --port $PORT --host $HOST"
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
  "8b-w8a8_tp2|0,1|nm-testing/Qwen3-8B-W8A8-INT8|--tensor-parallel-size 2"
  "4b-awq_1gpu|1|Qwen/Qwen3-4B-AWQ|"
  "4b-awq_dp2|0,1|Qwen/Qwen3-4B-AWQ|--data-parallel-size 2"
  "4b-awq_tp2|0,1|Qwen/Qwen3-4B-AWQ|--tensor-parallel-size 2"
  "4b-fp8_1gpu|1|Qwen/Qwen3-4B-FP8|"                              # M5: passes the quality gate; speed not yet measured
  "4b-fp8_dp2|0,1|Qwen/Qwen3-4B-FP8|--data-parallel-size 2"
  "8b-w8a8_tp2_fp8kv|0,1|nm-testing/Qwen3-8B-W8A8-INT8|--tensor-parallel-size 2 --kv-cache-dtype fp8"   # M5 step 2
  "8b-w8a8_tp2_fi|0,1|nm-testing/Qwen3-8B-W8A8-INT8|--tensor-parallel-size 2 --attention-backend FLASHINFER"   # control: FlashInfer, BF16 KV
  "8b-awq_tp2_fp8kv|0,1|Qwen/Qwen3-8B-AWQ|--tensor-parallel-size 2 --kv-cache-dtype fp8"   # H4 at equal KV dtype (vs 8b-w8a8_tp2_fp8kv)
  "8b-w8a8_tp2_fp8kv_s128|0,1|nm-testing/Qwen3-8B-W8A8-INT8|--tensor-parallel-size 2 --kv-cache-dtype fp8 --max-num-seqs 128"   # T3: c=128
  "8b-awq_tp2_fp8kv_s128|0,1|Qwen/Qwen3-8B-AWQ|--tensor-parallel-size 2 --kv-cache-dtype fp8 --max-num-seqs 128"   # T3: c=128
  # 8B as 2 copies: with torch.compile (Inductor) the GPU0 copy runs out of memory while autotuning a 2,048-token
  # logits buffer (1.16 GiB), so these run with the Inductor backend off; the _eager 1-GPU and TP=2 runs measure what
  # that costs, so H1 compares like with like.
  "8b-awq_dp2_eager|0,1|Qwen/Qwen3-8B-AWQ|--data-parallel-size 2 --compilation-config {\"backend\":\"eager\"}"
  "8b-awq_1gpu_eager|1|Qwen/Qwen3-8B-AWQ|--compilation-config {\"backend\":\"eager\"}"
  "8b-awq_tp2_eager|0,1|Qwen/Qwen3-8B-AWQ|--tensor-parallel-size 2 --compilation-config {\"backend\":\"eager\"}"
)
# vLLM server and the bench client run on CPUSET (taskset) so other CPU work on the PC cannot skew the numbers
CPUSET="${CPUSET:-0-19}"

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
  taskset -c "$CPUSET" "$VENV/bin/vllm" bench serve --backend vllm --model "$MODEL" --host $HOST --port $PORT --dataset-name random \
    --random-range-ratio 0 --ignore-eos --temperature 0 "$@" > "$log" 2>&1 || return 1
  bench_ok "$log"
}
SMOKE_C=64   # per config: seqs_of; a KV size is accepted only if the server also survives a burst at that concurrency
try_start() {  # try_start <label> <kv args>: start the server; 0 = up and survived the smoke test (left running)
  local A="$1" KVA="$2" i up=""
  LOG="$OUT/$NAME.try-$A.log"
  CUDA_VISIBLE_DEVICES=$GPUS setsid taskset -c "$CPUSET" env $(kv_env "$EXTRA") "$VENV/bin/vllm" serve "$MODEL" $COMMON $SEQ_ARG $KVA $EXTRA > "$LOG" 2>&1 &
  SPID=$!
  local tries=180; [ -n "$(kv_env "$EXTRA")" ] && tries=480   # first FP8-KV start compiles FlashInfer kernels
  for i in $(seq 1 $tries); do
    curl -sf "http://$HOST:$PORT/health" >/dev/null && { up=1; break; }
    kill -0 "$SPID" 2>/dev/null || break; sleep 5
  done
  if [ -z "$up" ]; then
    echo "  did not start ($A${KVA:+, $KVA}): $(grep -m1 -E 'No available memory|out of memory|Error' "$LOG" | sed 's/^.*\] //' | cut -c1-140)"
    stop_server "$SPID" || exit 1; return 1
  fi
  if bench "$OUT/$NAME.try-$A.smoke.log" --random-input-len $IN --random-output-len 16 --num-prompts $((2 * SMOKE_C)) \
       --max-concurrency $SMOKE_C && curl -sf "http://$HOST:$PORT/health" >/dev/null; then
    echo "  ok ($A${KVA:+, $KVA}): $(grep -h 'GPU KV cache size' "$LOG" | sed 's/^.*\] //' | head -1)"; return 0
  fi
  echo "  started but failed the c=$SMOKE_C smoke test ($A${KVA:+, $KVA}): $(grep -m1 -E 'out of memory|Error' "$LOG" | sed 's/^.*\] //' | cut -c1-140)"
  stop_server "$SPID" || exit 1; return 1
}

{ echo "run_id $RUN_ID"; echo "cpuset $CPUSET"; echo "idle_mib gpu0 ${IDLE[0]} gpu1 ${IDLE[1]}"; cat results/versions.txt 2>/dev/null; echo "common $COMMON (+ --max-num-seqs 64 unless the config sets it)"; echo "in $IN out $OUTLEN seed $SEED"; } > "$OUT/run.txt"
for c in "${CONFIGS[@]}"; do
  IFS='|' read -r NAME GPUS MODEL EXTRA <<<"$c"
  [ -n "${ONLY:-}" ] && [[ " $ONLY " != *" $NAME "* ]] && continue
  echo "=== $NAME (GPUs $GPUS)"
  SMOKE_C=$(seqs_of "$EXTRA"); SEQ_ARG=$(seq_arg "$EXTRA")
  "$VENV/bin/hf" download "$MODEL" >/dev/null 2>&1 || "$VENV/bin/huggingface-cli" download "$MODEL" >/dev/null   # outside the health timeout
  wait_idle || { echo "  GPUs busy before start, stopping"; exit 1; }
  # KV rule (same for every config): the largest KV size that both starts and survives a c=SMOKE_C burst.
  # 1) vLLM's automatic sizing, used to read its "--kv-cache-memory=X ... to fully utilize" suggestion;
  # 2) explicit sizes stepping down from X (KV_STEPS), first one that passes wins;
  # 3) if no explicit size larger than the automatic one passes, fall back to the automatic size.
  STARTED=""; KVARG=""
  try_start auto ""; AUTO_OK=$?
  SUGGEST=$(kv_suggest "$LOG"); AUTO_KV=$(kv_auto "$LOG")   # TP/PP ranks report different sizes: take the smallest
  [ $AUTO_OK = 0 ] && stop_server "$SPID"
  if [ -n "$SUGGEST" ]; then
    for A in $KV_STEPS; do
      KVB=$(python3 -c "print(int($SUGGEST*$A))")
      [ $AUTO_OK = 0 ] && [ "$KVB" -le "$AUTO_KV" ] && break      # not better than automatic sizing
      # an explicit KV size brings back vLLM's start-up check "free memory >= utilization x total" (default 0.92),
      # which GPU0 fails because it also drives the display; 0.85 only relaxes that check, KV size is set explicitly
      try_start "$A" "--kv-cache-memory-bytes $KVB --gpu-memory-utilization 0.85" && { STARTED="$A"; KVARG="--kv-cache-memory-bytes $KVB --gpu-memory-utilization 0.85"; break; }
    done
  fi
  if [ -z "$STARTED" ] && [ $AUTO_OK = 0 ]; then try_start auto2 "" && STARTED=auto2; fi
  if [ -z "$STARTED" ]; then echo "  $NAME: does not fit (no KV size started and survived load), see $OUT/$NAME.try-*.log"; continue; fi
  cp "$OUT/$NAME.try-$STARTED.log" "$OUT/$NAME.log"
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
