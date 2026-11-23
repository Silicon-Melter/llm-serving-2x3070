#!/usr/bin/env bash
# M4 quality: for each config, start `vllm serve`, then
#   Stage A  07_kld.py collect   -> results/quality/<run>/<name>/logprobs.jsonl.gz   (top-20 logprobs on a fixed corpus)
#   Stage B  lm-eval gsm8k (all 1,319, 5-shot, greedy) + mmlu (20 per subject, 5-shot) -> .../<name>/lm_eval/
# and stop it. Comparisons against a reference are done afterwards with 07_kld.py compare / 08_quality_compare.py.
# Completions endpoint, no chat template: Qwen3's thinking mode is never switched on, identically for every model.
#   ONLY="ref4b-bf16_tp2 4b-awq_1gpu" bash scripts/06_quality.sh
#   STAGES="kld" bash scripts/06_quality.sh        # quick screen only (Stage A)
# Resumable: a stage whose output already exists (and is not an empty or cut-off file) is skipped, so re-running
# with the same RUN_ID after a crash only does what is missing. FORCE=1 redoes everything.
set -uo pipefail
# stage_done <config dir> <stage>: 0 if that stage's output is complete. A reset can leave 0-byte or cut-off files.
stage_done() { local d="$1"
  case "$2" in
    kld)   [ -s "$d/logprobs.jsonl.gz" ] && gzip -t "$d/logprobs.jsonl.gz" 2>/dev/null ;;
    gsm8k) [ -n "$(find "$d/lm_eval" -name 'samples_gsm8k_*.jsonl' -size +0 2>/dev/null | head -1)" ] ;;
    mmlu)  [ "$(find "$d/lm_eval" -name 'samples_mmlu_*.jsonl' -size +0 2>/dev/null | wc -l)" -ge 57 ] ;;
    *) return 1 ;;
  esac; }
[ "${1:-}" = "--stage-done" ] && { stage_done "$2" "$3"; exit $?; }   # used by tests/test_scripts.py
BULK="${BULK:-/media/earth/SSD2/llm-serving}"; VENV="$BULK/envs/serve-vllm"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$BULK/hf}"
PORT=8000; HOST=127.0.0.1; CPUSET="${CPUSET:-0-19}"; SEED=1234
STAGES="${STAGES:-kld gsm8k mmlu}"; MMLU_LIMIT="${MMLU_LIMIT:-20}"; GSM8K_LIMIT="${GSM8K_LIMIT:-}"
CONCURRENCY="${CONCURRENCY:-32}"   # requests in flight from the evaluator; a second run with another value = noise floor
RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"; OUT="results/quality/$RUN_ID"; mkdir -p "$OUT"
CORPUS="results/quality/corpus.json"
COMMON="--max-model-len 4096 --max-num-seqs 32 --port $PORT --host $HOST --seed $SEED"
CONFIGS=(   # name|GPUs|model|extra   (quality does not depend on the GPU layout, so one layout per checkpoint)
# ref8b: BF16 8B does not fit in 2 x 8 GiB, so 24 of 36 layers per GPU stay in host RAM and are copied in each step
# (prefetch offload, ~0.39 s per step whatever the batch): 83 tok/s with 32 requests at once (probe 2026-11-23).
# UVA offload (--cpu-offload-gb) re-reads host memory per batch tile: 13-15 tok/s at any batch, GSM8K timed out.
# Automatic KV sizing fails with prefetch ("0 blocks fit"), so the KV size is given: 2.6 GiB = ~38k tokens.
  "ref8b-bf16_tp2|0,1|Qwen/Qwen3-8B|--tensor-parallel-size 2 --offload-backend prefetch --offload-group-size 3 --offload-num-in-group 2 --kv-cache-memory-bytes 2791728742 --gpu-memory-utilization 0.85"
  "8b-awq_tp2|0,1|Qwen/Qwen3-8B-AWQ|--tensor-parallel-size 2"
  "8b-awq_tp2_fp8kv|0,1|Qwen/Qwen3-8B-AWQ|--tensor-parallel-size 2 --kv-cache-dtype fp8"
  "8b-fp8_tp2|0,1|Qwen/Qwen3-8B-FP8|--tensor-parallel-size 2"
  "8b-fp8_tp2_fp8kv|0,1|Qwen/Qwen3-8B-FP8|--tensor-parallel-size 2 --kv-cache-dtype fp8"
  "8b-w8a8_tp2|0,1|nm-testing/Qwen3-8B-W8A8-INT8|--tensor-parallel-size 2"
  "ref4b-bf16_tp2|0,1|Qwen/Qwen3-4B|--tensor-parallel-size 2"
  "4b-awq_1gpu|1|Qwen/Qwen3-4B-AWQ|"
  "4b-awq_1gpu_fp8kv|1|Qwen/Qwen3-4B-AWQ|--kv-cache-dtype fp8"
  "4b-fp8_1gpu|1|Qwen/Qwen3-4B-FP8|"
)

gpu_used() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$1" | tr -d ' '; }
declare -A IDLE; for g in 0 1; do IDLE[$g]=$(gpu_used $g); done
wait_idle() { local i g busy; for i in $(seq 1 60); do busy=""
    for g in 0 1; do [ $(( $(gpu_used $g) - ${IDLE[$g]} )) -gt 300 ] && busy=1; done
    [ -z "$busy" ] && return 0; sleep 5; done; return 1; }
stop_server() { kill -INT -- -"$1" 2>/dev/null; for _ in $(seq 1 30); do kill -0 "$1" 2>/dev/null || break; sleep 2; done
  kill -KILL -- -"$1" 2>/dev/null; wait "$1" 2>/dev/null; wait_idle; }

[ -f "$CORPUS" ] || "$VENV/bin/python" scripts/07_kld.py corpus --model Qwen/Qwen3-8B-AWQ --out "$CORPUS" || exit 1
{ echo "run_id $RUN_ID"; echo "stages $STAGES"; echo "mmlu_limit $MMLU_LIMIT gsm8k_limit ${GSM8K_LIMIT:-all}"
  echo "concurrency $CONCURRENCY"; echo "common $COMMON"; echo "cpuset $CPUSET"
  echo "started $(date -Is) commit $(git rev-parse --short HEAD 2>/dev/null)"; cat results/versions.txt 2>/dev/null; } >> "$OUT/run.txt"
for c in "${CONFIGS[@]}"; do
  IFS='|' read -r NAME GPUS MODEL EXTRA <<<"$c"
  [ -n "${ONLY:-}" ] && [[ " $ONLY " != *" $NAME "* ]] && continue
  D="$OUT/$NAME"; mkdir -p "$D"; TODO=""
  for s in $STAGES; do { [ -z "${FORCE:-}" ] && stage_done "$D" "$s"; } || TODO="$TODO $s"; done
  [ -z "$TODO" ] && { echo "=== $NAME: all stages already done, skipped"; continue; }
  echo "=== $NAME ($MODEL, GPUs $GPUS), stages:$TODO"
  wait_idle || { echo "  GPUs busy before start, stopping"; exit 1; }
  CUDA_VISIBLE_DEVICES=$GPUS setsid taskset -c "$CPUSET" "$VENV/bin/vllm" serve "$MODEL" $COMMON $EXTRA > "$D/server.log" 2>&1 &
  SPID=$!; up=""
  for _ in $(seq 1 240); do curl -sf "http://$HOST:$PORT/health" >/dev/null && { up=1; break; }
    kill -0 "$SPID" 2>/dev/null || break; sleep 5; done
  if [ -z "$up" ]; then echo "  did not start: $(grep -m1 -E 'out of memory|Error' "$D/server.log" | sed 's/^.*\] //' | cut -c1-160)"
    stop_server "$SPID"; continue; fi
  grep -hE "GPU KV cache size|Model loading took|kv_cache_dtype|MarlinLinearKernel|quantization" "$D/server.log" | sed 's/^.*\] //' | sort -u > "$D/server.txt"
  T0=$(date +%s)
  if [[ " $TODO " == *" kld "* ]]; then
    taskset -c "$CPUSET" "$VENV/bin/python" scripts/07_kld.py collect --corpus "$CORPUS" --model "$MODEL" \
      --out "$D/logprobs.jsonl.gz" --concurrency "$CONCURRENCY" && echo "  kld: collected ($(( $(date +%s) - T0 )) s)" || echo "  kld: FAILED"
  fi
  LM="$VENV/bin/lm_eval --model local-completions --model_args model=$MODEL,base_url=http://$HOST:$PORT/v1/completions,num_concurrent=$CONCURRENCY,max_retries=3,timeout=3600,tokenized_requests=False,tokenizer=$MODEL,max_length=4096 --batch_size 1 --log_samples --seed $SEED --output_path $D/lm_eval"
  if [[ " $TODO " == *" gsm8k "* ]]; then T0=$(date +%s)
    taskset -c "$CPUSET" $LM --tasks gsm8k --num_fewshot 5 --gen_kwargs temperature=0 ${GSM8K_LIMIT:+--limit $GSM8K_LIMIT} > "$D/gsm8k.log" 2>&1 \
      && echo "  gsm8k: $(grep -E 'strict-match' "$D/gsm8k.log" | head -1 | tr -s ' ') ($(( $(date +%s) - T0 )) s)" || echo "  gsm8k: FAILED (see $D/gsm8k.log)"
  fi
  if [[ " $TODO " == *" mmlu "* ]]; then T0=$(date +%s)
    taskset -c "$CPUSET" $LM --tasks mmlu --num_fewshot 5 --limit "$MMLU_LIMIT" > "$D/mmlu.log" 2>&1 \
      && echo "  mmlu: $(grep -E '^\|mmlu ' "$D/mmlu.log" | head -1 | tr -s ' ') ($(( $(date +%s) - T0 )) s)" || echo "  mmlu: FAILED (see $D/mmlu.log)"
  fi
  sync   # results on disk before the next config (a reset lost unflushed sample files once)
  stop_server "$SPID" || { echo "  GPUs did not return to idle, stopping"; exit 1; }
done
echo "done: $OUT"
