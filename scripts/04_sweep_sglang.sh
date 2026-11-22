#!/usr/bin/env bash
# Same sweep for SGLang (TP and DP; SGLang has no PP for these models in most versions). Check flags with --help.
set -uo pipefail
PORT=30000; OUT=results/sglang; mkdir -p "$OUT" logs
CONC="1 4 16 64"; IN=512; OUTLEN=256; N=200
CONFIGS=(
  "8b-awq_1gpu|CUDA_VISIBLE_DEVICES=0|Qwen/Qwen3-8B-AWQ|"
  "8b-awq_tp2|CUDA_VISIBLE_DEVICES=0,1|Qwen/Qwen3-8B-AWQ|--tp 2"
  "8b-awq_dp2|CUDA_VISIBLE_DEVICES=0,1|Qwen/Qwen3-8B-AWQ|--dp 2"
  "14b-awq_tp2|CUDA_VISIBLE_DEVICES=0,1|Qwen/Qwen3-14B-AWQ|--tp 2"
)
for c in "${CONFIGS[@]}"; do
  IFS='|' read -r NAME ENVV MODEL EXTRA <<<"$c"
  echo "=== $NAME"
  env $ENVV conda run --no-capture-output -n serve-sglang python -m sglang.launch_server --model-path "$MODEL" \
    --port $PORT --context-length 4096 --mem-fraction-static 0.85 $EXTRA > "logs/sgl_$NAME.log" 2>&1 &
  SPID=$!
  for i in $(seq 1 180); do curl -sf localhost:$PORT/health >/dev/null && break; sleep 5; done
  if ! curl -sf localhost:$PORT/health >/dev/null; then echo "  did not start, see logs/sgl_$NAME.log"; kill $SPID 2>/dev/null; wait $SPID 2>/dev/null; continue; fi
  for C in $CONC; do
    conda run -n serve-sglang python -m sglang.bench_serving --backend sglang --port $PORT \
      --dataset-name random --random-input-len $IN --random-output-len $OUTLEN --random-range-ratio 1 \
      --num-prompts $N --max-concurrency "$C" --output-file "$OUT/${NAME}_c${C}.jsonl" > "logs/sgl_${NAME}_c${C}.bench.log" 2>&1
    echo "  c=$C done"
  done
  kill $SPID; wait $SPID 2>/dev/null; sleep 10
done
