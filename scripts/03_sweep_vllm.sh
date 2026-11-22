#!/usr/bin/env bash
# For each config: start `vllm serve`, wait for /health, sweep concurrency with `vllm bench serve`, stop.
# Check the flag names against `vllm serve --help` for the version you installed.
set -uo pipefail
PORT=8000; OUT=results/vllm; mkdir -p "$OUT" logs
CONC="1 4 16 64"; IN=512; OUTLEN=256; N=200
COMMON="--max-model-len 4096 --gpu-memory-utilization 0.90 --port $PORT"
# name | env | model | extra args
CONFIGS=(
  "8b-awq_1gpu|CUDA_VISIBLE_DEVICES=0|Qwen/Qwen3-8B-AWQ|"
  "8b-awq_tp2|CUDA_VISIBLE_DEVICES=0,1|Qwen/Qwen3-8B-AWQ|--tensor-parallel-size 2"
  "8b-awq_tp2_nocustomar|CUDA_VISIBLE_DEVICES=0,1|Qwen/Qwen3-8B-AWQ|--tensor-parallel-size 2 --disable-custom-all-reduce"
  "8b-awq_pp2|CUDA_VISIBLE_DEVICES=0,1|Qwen/Qwen3-8B-AWQ|--pipeline-parallel-size 2"
  "8b-awq_dp2|CUDA_VISIBLE_DEVICES=0,1|Qwen/Qwen3-8B-AWQ|--data-parallel-size 2"
  "8b-fp8_tp2|CUDA_VISIBLE_DEVICES=0,1|Qwen/Qwen3-8B-FP8|--tensor-parallel-size 2"
  "8b-fp8_pp2|CUDA_VISIBLE_DEVICES=0,1|Qwen/Qwen3-8B-FP8|--pipeline-parallel-size 2"
  "14b-awq_tp2|CUDA_VISIBLE_DEVICES=0,1|Qwen/Qwen3-14B-AWQ|--tensor-parallel-size 2"
  "14b-awq_pp2|CUDA_VISIBLE_DEVICES=0,1|Qwen/Qwen3-14B-AWQ|--pipeline-parallel-size 2"
  "4b-bf16_tp2|CUDA_VISIBLE_DEVICES=0,1|Qwen/Qwen3-4B|--tensor-parallel-size 2"
)
for c in "${CONFIGS[@]}"; do
  IFS='|' read -r NAME ENVV MODEL EXTRA <<<"$c"
  echo "=== $NAME"
  env $ENVV conda run --no-capture-output -n serve-vllm vllm serve "$MODEL" $COMMON $EXTRA > "logs/$NAME.log" 2>&1 &
  SPID=$!
  for i in $(seq 1 180); do curl -sf localhost:$PORT/health >/dev/null && break; sleep 5; done
  if ! curl -sf localhost:$PORT/health >/dev/null; then echo "  did not start (OOM?), see logs/$NAME.log"; kill $SPID 2>/dev/null; wait $SPID 2>/dev/null; continue; fi
  for C in $CONC; do
    conda run -n serve-vllm vllm bench serve --backend vllm --model "$MODEL" --port $PORT \
      --dataset-name random --random-input-len $IN --random-output-len $OUTLEN \
      --num-prompts $N --max-concurrency "$C" --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,99 \
      --save-result --result-dir "$OUT" --result-filename "${NAME}_c${C}.json" > "logs/${NAME}_c${C}.bench.log" 2>&1
    echo "  c=$C done"
  done
  kill $SPID; wait $SPID 2>/dev/null; sleep 10
done
