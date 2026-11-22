#!/usr/bin/env bash
# H3: profile 8B-AWQ TP=2 decode with vLLM's torch profiler (version-matched; the installed Nsight Systems 2023.3 predates
# CUDA 13), at 1 and 64 users, then break each rank's GPU time down with 11_trace_breakdown.py.
# Same server settings and KV size as M3 (run m3-581eb68). Short runs: profiling adds overhead, only shares are used.
#   bash scripts/10_profile_tp.sh            # -> results/profile/<run>/c1, c64 (+ breakdown.json)
set -uo pipefail
BULK="${BULK:-/media/earth/SSD2/llm-serving}"; VENV="$BULK/envs/serve-vllm"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$BULK/hf}"
PORT=8000; HOST=127.0.0.1; CPUSET="${CPUSET:-0-19}"; MODEL=Qwen/Qwen3-8B-AWQ
RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"; OUT="results/profile/$RUN_ID"; mkdir -p "$OUT"
TRACE_TMP="$BULK/profile-tmp/$RUN_ID"; mkdir -p "$TRACE_TMP"
KV="${KV:---kv-cache-memory-bytes 3429174476 --gpu-memory-utilization 0.85}"   # M3's accepted TP=2 KV size
# No Python stacks (the default records one per op: the first c=64 attempt grew a worker to 27.6 GB RAM and was
# OOM-killed), and a fixed window of engine steps after a short delay so the trace size does not grow with load.
PROF="{\"profiler\":\"torch\",\"torch_profiler_dir\":\"$TRACE_TMP\",\"torch_profiler_with_stack\":false,\"delay_iterations\":${DELAY:-30},\"max_iterations\":${STEPS:-120}}"
CUDA_VISIBLE_DEVICES=0,1 setsid taskset -c "$CPUSET" "$VENV/bin/vllm" serve $MODEL --tensor-parallel-size 2 \
  --max-model-len 4096 --max-num-seqs 64 --disable-prefix-caching --host $HOST --port $PORT $KV \
  --profiler-config "$PROF" > "$OUT/server.log" 2>&1 &
SPID=$!
for _ in $(seq 1 180); do curl -sf "http://$HOST:$PORT/health" >/dev/null && break; kill -0 $SPID 2>/dev/null || break; sleep 5; done
curl -sf "http://$HOST:$PORT/health" >/dev/null || { echo "server did not start"; tail -5 "$OUT/server.log"; exit 1; }
bench() { taskset -c "$CPUSET" "$VENV/bin/vllm" bench serve --backend vllm --model $MODEL --host $HOST --port $PORT \
  --dataset-name random --random-range-ratio 0 --ignore-eos --temperature 0 --random-input-len 512 "$@"; }
bench --random-output-len 32 --num-prompts 64 --max-concurrency 64 > "$OUT/warmup.log" 2>&1
for C in 1 64; do
  N=$(( C == 1 ? 4 : 128 )); OL=$(( C == 1 ? 64 : 64 ))
  curl -sf -X POST "http://$HOST:$PORT/start_profile" >/dev/null || { echo "start_profile failed"; break; }
  bench --random-output-len $OL --num-prompts $N --max-concurrency $C > "$OUT/c$C.bench.log" 2>&1
  curl -sf -X POST "http://$HOST:$PORT/stop_profile" >/dev/null
  sleep 20   # traces are written asynchronously after stop
  mkdir -p "$OUT/c$C"; find "$TRACE_TMP" -type f -exec mv -t "$OUT/c$C/" {} +   # everything written since the last stop
  echo "c=$C: $(ls "$OUT/c$C" | wc -l) trace files, $(grep -E '^Output token throughput' "$OUT/c$C.bench.log" | tr -s ' ')"
  "$VENV/bin/python" scripts/11_trace_breakdown.py "$OUT/c$C" --json "$OUT/c$C/breakdown.json"
done
kill -INT -- -$SPID 2>/dev/null; sleep 10; kill -KILL -- -$SPID 2>/dev/null
echo "done: $OUT"
