#!/usr/bin/env bash
# T1 / T2: single-stream decode with llama.cpp, the community's method, on our box, with their GGUF files.
#   T2: Llama-3-8B Q4_K_M on one 3070 (XiongjieDai: 72.79 tok/s, llama-bench tg512 on a 3070).
#   T1: Qwen3-14B Q4_K_M on both cards, layer / row / tensor split (XDA, 2 x 5060 Ti: 46.37 / - / 79.64 tok/s).
#   Also Qwen3-8B Q4_K_M split three ways, next to vLLM's 8B-AWQ TP=2 (110 tok/s at c=1).
# Steps (each skipped if already done): conda CUDA 13.0 toolkit with cuBLAS ($BULK/envs/cuda-llama), llama.cpp
# checkout + CUDA build for sm_86 ($BULK/build/llama.cpp), GGUF downloads ($BULK/gguf), then llama-bench, 5 repeats,
# JSON per run -> results/llamacpp/<run>/.
#   RUN_ID=t1-xxx bash scripts/13_llamacpp.sh            STEPS="build" bash scripts/13_llamacpp.sh
set -euo pipefail
BULK="${BULK:-/media/earth/SSD2/llm-serving}"; CT="$BULK/envs/cuda-llama"; SRC="$BULK/build/llama.cpp"
BIN="$SRC/build/bin"; HF="$BULK/envs/serve-vllm/bin/hf"; G="$BULK/gguf"
export HF_HUB_CACHE="$BULK/hf"
STEPS="${STEPS:-build download bench}"; REPS="${REPS:-5}"; CPUSET="${CPUSET:-0-19}"
LLAMA_REF="${LLAMA_REF:-}"   # empty = keep the checked-out commit (recorded in run.txt)
declare -A GGUF=(   # name -> repo|file
  [llama3-8b]="JaaackXD/Llama-3-8B-GGUF|ggml-model-Q4_K_M.gguf"
  [qwen3-8b]="Qwen/Qwen3-8B-GGUF|Qwen3-8B-Q4_K_M.gguf"
  [qwen3-14b]="Qwen/Qwen3-14B-GGUF|Qwen3-14B-Q4_K_M.gguf"
)
BENCHES=(   # name | GPUs | gguf | llama-bench args (besides -m, -r, -o json)
  "t2-llama3-8b_1gpu_fa0|1|llama3-8b|-ngl 99 -p 512 -n 512 -fa 0"
  "t2-llama3-8b_1gpu_fa1|1|llama3-8b|-ngl 99 -p 512 -n 512 -fa 1"
  "qwen3-8b_1gpu|1|qwen3-8b|-ngl 99 -p 512 -n 128 -fa 1"
  "qwen3-8b_layer|0,1|qwen3-8b|-ngl 99 -p 512 -n 128 -fa 1 -sm layer"
  "qwen3-8b_row|0,1|qwen3-8b|-ngl 99 -p 512 -n 128 -fa 1 -sm row"
  "qwen3-8b_tensor|0,1|qwen3-8b|-ngl 99 -p 512 -n 128 -fa 1 -sm tensor"
  "t1-qwen3-14b_layer|0,1|qwen3-14b|-ngl 99 -p 512 -n 128 -fa 1 -sm layer"
  "t1-qwen3-14b_row|0,1|qwen3-14b|-ngl 99 -p 512 -n 128 -fa 1 -sm row"
  "t1-qwen3-14b_tensor|0,1|qwen3-14b|-ngl 99 -p 512 -n 128 -fa 1 -sm tensor"
)
gguf_path() { local f="${GGUF[$1]#*|}"; echo "$G/$1/$f"; }
[ "${1:-}" = "--benches" ] && { printf '%s\n' "${BENCHES[@]}"; exit 0; }   # used by tests/test_scripts.py
[ "${1:-}" = "--gguf-path" ] && { gguf_path "$2"; exit 0; }

if [[ " $STEPS " == *" build "* ]]; then
  # nvml.h: ggml-cuda includes it (cuda-nvml-dev); a missing package is installed into an existing env
  PKGS="cuda-version=13.0 cuda-nvcc=13.0 cuda-cudart-dev=13.0 libcublas-dev cuda-nvml-dev cmake ninja"
  if [ ! -x "$CT/bin/nvcc" ] || [ ! -e "$CT/targets/x86_64-linux/include/nvml.h" ]; then
    set +u; eval "$(conda shell.bash hook)"   # conda's scripts read unset variables
    if [ -d "$CT/conda-meta" ]; then conda install -y -q -p "$CT" -c nvidia -c conda-forge $PKGS
    else conda create -y -q -p "$CT" -c nvidia -c conda-forge $PKGS; fi
    set -u
  fi
  [ -d "$SRC/.git" ] || git clone -q https://github.com/ggml-org/llama.cpp.git "$SRC"
  [ -n "$LLAMA_REF" ] && git -C "$SRC" checkout -q "$LLAMA_REF"
  if [ ! -x "$BIN/llama-bench" ] || [ "$(cat "$SRC/build/.built-commit" 2>/dev/null)" != "$(git -C "$SRC" rev-parse HEAD)" ]; then
    PATH="$CT/bin:$PATH" CUDACXX="$CT/bin/nvcc" "$CT/bin/cmake" -S "$SRC" -B "$SRC/build" -G Ninja \
      -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 -DLLAMA_CURL=OFF \
      -DCUDAToolkit_ROOT="$CT" > "$BULK/logs/llamacpp-cmake.log" 2>&1
    PATH="$CT/bin:$PATH" nice "$CT/bin/cmake" --build "$SRC/build" -j "${JOBS:-12}" --target llama-bench llama-cli \
      > "$BULK/logs/llamacpp-build.log" 2>&1
    git -C "$SRC" rev-parse HEAD > "$SRC/build/.built-commit"
  fi
  { "$BIN/llama-bench" --help 2>&1 || true; } | grep -E -- "split-mode" | head -2 || true
fi

if [[ " $STEPS " == *" download "* ]]; then
  for n in "${!GGUF[@]}"; do
    [ -s "$(gguf_path "$n")" ] || nice "$HF" download "${GGUF[$n]%%|*}" "${GGUF[$n]#*|}" --local-dir "$G/$n" >/dev/null
  done
fi

if [[ " $STEPS " == *" bench "* ]]; then
  RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"; OUT="results/llamacpp/$RUN_ID"; mkdir -p "$OUT"
  { echo "run_id $RUN_ID"; echo "llama.cpp $(git -C "$SRC" log -1 --format='%h %cd' --date=iso)"; echo "reps $REPS cpuset $CPUSET"
    echo "started $(date -Is) commit $(git rev-parse --short HEAD 2>/dev/null)"
    for n in "${!GGUF[@]}"; do echo "gguf $n ${GGUF[$n]} $(stat -c %s "$(gguf_path "$n")" 2>/dev/null) bytes"; done
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | sed 's/^/idle /'; } > "$OUT/run.txt"
  for b in "${BENCHES[@]}"; do
    IFS='|' read -r NAME GPUS MODEL ARGS <<<"$b"
    [ -n "${ONLY:-}" ] && [[ " $ONLY " != *" $NAME "* ]] && continue
    CUDA_VISIBLE_DEVICES=$GPUS taskset -c "$CPUSET" "$BIN/llama-bench" -m "$(gguf_path "$MODEL")" $ARGS -r "$REPS" \
      -o json > "$OUT/$NAME.json" 2> "$OUT/$NAME.log" \
      && echo "  $NAME: $(python3 -c "import json,sys; print(', '.join(f\"{'pp' if r['n_prompt'] else 'tg'}{r['n_prompt'] or r['n_gen']} {r['avg_ts']:.2f}±{r['stddev_ts']:.2f}\" for r in json.load(open(sys.argv[1]))))" "$OUT/$NAME.json")" \
      || echo "  $NAME: FAILED ($(grep -m1 -iE 'error|out of memory|failed' "$OUT/$NAME.log" | cut -c1-140)), see $OUT/$NAME.log"
  done
  echo "done: $OUT"
fi
