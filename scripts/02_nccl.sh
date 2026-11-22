#!/usr/bin/env bash
# All-reduce across both GPUs: time (latency) and bus bandwidth from 8 B to 256 MB, P2P allowed (default) and
# forced off. Decode-step all-reduces are batch x hidden x 2 B = 8 KB (c=1) to 512 KB (c=64), so the small-size
# latency is what decides TP; the 256 MB number is only the link's bandwidth ceiling.
# Usage: bash scripts/02_nccl.sh [--print-env]   (OUT=dir to choose the output folder)
# Also logs which transport NCCL picked and the PCIe gen/width while traffic is flowing.
set -euo pipefail
BULK="${BULK:-/media/earth/SSD2/llm-serving}"
VLLM_ENV="$BULK/envs/serve-vllm"; BIN="$BULK/build/nccl-tests/build/all_reduce_perf"
NCCL_LIB="$("$VLLM_ENV/bin/python" -c "import nvidia.nccl,os;print(list(nvidia.nccl.__path__)[0])")/lib"
CUDART_LIB="$VLLM_ENV/lib/python3.12/site-packages/nvidia/cu13/lib"   # same CUDA 13 runtime torch loads
export LD_LIBRARY_PATH="$NCCL_LIB:$CUDART_LIB:${LD_LIBRARY_PATH:-}"
OUT="${OUT:-results/nccl/$(date +%Y%m%d-%H%M%S)}"
[ "${1:-}" = "--print-env" ] && { echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"; echo "BIN=$BIN"; exit 0; }
mkdir -p "$OUT"
run() {  # run <tag> [env...]
  local tag="$1"; shift
  ( while sleep 1; do nvidia-smi --query-gpu=index,pcie.link.gen.current,pcie.link.width.current --format=csv,noheader; done ) \
    > "$OUT/pcie_${tag}.txt" & local mon=$!
  env "$@" NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,P2P,SHM NCCL_DEBUG_FILE="$OUT/${tag}.debug.txt" "$BIN" -b 8 -e 256M -f 2 -g 2 -w 20 -n 100 \
    | tee "$OUT/allreduce_${tag}.txt"
  kill $mon 2>/dev/null || true
  grep -hoE "via (P2P|SHM|NET)[^ ]*( [^ ]*)?" "$OUT/${tag}.debug.txt" | sort | uniq -c | sed "s/^/[$tag transport] /" || true
  sort "$OUT/pcie_${tag}.txt" | uniq -c | sort -rn | head -3 | sed "s/^/[$tag pcie] /"
}
run p2p_default
run p2p_off NCCL_P2P_DISABLE=1
