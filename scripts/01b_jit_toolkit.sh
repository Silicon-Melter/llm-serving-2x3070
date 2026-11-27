#!/usr/bin/env bash
# CUDA toolkit for FlashInfer's JIT, which vLLM uses for FP8 KV on sm_86 (kv_env in 03_sweep_vllm.sh / 06_quality.sh).
# Why not the venv's own nvidia/cu13: its nvcc wheel is 13.5 while its cudart headers are 13.0 (torch cu130), and
# CCCL stops every build with "CUDA compiler and CUDA toolkit headers are incompatible". /usr/local/cuda is 12.3.
# So: a 13.0-only toolkit from the same NVIDIA wheels, matched to torch's cudart 13.0.96, in $BULK/cuda-jit-13.0.
set -euo pipefail
BULK="${BULK:-/media/earth/SSD2/llm-serving}"
JIT="$BULK/cuda-jit-13.0"; HOME_CU="$JIT/nvidia/cu13"
[ "${1:-}" = "--home" ] && { echo "$HOME_CU"; exit 0; }   # the CUDA_HOME kv_env must point at (tests/test_scripts.py)
PY="$BULK/envs/serve-vllm/bin/python"; export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$BULK/pip-cache}"

# One pip call: pip --target will not merge a second wheel into an existing nvidia/ folder.
if [ ! -x "$HOME_CU/nvvm/bin/cicc" ] || [ ! -e "$HOME_CU/include/curand.h" ]; then
  rm -rf "$JIT"
  "$PY" -m pip install -q --no-deps --target "$JIT" \
    nvidia-cuda-nvcc==13.0.88 nvidia-cuda-crt==13.0.88 nvidia-nvvm==13.0.88 \
    nvidia-cuda-runtime==13.0.96 nvidia-cuda-cccl==13.0.85 nvidia-curand==10.4.0.35   # curand.h: sampling kernels
fi
# FlashInfer links with -L$CUDA_HOME/lib64 -L$CUDA_HOME/lib64/stubs -lcudart -lcuda; the wheels have lib/ only.
mkdir -p "$HOME_CU/lib64/stubs"
ln -sfn ../lib/libcudart.so.13 "$HOME_CU/lib64/libcudart.so"
ln -sfn "$(ldconfig -p | awk '$1 == "libcuda.so" {print $NF; exit}')" "$HOME_CU/lib64/stubs/libcuda.so"

"$HOME_CU/bin/nvcc" --version | grep -q "release 13.0," || { echo "nvcc in $HOME_CU is not 13.0" >&2; exit 1; }
grep -q "define CUDART_VERSION  13000" "$HOME_CU/include/cuda_runtime_api.h" || { echo "cudart headers not 13.0" >&2; exit 1; }
echo "flashinfer jit toolkit $HOME_CU (nvcc 13.0.88, cudart 13.0.96)"
