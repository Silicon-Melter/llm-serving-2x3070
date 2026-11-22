#!/usr/bin/env bash
# Two envs because vLLM and SGLang pin different torch versions. Envs, caches and builds live on a big disk
# ($BULK, default /media/earth/SSD2/llm-serving) so the root SSD is not filled. Versions -> results/versions.txt.
set -euo pipefail
BULK="${BULK:-/media/earth/SSD2/llm-serving}"
export PIP_CACHE_DIR="$BULK/pip-cache" UV_CACHE_DIR="$BULK/pip-cache/uv" HF_HUB_CACHE="$BULK/hf"
VLLM_ENV="$BULK/envs/serve-vllm"; SGL_ENV="$BULK/envs/serve-sglang"
mkdir -p results "$BULK"/{envs,build,pip-cache}
eval "$(conda shell.bash hook)"

mkenv() {  # mkenv <prefix> <pip args...>
  local prefix="$1"; shift
  [ -x "$prefix/bin/python" ] || conda create -y -q -p "$prefix" python=3.12
  "$prefix/bin/python" -m pip install -q -U pip uv
  "$prefix/bin/python" -m uv pip install --python "$prefix/bin/python" "$@"
}
mkenv "$VLLM_ENV" vllm "lm-eval[api]" pandas matplotlib
mkenv "$SGL_ENV" "sglang[all]"

{ echo "date $(date -Is)"; nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | sed 's/^/driver /'
  "$VLLM_ENV/bin/python" -c "import vllm,torch;print('vllm',vllm.__version__,'torch',torch.__version__,'cuda',torch.version.cuda,'nccl',torch.cuda.nccl.version())"
  "$SGL_ENV/bin/python" -c "import sglang,torch;print('sglang',sglang.__version__,'torch',torch.__version__,'cuda',torch.version.cuda)"
} | tee results/versions.txt

# nccl-tests against the NCCL that vLLM's torch actually loads (there is no system NCCL on ad-pc)
NCCL_HOME="$("$VLLM_ENV/bin/python" -c "import nvidia.nccl,os;print(os.path.dirname(nvidia.nccl.__file__) if nvidia.nccl.__file__ else list(nvidia.nccl.__path__)[0])")"
[ -e "$NCCL_HOME/lib/libnccl.so" ] || ln -sf "$(ls "$NCCL_HOME"/lib/libnccl.so.* | head -1)" "$NCCL_HOME/lib/libnccl.so"
[ -d "$BULK/build/nccl-tests" ] || git clone -q https://github.com/NVIDIA/nccl-tests.git "$BULK/build/nccl-tests"
# The system CUDA 12.3 nvcc rejects gcc 13, and torch here ships CUDA 13.0 + NCCL 2.30: build with a matching nvcc
CUDA_BUILD="$BULK/envs/cuda-build"
[ -x "$CUDA_BUILD/bin/nvcc" ] || conda create -y -q -p "$CUDA_BUILD" -c nvidia -c conda-forge cuda-nvcc=13.0 cuda-cudart-dev=13.0
make -C "$BULK/build/nccl-tests" -j16 CUDA_HOME="$CUDA_BUILD" CUDA_LIB="$CUDA_BUILD/targets/x86_64-linux/lib" \
  CUDA_INC="$CUDA_BUILD/targets/x86_64-linux/include" NCCL_HOME="$NCCL_HOME" NVCC_GENCODE="-gencode=arch=compute_86,code=sm_86" >/dev/null
echo "nccl-tests built against $NCCL_HOME" | tee -a results/versions.txt
