#!/usr/bin/env bash
# Two envs because vLLM and SGLang pin different torch versions. Record the versions you get in results/versions.txt.
set -euo pipefail
mkdir -p results
conda create -y -n serve-vllm python=3.12 && conda run -n serve-vllm pip install -U vllm "lm-eval[api]" pandas matplotlib
conda create -y -n serve-sglang python=3.12 && conda run -n serve-sglang pip install -U "sglang[all]"
{ conda run -n serve-vllm python -c "import vllm,torch;print('vllm',vllm.__version__,'torch',torch.__version__)";
  conda run -n serve-sglang python -c "import sglang,torch;print('sglang',sglang.__version__,'torch',torch.__version__)"; } | tee results/versions.txt
# nccl-tests for the raw interconnect numbers
[ -d nccl-tests ] || git clone https://github.com/NVIDIA/nccl-tests.git
make -C nccl-tests -j CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
