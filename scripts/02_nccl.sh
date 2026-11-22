#!/usr/bin/env bash
# All-reduce bandwidth across both GPUs, with P2P allowed (default) and forced off, so the write-up can show what PCIe gives.
set -euo pipefail
mkdir -p results
./nccl-tests/build/all_reduce_perf -b 8K -e 256M -f 2 -g 2 | tee results/nccl_allreduce.txt
NCCL_P2P_DISABLE=1 ./nccl-tests/build/all_reduce_perf -b 8K -e 256M -f 2 -g 2 | tee results/nccl_allreduce_p2p_off.txt
