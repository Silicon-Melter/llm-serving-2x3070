#!/usr/bin/env bash
# Records the facts the write-up needs: GPUs, driver, PCIe links, topology, P2P support.
set -u
nvidia-smi --query-gpu=index,name,memory.total,driver_version,pcie.link.gen.max,pcie.link.width.max,pcie.link.gen.current,pcie.link.width.current --format=csv
echo; nvidia-smi topo -m
echo; nvidia-smi topo -p2p r 2>/dev/null || echo "p2p query not supported by this nvidia-smi"
echo; python - <<'PY' 2>/dev/null || echo "torch not installed yet"
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("peer access 0->1:", torch.cuda.can_device_access_peer(0, 1))
PY
