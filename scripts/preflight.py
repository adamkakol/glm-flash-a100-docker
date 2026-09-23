"""Check the container's CUDA visibility and peer access before loading weights."""
import os
import subprocess
import sys
from gpu_topology import require_nvlink_pair


def main():
    import torch
    import torch.distributed as dist
    print(f"PyTorch {torch.__version__}; CUDA {torch.version.cuda}", flush=True)
    expected = os.environ.get("EXPECTED_GPU_UUIDS", "").split(",")
    if len(expected) != 3 or os.environ.get("CUDA_VISIBLE_DEVICES") != ",".join(expected):
        raise RuntimeError("CUDA_VISIBLE_DEVICES must be the three selected GPU UUIDs in order.")
    if torch.cuda.device_count() != 3:
        raise RuntimeError(f"Expected three CUDA devices, found {torch.cuda.device_count()}. Check NVIDIA Container Toolkit.")
    if not dist.is_nccl_available():
        raise RuntimeError("The installed PyTorch does not provide NCCL.")
    for device in range(3):
        props = torch.cuda.get_device_properties(device)
        with torch.cuda.device(device):
            free, total = torch.cuda.mem_get_info()
        print(f"CUDA {device}: {props.name}; free={free / 2**30:.2f} GiB; total={total / 2**30:.2f} GiB; selected UUID={expected[device]}")
        if "A100" not in props.name or total < 70 * 2**30:
            raise RuntimeError("This deployment requires three A100 80GB cards.")
        if free < 70 * 2**30:
            raise RuntimeError(f"GPU {device} has too little free memory. Stop other GPU workloads before loading.")
    for src in range(3):
        print("Peer access", src, [True if src == dst else torch.cuda.can_device_access_peer(src, dst) for dst in range(3)])
    if not (torch.cuda.can_device_access_peer(0, 1) and torch.cuda.can_device_access_peer(1, 0)):
        raise RuntimeError("CUDA peer access is unavailable between the selected NVLink GPUs. Check the host/VM GPU configuration.")
    topology = subprocess.check_output(['nvidia-smi', 'topo', '-m'], text=True, timeout=30)
    inventory = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader'],
                                        text=True, timeout=30)
    pair = require_nvlink_pair(topology, inventory, expected[:2])
    print(topology)
    print(f'Container topology confirms selected NVLink pair: nvidia-smi GPUs {pair}')
    result = subprocess.run(['nvidia-smi', 'nvlink', '--status'], text=True, capture_output=True, timeout=30)
    print(result.stdout or result.stderr)
    if result.returncode:
        print('NVLink status diagnostic unavailable; topology and peer access passed, traffic remains unverified.')
    print("Preflight passed. This verifies topology and peer capability, not the transport used by inference or model performance.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        sys.exit(f"Preflight failed: {exc}")
