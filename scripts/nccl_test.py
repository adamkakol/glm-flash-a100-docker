"""Run with torchrun before starting the server; no model weights are loaded."""
from datetime import timedelta
import os
import time
import torch
import torch.distributed as dist

rank = int(os.environ["LOCAL_RANK"])
size = int(os.environ["WORLD_SIZE"])
torch.cuda.set_device(rank)
dist.init_process_group("nccl", timeout=timedelta(seconds=120))
try:
    for elements in [4096, 4 * 1024 * 1024]:
        payload = torch.ones(elements, dtype=torch.float32, device=rank)
        dist.all_reduce(payload)
        if not torch.all(payload == size).item():
            raise RuntimeError("NCCL reduction returned incorrect values.")
        for _ in range(10):
            payload.zero_()
            dist.all_reduce(payload)
        torch.cuda.synchronize()
        dist.barrier()
        started = time.perf_counter()
        for _ in range(50):
            dist.all_reduce(payload)
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - started) * 1000 / 50
        duration = torch.tensor([elapsed_ms], device=rank)
        dist.all_reduce(duration, op=dist.ReduceOp.MAX)
        if rank == 0:
            print(f"NCCL PASS: {size} GPUs, {elements * 4 / 2**20:.3f} MiB all-reduce, {duration.item():.3f} ms/iteration (slowest rank)", flush=True)
finally:
    dist.destroy_process_group()
