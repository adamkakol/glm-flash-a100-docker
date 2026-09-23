# GLM-5.3-Flash on three A100 80GB GPUs

Deploy `turboderp/GLM-5.3-Flash-exl3` **4.05bpw** with pinned **ExLlamaV3 1.5.1,
TabbyAPI, and PyTorch 2.9.0 / CUDA 12.8**. Two of the three cards are connected
by NVLink. All three participate in one shared model instance.

The primary workload is **two concurrent sessions with approximately 260,000
input tokens each, plus generation**. This is a performance target, not a
limit of two clients or a fixed 260k context ceiling. Shorter sessions can use
spare capacity, and additional requests can queue.

**Validation:** host-side tests pass, including real HTTP/SSE parsing against a
mock server. The target A100 server has not been benchmarked by this project.
The autotuner must run there to establish fit and select a measured configuration.
No model weights are included in this repository.

## Resource limits

| Setting | Initial value | Purpose |
|---|---:|---|
| Per-request context, input plus output | 524,288 | Allows requests beyond the 260k target |
| Shared token cache | 1,048,576 | Pooled across requests; not reserved in full per user |
| Maximum active generation jobs | 4 | Allows a third and fourth request when cache permits |
| Cache precision | FP16 | Preserve quality initially |
| Allocator reserve per GPU | 4 GiB | Space outside planned model/cache allocations |
| Prefill chunk | 2,048 | Starting point; tuner compares alternatives |
| Reasoning default | `max` | Clients may override |
| Vision | Disabled | Text serving baseline |

The scheduler starts requests when both active slots and cache pages are
available. It may admit a smaller request while a larger one waits, with a
skip limit for approximate fairness. **Four slots do not promise four full
512k windows at once.** There is no per-user reservation: a user can submit
more than one request. The default two API keys do not limit client count.

The cache and batch limits bound planned GPU allocations. They do not bound
HTTP queue length or guarantee latency under unlimited submitted work. Use
trusted clients and, for broader exposure, an authenticated proxy with a
bounded request queue. Avoid flooding the service with queued giant prompts.

A complete 260,000-token prompt plus 65,536 output tokens fits inside the
initial window. System messages, tool definitions, chat formatting and
reasoning all count. A single request is always capped by the configured
context and the model's native 1,048,576-token limit.

The initial shared FP16 attention cache is approximately **17.19 GiB total**
for this model's 11 sparse MLA layers; its 34 linear-attention layers use
recurrent state. This is a source-derived estimate, excluding weights,
temporary buffers, recurrent history, CUDA graphs and placement imbalance.
The complete download is approximately 165.15 GB / 153.81 GiB, including MTP
weights. Download size is not resident GPU memory.

## Why the GPU layout needs a benchmark

The NVLink pair is ordered first by GPU UUID. NCCL can use its bridge and the
available PCIe/host paths to the third GPU. Do not force
`NCCL_P2P_LEVEL=NVL`: that excludes PCIe P2P paths to the third card.

The three supported allocation modes are:

- `nccl`: tensor parallelism using NCCL reductions, with native fallbacks for
  some operations.
- `native`: ExLlama's native tensor parallel communication backend.
- `layer`: layer splitting, which reduces cross-GPU coordination but runs less
  work across devices simultaneously.

The pinned ExLlama allocator supports uneven channel splits across three
cards. Some modules, including MLA attention in this version, stay whole on
one card. A two-card NVLink bridge does not make these three GPUs a uniform
interconnect, so topology alone cannot identify the fastest mode.

## Setup on the A100 server

Requirements: Linux x86-64, three exclusive A100 **80GB** GPUs with MIG disabled,
NVIDIA driver >=570.124.06, NVIDIA Container Toolkit, Docker Engine and Compose
v2, and host Python 3.10+. Plan for roughly 250 GB free SSD storage and 128 GB
system RAM; these are allowances, not measured minima.

```bash
git clone https://github.com/adamkakol/glm-flash-a100-docker.git
cd glm-flash-a100-docker
python3 configure.py
docker compose build

# Checks before downloading weights:
docker compose run --rm --entrypoint python server /deploy/preflight.py
docker compose run --rm -e NCCL_DEBUG=INFO --entrypoint torchrun server --standalone --nproc_per_node=3 /deploy/nccl_test.py

# The ONLY model download step; run it on the A100 server:
docker compose run --rm --entrypoint python server /deploy/download_model.py --verify-hashes

docker compose up -d --wait --wait-timeout 1800
python3 scripts/api_check.py smoke
```

`configure.py --gpus 0,1,2` selects host `nvidia-smi` indices if more than three
GPUs exist. The NVLink pair may use any two indices. Configuration creates
`.env`, `config.yml`, `deployment.json`, two user keys and a separate admin key.
Use the user keys from `secrets/api_tokens.yml`, model
`GLM-5.3-Flash-exl3-4.05bpw`, and `http://127.0.0.1:5000/v1`.

For an existing checkout, run `git pull`, stop the service, then
`python3 configure.py --force` and rebuild the image. Old two-slot configuration
is replaced by the new defaults if no `deployment.json` exists. Subsequent
`--force` runs retain recorded profile settings, selected GPU UUIDs, port and
existing keys. Explicit options override those values. Manual `config.yml`
edits are replaced; copy them before regeneration.

```bash
python3 configure.py --force --mode native --max-batch-size 4 \
  --max-seq-len 524288 --cache-size 1048576 --chunk-size 2048 --draft-tokens 0
```

Configuration guards allow 3–8 active slots and require enough shared cache
for two target windows (655,360 tokens total), without requiring
`cache_size >= max_batch_size * max_seq_len`. Manual increases still require
hardware testing. Never run diagnostics or tuning alongside another GPU workload.

## Select an efficient configuration automatically

After the image and pinned model have been installed on the A100 server:

```bash
# Inspection only: does not start containers or load/download anything.
python3 autotune.py --plan

# Maintenance operation: disconnect clients first.
python3 autotune.py
```

The default run measures **18 configurations**: three allocation modes ×
2048/4096 prefill chunks × MTP disabled/one draft token/two draft tokens.
Every candidate gets a smoke check and **two repetitions** of both workloads:

1. Two distinct 260k prompts, each producing at least 4,096 tokens, with
   sustained overlapping output required.
2. The same two long sessions, followed by extra short requests **after both
   long sessions begin generating**. With the default four slots, two extra
   sessions join. They must produce output while both long sessions remain
   active. The benchmark measures their TTFT and the long sessions' delivery
   gaps, exposing interference.

Candidates are ranked by a normalized geometric score, lower is better:
**45% slower-user decode time per token, 25% worst long-session TTFT,
20% extra-user TTFT, and 10% worst long-session delivery pause in the mixed
test.** Repeated trials use median metrics. Capacity is only eligibility;
a configuration that merely loads cannot win. Failed inference, truncated
input, short output, absent overlap, failed health or less than **3 GiB sampled
free VRAM on any selected GPU** excludes a candidate. Sampling every 0.5 seconds
is an observation, not a guarantee against sub-sample memory spikes.

The best candidate must also pass a single request near its advertised context
ceiling. If it fails, the tuner tries the next ranked candidate. It then tests
an optional expansion to **1,048,576 context / 1,572,864 shared cache**. Expansion
is retained only if it passes the same workload and boundary checks, preserves
headroom, and none of the four scored costs regresses by more than 5%.
Use `--no-expand` to skip this additional experiment. Larger cache alone is
not considered an optimization.

The selected profile is saved and started automatically. Reports, per-stream
timestamps, actual token counts, memory peaks, mode, package pins, image ID,
hardware/driver information and the original configuration go under
`reports/autotune-*/`. A failed or interrupted run restores the previous config
and whether the service was running. SIGKILL, power loss and machine crashes
cannot execute that rollback; use `original-config.yml` and
`original-deployment.json` from the report directory for manual recovery.
No model or image is downloaded by the tuner; it uses `--no-build --pull never`.

**This can take many hours**, especially the near-1M boundary run. An initial,
narrower comparison is available; its winner is only the best of those candidates:

```bash
python3 autotune.py --chunks 2048 --draft-tokens 0 --repeats 1 --no-expand
```

For a more representative workload, provide your own text/code body:

```bash
python3 autotune.py --corpus /path/to/representative-code.txt
```

The text is repeated to reach the target lengths. New unique prefixes prevent
cross-request prefix sharing and accidental warm-cache comparisons. Defaults
use synthetic records. Forced-length output measures serving performance,
not answer quality. The default benchmark uses `reasoning_effort=max`, matching
production; `--reasoning-effort high` is available if that is your actual workload.

This selects the **best measured eligible configuration in the chosen search**,
not a universal optimum. Re-run after changing the model, engine, driver,
GPU placement or workload. It does not switch allocation mode during live
requests or change precision/reasoning quality to improve its score.

## Independent checks and quality

```bash
python3 scripts/api_check.py long --tokens 260000 --users 2
python3 scripts/api_check.py mixed --tokens 260000 --users 4
# A longer generation-budget test; this forces substantial actual output:
python3 scripts/api_check.py long --tokens 260000 --users 2 --max-output 32768
```

Before constructing long prompts, the checker calibrates the tokenization
endpoint against actual chat usage: the endpoint omits the assistant generation
prefix. Reported prompt lengths include that measured difference.

Long and mixed tests default to a minimum output equal to `--max-output`.
`--min-output` can override the minimum. Missing overlap, too little output or
input truncation produces a failed report and nonzero exit status. These are
capacity tests, not model-quality tests. `min_tokens` suppresses EOS but other
stop conditions may still end output; actual counts are always checked.
Reported inter-event gaps are SSE delivery gaps, not exact per-token latency
when the server groups multiple tokens in one event.

Keep 4.05bpw weights and FP16 cache as the baseline. Default sampling is
`temperature=1.0, top_p=0.95`, with neutral top-k/min-p/repetition penalties;
clients may override it. `clear_thinking=true` is explicit for chat.
Test real retrieval, reasoning, coding and tool workflows separately, including
facts near the start, middle and end of long histories. Also evaluate follow-up
turns and prefix reuse; cold-prompt tuning does not characterize every chat.

MTP has target-model verification, but its practical speedup is measured rather
than assumed. It adds draft weights, cache and recurrent-history storage. If
plain 4.05bpw quality is inadequate, compare a separately evaluated checkpoint;
the tuner deliberately does not select more aggressive quantization.

## Operations and validation scope

Warmup shifts initial JIT/autotuning toward model loading. The healthcheck
checks both `/health` and the expected model/configuration. Docker health status
alone does not restart a running process; investigate an unhealthy service and
restart it if recovery fails. The ordinary restart policy handles process exits.

The API binds to localhost. For remote access, use an SSH tunnel:

```bash
ssh -N -L 5000:127.0.0.1:5000 your-user@your-server
```

For shared LAN access use an authenticated TLS proxy, long prefill timeouts and
unbuffered SSE. The pinned Tabby version prints API/admin keys at startup;
limit access to Docker logs. Prompt logging is disabled.

Source, wheel, CUDA image and model revisions/hashes are pinned. Supporting
packages still resolve at build time. Preserve a successful image by digest;
`/opt/runtime-requirements.txt` records its resolved Python packages. This is
not a complete transitive reproducibility lock.

Run CPU-only development checks with Python and PyYAML installed:

```bash
python3 -m unittest discover -s tests -v
python3 autotune.py --plan
```

See [VALIDATION.md](VALIDATION.md) for what was actually checked. Unit tests and
mock HTTP streams are not evidence of A100 throughput or model quality.

## Sources and licenses

- [ExLlamaV3 1.5.1: GLM-5.3 TP support](https://github.com/turboderp-org/exllamav3/releases/tag/v1.5.1)
- [Pinned allocator](https://github.com/turboderp-org/exllamav3/blob/958ec933361b24eb8426ec7222e5b0062a679dcd/exllamav3/model/model_tp_alloc.py)
- [Pinned shared-cache scheduler](https://github.com/turboderp-org/exllamav3/blob/958ec933361b24eb8426ec7222e5b0062a679dcd/exllamav3/generator/generator.py)
- [Pinned Tabby settings](https://github.com/theroyallab/tabbyAPI/blob/f07131cd8fe34e449fe87cdd3a066b52b96d3cac/config_sample.yml)
- [Model guidance](https://huggingface.co/zai-org/GLM-5.3-Flash)
- [Pinned quantized checkpoint](https://huggingface.co/turboderp/GLM-5.3-Flash-exl3/tree/2a30229e67012798ba9f0cd832bb78abf4c363d5)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- [NCCL settings](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#nccl-p2p-level)

The model/checkpoint and ExLlamaV3 are MIT licensed. TabbyAPI is AGPL-3.0;
its terms are separate from the model license. This repository configures the
server without patching its source. The downloaded model retains its LICENSE,
and the image retains upstream source/notices. CUDA, PyTorch and other
components retain their respective licenses.
