# GLM-5.3-Flash on three A100 80GB GPUs

Docker deployment for `turboderp/GLM-5.3-Flash-exl3`, **4.05bpw**, using
**ExLlamaV3 1.5.1 + TabbyAPI + PyTorch 2.9.0 / CUDA 12.8**. Prepared 2026-09-23.

This bundle targets **three full A100 80GB GPUs**, with two connected by NVLink.
It is not sized for three 40GB A100s. GPU memory, MIG mode, bridge topology,
CUDA visibility and peer access are checked before loading the model.

**Validation status:** configuration and scripts have been checked against the
pinned upstream source. This environment has no Docker daemon or NVIDIA GPU,
so the image build, model fit, GPU kernels and two simultaneous 250k-token
requests still need to pass the included checks on your server. These are
deployment settings, not a measured performance or capacity guarantee.

## How the bridge is used

One model instance spans all three GPUs. The default enables ExLlama's tensor
parallel mode with the NCCL communication backend. NCCL can use the NVLink
connection between the paired cards and PCIe or host transport to the third
card, according to the actual topology and peer access support.

ExLlamaV3 1.5.1 specifically added tensor parallel support for GLM-5.3.
Its allocator supports uneven splits; it does not require all head counts to
divide evenly by three. Some components, including MLA attention blocks in
this version, stay whole on one GPU while other components are split.

The third card can limit synchronization speed. This is **not** a special
“TP=2 plus one extra memory GPU” arrangement, and the bridge does not combine
two cards into one CUDA device. All three participate in inference. The full
checkpoint is about **165.15 GB / 153.81 GiB on disk**; keeping it on the pair
alone would leave inadequate planning headroom for two long contexts and the
runtime. Actual loaded VRAM differs from file size.

The configuration intentionally leaves `NCCL_P2P_LEVEL`, `NCCL_P2P_DISABLE`,
`NCCL_ALGO` and `NCCL_TOPO_FILE` unset. In particular, forcing
`NCCL_P2P_LEVEL=NVL` would exclude the third card's PCIe P2P paths.
NCCL handles reductions; ExLlama also uses its native backend for some other
communication operations. A bridge alone does not guarantee a speedup.

## Settings

| Setting | Value |
|---|---|
| Quantization | EXL3 4.05 bits/weight checkpoint |
| Model instances | 1, shared by both users |
| Context per request, input plus output | 327,680 tokens |
| Shared token cache | 655,360 tokens |
| Maximum active generation jobs | 2 |
| Cache mode | FP16; no additional cache quantization |
| Prefill chunk | 2,048 tokens |
| VRAM reserved outside model allocation | 4 GiB per card |
| Reasoning effort default | max; clients can override |
| Vision / speculative decoding | Disabled |
| API | OpenAI-compatible, `http://127.0.0.1:5000/v1` |

250,000 input tokens leave 77,680 tokens for chat formatting and output within
the configured window. Thinking tokens also consume output/context space.
The cache budget is shared across jobs: setting only `max_seq_len` would not
provide capacity for two complete windows. More clients can queue; this is
not an HTTP admission limit of two connections.

## Host prerequisites

- Linux x86-64, Docker Engine and the Docker Compose v2 plugin.
- NVIDIA Container Toolkit configured for Docker, and driver **570.124.06 or
  newer**. This bundle uses that conservative driver floor for CUDA 12.8.1 and
  Triton JIT; it does not rely on older-driver minor-version compatibility.
- Three A100 80GB GPUs available exclusively to this service, MIG disabled.
- Approximately **250 GB free SSD space** for weights, image and caches;
  **128 GB system RAM recommended** as a planning allowance, not a measured
  minimum. Model weights remain on GPUs during serving.
- Python 3.10+ on the host for the configuration and API check scripts.

Follow NVIDIA's [Container Toolkit installation guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
if GPU access from Docker is not already configured. Do not install the GPU
driver inside this image. Configuring Docker or updating drivers is a host
administration task; this bundle does not change them automatically.

## Build and start

Clone the repository and run these commands as your normal Docker user.

```bash
git clone https://github.com/adamkakol/glm-flash-a100-docker.git
cd glm-flash-a100-docker
python3 configure.py
docker compose build

# Small GPU checks before downloading the 165 GB checkpoint:
docker compose run --rm --entrypoint python server /deploy/preflight.py
docker compose run --rm -e NCCL_DEBUG=INFO -e NCCL_DEBUG_SUBSYS=INIT,GRAPH,P2P --entrypoint torchrun server --standalone --nproc_per_node=3 /deploy/nccl_test.py

# Download the immutable revision; rerunning resumes an interrupted download.
docker compose run --rm --entrypoint python server /deploy/download_model.py --verify-hashes

docker compose up -d
docker compose logs -f server
```

`configure.py` detects the NVLink pair and orders the GPUs by UUID. If the host
has more than three GPUs, select the intended ones with
`python3 configure.py --gpus 0,1,2` using **host `nvidia-smi` indices**. The pair
does not have to be GPU 0 and GPU 1. It creates local directories, `.env`,
`config.yml`, two user API keys and a separate admin key. It keeps existing keys
on subsequent runs. No real credentials or weights are shipped in this bundle.

The Docker build follows the pinned upstream CUDA-12 dependency set. The CUDA
base digest, Tabby source archive and ExLlama wheel are checked/pinned, and the
model is pinned by its complete Hugging Face commit hash. Supporting Python
and Ubuntu packages are resolved at build time; this is not a fully frozen
transitive dependency lock. The resolved Python packages are recorded inside
the image at `/opt/runtime-requirements.txt`. Retain your built image for exact
runtime reuse.

First load and Triton compilation can take several minutes. The 30-minute
healthcheck start period accommodates slow initialization; it is not a promise
of load time. Startup runs preflight again and checks all model file sizes and
the revision marker. It never silently downloads a different model.

## Verify the model and two long requests

After the service is healthy:

```bash
docker compose ps
python3 scripts/api_check.py smoke
python3 scripts/api_check.py long --tokens 250000 --users 2
```

The long check builds two different synthetic prompts, tokenizes them through
the server, and sends both requests concurrently. It checks the **reported
input token counts** and records time to first token, throughput when reported,
completion status and generation interval overlap in `reports/api-check.json`.
It can take a long time on A100s. Run it before relying on this service for
two simultaneous large documents.

Watch memory and GPU utilization in another terminal:

```bash
nvidia-smi --query-gpu=index,uuid,memory.used,memory.total,utilization.gpu --format=csv -l 1
```

Acceptance means the image builds, preflight and the three-GPU NCCL test pass,
the correct model becomes healthy, and both >=250k input requests complete
without GPU errors or input truncation. Check generation overlap and latency
against your users' requirements. Overlapping HTTP requests alone do not prove
concurrent decoding. The synthetic check establishes capacity and basic
execution; evaluate representative reasoning, coding and retrieval tasks
separately to judge quality at 250k.

## Client access

Use the user keys in `secrets/api_tokens.yml`, with model name
`GLM-5.3-Flash-exl3-4.05bpw` and a normal bearer `Authorization` header.
The keys file is JSON syntax, which is also valid YAML; preserve that syntax
when editing so the dependency-free host test client can read it.
The admin key permits model management; give users the user keys.

The default host binding is localhost. For a remote workstation:

```bash
ssh -N -L 5000:127.0.0.1:5000 your-user@your-server
```

Then point the workstation client at `http://127.0.0.1:5000/v1`. For a shared
LAN deployment, change the bind deliberately and use your existing authenticated
TLS proxy/firewall. Give that proxy long timeouts for prefill and disable SSE
buffering. Tabby emits SSE keepalives during long requests.

TabbyAPI's pinned upstream version prints API/admin keys at startup, so Docker
log access must be limited to the operators of this service. Prompt/request
logging is disabled here.

## Compare performance on the actual topology

Before serving users, also try native tensor parallel communication or layer
splitting. With only 1–2 users, avoiding frequent communication with the third
card may outweigh using all GPUs in parallel. These modes keep the **same
weights, cache precision and context limits**.

```bash
docker compose down
python3 configure.py --mode layer --force
docker compose up -d
# Wait for health; rerun the same smoke/long checks and save a separate report:
python3 scripts/api_check.py long --users 2 --report reports/layer.json
```

Use `--mode native` for ExLlama's native TP backend or `--mode nccl` to return
to the default. `--force` regenerates `.env` and `config.yml`, replacing manual
edits but retaining model data and keys; repeat `--gpus` / `--port` if customized.
Layer splitting still benefits from peer transfers where the assigned layers
cross the bridged pair, but performs less work across GPUs simultaneously.
Select the mode using measured prefill time and decode throughput.

For a bridge-only NCCL diagnostic, run the same `nccl_test.py` command with
`--nproc_per_node=2` **while the server is stopped**. UUID order ensures those
two ranks use the paired cards. This tests communication, not model fit on two
cards. Do not run the GPU diagnostics while a loaded model is using the GPUs.

If loading fails with OOM, first confirm the GPUs are idle and read the logged
allocation. Try `chunk_size: 1024` or layer mode while retaining the context
budget. Do not reduce cache size below two context windows and still claim
two full-context users. If the full test cannot pass, report it as a failed
capacity target; reducing context, precision or concurrency is a different
deployment decision. If P2P/NCCL fails, diagnose the host topology, VM
passthrough and driver/container runtime before changing transport settings.

## Sources and licenses

- [ExLlamaV3 1.5.1 release: GLM-5.3 TP support](https://github.com/turboderp-org/exllamav3/releases/tag/v1.5.1)
- [Pinned ExLlama allocator](https://github.com/turboderp-org/exllamav3/blob/958ec933361b24eb8426ec7222e5b0062a679dcd/exllamav3/model/model_tp_alloc.py)
- [Pinned TabbyAPI configuration](https://github.com/theroyallab/tabbyAPI/blob/f07131cd8fe34e449fe87cdd3a066b52b96d3cac/config_sample.yml)
- [Pinned upstream Dockerfile](https://github.com/theroyallab/tabbyAPI/blob/f07131cd8fe34e449fe87cdd3a066b52b96d3cac/docker/Dockerfile)
- [Quantized checkpoint](https://huggingface.co/turboderp/GLM-5.3-Flash-exl3/tree/2a30229e67012798ba9f0cd832bb78abf4c363d5)
- [NCCL transport settings](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#nccl-p2p-level)
- [CUDA 12.8.1 driver release notes](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-toolkit-release-notes/index.html)

GLM-5.3-Flash and this EXL3 checkpoint are MIT licensed, permitting commercial
use subject to the license notices. ExLlamaV3 is MIT. TabbyAPI is AGPL-3.0;
its license obligations are separate from the model license, particularly if
you modify the server and provide network access. The image contains upstream
source and notices; the model download retains its LICENSE file. CUDA/PyTorch
and other dependencies retain their respective licenses.

These deployment scripts do not patch the inference server. See the upstream
[model license](https://huggingface.co/zai-org/GLM-5.3-Flash/blob/main/LICENSE),
[ExLlama license](https://github.com/turboderp-org/exllamav3/blob/v1.5.1/LICENSE)
and [TabbyAPI license](https://github.com/theroyallab/tabbyAPI/blob/f07131cd8fe34e449fe87cdd3a066b52b96d3cac/LICENSE)
for their full terms.
