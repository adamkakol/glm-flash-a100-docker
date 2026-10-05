# Qwen3.8-27B on three A100 80GB GPUs

A staged vLLM benchmark and deployment launcher for two concurrent users with
**260,000 input tokens plus 32,768 generated tokens each**. It compares the
NVLink pair against independent replicas, retains image/video support, and
exports a deployment recommendation only after all required checks pass.

**Validation boundary:** the host logic and real HAProxy proxy have been tested
locally. No Qwen weights were downloaded and no A100 inference was performed
during development. The pinned model/runtime combination must pass this tool's
qualification on your server. A healthy container or a source-code review is
not evidence of two long sessions working well. There is no guaranteed token
speed and no automatic fallback to an unqualified configuration.

## Requirements and preparation

- Linux x86-64, Python 3.10+, Docker Engine, NVIDIA Container Toolkit.
- Three exclusively available A100 **80GB** cards, MIG disabled, exactly one
  bidirectional NVLink pair. GPU UUIDs are detected; host index order is irrelevant.
- An NVIDIA driver compatible with the pinned vLLM image. The explicit download
  command first runs a **model-free CUDA/BF16/P2P probe** inside that image and
  checks Transformers/model imports before downloading weights.
- Budget approximately 150 GB free disk for the two checkpoints, images and
  caches, and 128 GB or more system RAM. These are planning allowances, not
  measured minima; monitor host RAM while loading independent replicas.
- Stop your existing GLM service and disconnect clients before benchmarking.
  This tool refuses busy GPUs and never stops an unrelated service.

The host scripts use only Python's standard library. No local pip environment,
Transformers installation, or tokenizer download is needed.

```bash
# On the A100 server, inside the existing repository:
git pull --ff-only
cd models/qwen3.8-27b
python3 bench.py init
python3 bench.py plan
python3 bench.py preflight

# Explicit downloads occur only here, on the A100 server:
python3 bench.py download

# Run in tmux/screen so an SSH disconnect does not end your benchmark:
python3 bench.py run
```

`init` creates editable `config.json` and a private `secrets/api-key`. It refuses
to overwrite an existing config/key. Review [config.example.json](config.example.json)
for defaults. All relative paths in config are resolved against the config
file's directory. An alternate config uses `python3 bench.py --config /path/config.json run`.

`download` pulls immutable image digests and pinned official HF revisions. It
reuses the Hugging Face cache and can be rerun after an interrupted transfer.
`run` and `serve` use offline/read-only snapshots and `--pull never`: they cannot
silently fetch another model or upgrade the runtime. Weight shards must exist;
the loader validates their format. The tool does not reread all weight bytes
to independently SHA-256 verify the existing Hugging Face cache.

## What is compared

| Choice | Candidates |
|---|---|
| Weights | Official BF16 and official FP8 |
| Placement | TP2 on the NVLink pair; three independent TP1 replicas behind HAProxy |
| Prefill chunk | 2,048 and 8,192 tokens |
| MTP | Off during screening; off, one, or three draft tokens during refinement |
| Compute / KV | BF16 compute and BF16 KV (`--kv-cache-dtype auto` follows BF16) |
| Model context | 327,680 tokens total, with documented factor-2 YaRN extension |
| Active requests | Four per backend, plus a bounded waiting queue |
| Reasoning | `xhigh` for all performance comparisons |

On A100, FP8 is a **weight-storage/Marlin** comparison, not native FP8 arithmetic.
It can save memory and help decode while hurting prefill; the benchmark decides.
No Blackwell NVFP4 claim is transferred to A100. TP3 is excluded because the
model's KV/linear-attention head counts do not divide by three. PP3 is omitted
from this initial focused matrix: TP2 already supplies substantial expected
capacity. Independent replicas need one complete weight copy per GPU, but share
the checkpoint files on disk. There is no CPU weight offload.

The model's native **262,144-token limit includes output**. Increasing only
`max_model_len` would not implement the required extension. The launcher applies
YaRN under `text_config.rope_parameters`; supported configured ceilings are
292,768 through 524,288 with this factor-2 profile. The default 327,680 ceiling
leaves additional headroom beyond the qualification workload. Increasing it
requires a new run and might make a previously viable single-GPU replica fail.

The context limit protects resources; it does not limit the system to two
clients. Additional work can use available slots or queue. Four active requests
do **not** promise four full-length contexts. No strict per-user reservations
are implemented.

### Stages and acceptance

1. **Screen:** eight non-speculative configurations, each at two 8k inputs and
   512-token outputs, plus a short request arriving during both decodes.
2. **Refine:** the two selected configurations and their MTP variants at two
   65,536-token inputs and 1,024-token outputs. Preserve a viable BF16 reference
   when possible. Requested MTP must actually produce draft-token metrics.
3. **Qualify:** two finalists undergo repeated functional checks, simultaneous
   long-context retrieval, two trials of two distinct 260k inputs with **32k
   forced output tokens per user**, and a separate repeated-prefix trial. A
   third short request must emit output while both long sessions are decoding.

Before benchmarking, observed vLLM KV-capacity logs must cover two configured
windows for TP2, or one window **on every replica**. Unknown capacity fails.
Every configuration must pass text, image, native video, Chat Completions tool
calling, streamed Responses function-call/follow-up, and cancellation checks.
The video fixture is bundled; no remote media fetch is needed. After cancellation,
every backend must return to zero running/waiting requests within 30 seconds.

Prompt length is counted by the running server's `/tokenize` endpoint with its
chat template and checked against inference usage. A mismatch fails instead of
silently truncating input. Different users get different initial prompt content.
The repeated-prefix trial is reported separately; cache-hit counters establish
whether a replica actually reused its prefix. It is excluded from the cold-score
ranking, so routing luck cannot make it win.

Default acceptance budgets are **300 seconds to first useful output**,
**120 seconds between useful output events**, no observed KV preemption, and
at least **3 GiB sampled free VRAM on every participating GPU**. A dropped stream,
incomplete terminal status, missing usage, bad tool arguments, incorrect retrieval,
absent concurrent decode, metric outage or memory-margin failure rejects the
candidate. Comments/keepalives do not count as generated output. Sampling cannot
prove the absence of sub-sample memory spikes; actual request failure also rejects.

Successful candidates are ranked by a weighted geometric cost: 50% slower-user
seconds per output token, 25% worst long-request TTFT, 15% short-request TTFT,
10% worst useful-event pause. Final ranking uses the median of cold-trial costs.
Token counts come from usage, never SSE event counts. Decode rate spans the first
to last useful output; batched streaming events make this a delivery-rate estimate.

Forced-length stress output is a capacity/performance test, **not an intelligence
test**. Known-answer/tool/retrieval gates catch regressions, but do not establish
an Artificial Analysis score or broad coding capability. Review the candidate
on your own representative coding tasks before broad rollout. Qwen supports
text, images and video here; native audio is not a capability of this model.
The bundled visual probes are small functional tests, not a benchmark of large
documents, maximum-resolution image batches, or long videos.

Change latency budgets only to match deliberate client/SLA choices, then rerun.
Do not hide stalls by merely setting enormous timeouts. Long tests can take
hours because 32k output tokens per user must actually be generated; only
finalists receive that expensive workload.

## Progress, interruption and recovery

The terminal shows stage/candidate counters, a progress bar, current activity and
a heartbeat every 15 seconds. `progress.json` records per-request activity and
completed metrics. Counts are per stage, not a claim that the entire run is done.

```bash
python3 bench.py status
tail -f reports/run-*/progress.log

# Resume the specific run directory printed by the previous invocation:
python3 bench.py run --resume reports/run-YYYYMMDD-HHMMSS-ID

# Also retry failed candidates after fixing a transient host issue:
python3 bench.py run --resume reports/run-YYYYMMDD-HHMMSS-ID --retry-failed

# After SIGKILL/power failure: removes only matching owner-labelled containers.
# Resume performs this recovery automatically as well.
python3 bench.py cleanup reports/run-YYYYMMDD-HHMMSS-ID
python3 bench.py cleanup state/download
```

Completed results survive interruption. Changing configuration, code, GPU UUIDs,
topology or driver requires a fresh run. Each candidate has `result.json`, bounded
container logs and timestamped `telemetry.jsonl`; the run has `run.json` and
`progress.json`. No recommendation is written if all finalists fail.

NVLink detection plus CUDA peer access verifies the selected pair, not actual
traffic. NVLink counters are sampled where the driver supports them. Inspect
their time series under inference load, or use DCGM/Nsight if unavailable. The
report does not label NVLink traffic verified solely from topology or nonzero
historical counters.

## Deploy the measured winner

```bash
python3 bench.py serve reports/run-YYYYMMDD-HHMMSS-ID/recommendation.json --port 8000
python3 bench.py status
python3 bench.py stop
```

`serve` requires a successful recommendation with matching code/config/hardware,
starts the selected topology, reruns functional checks, and only then records
the active endpoint. Failure cleans up the attempted deployment. Docker uses
`restart=unless-stopped`; enable the Docker service at boot on your host. There
is no automatic upgrade, rollout replacement, or benchmark-driven switch of a
running service. `stop` and `cleanup` work even if config or API-key files were lost.

Endpoint: `http://127.0.0.1:8000/v1`, model name **`qwen3.8-27b`**. Read the API
key from `secrets/api-key`; it is never printed in a plan or report. Backends bind
only to loopback on ports 18100–18102; the benchmark gateway uses port 18080.
Choose different ports in config if needed **before** benchmarking.

HAProxy is pinned, streams without response buffering, requires authentication,
limits request bodies to 64 MiB, and has bounded admission: four active and eight
queued requests per backend, a 30-second queue timeout, and a 64-connection
frontend ceiling. Excess load receives HTTP 503 or waits within those bounds.
POST requests are never automatically replayed. Request uploads must include
Content-Length (normal JSON SDK requests do); chunked uploads are rejected.
Clients should handle overload deliberately rather than retrying immediately.

Responses clients must send **full conversation history with `store:false`**.
Cross-replica `previous_response_id` state, WebSockets, hosted web search,
server-side file storage, and built-in computer/shell tools are not supported
by this deployment. Client-executed function tools are tested. This is a local
trusted-client service; remote access should use an SSH tunnel or your existing
authenticated TLS ingress, not an unprotected public port.

For a remote client, forward the endpoint:

```bash
ssh -N -L 8000:127.0.0.1:8000 YOUR_USER@YOUR_A100_SERVER
```

### Codex connection

Merge provider settings into the **user-level** `~/.codex/config.toml`; the tool
does not modify your PC's Codex configuration. Set `QWEN_API_KEY` in the client
environment to the server key through your normal secret-management mechanism.

```toml
model = "qwen3.8-27b"
model_provider = "a100_qwen"
model_context_window = 327680
model_reasoning_effort = "xhigh"

[model_providers.a100_qwen]
name = "Qwen on A100"
base_url = "http://127.0.0.1:8000/v1"
env_key = "QWEN_API_KEY"
wire_api = "responses"
requires_openai_auth = false
supports_websockets = false
stream_idle_timeout_ms = 300000
request_max_retries = 0
stream_max_retries = 0
```

Start with retries disabled so protocol failures remain visible. Verify a real
multi-turn coding task with your installed Codex version: the protocol tests do
not simulate every Codex feature or replace a client-level acceptance test.
Older/newer client model metadata can also require a custom model catalogue.

## Tests and pinned sources

From the repository root, without GPUs or weights:

The existing GLM tests additionally require PyYAML and ffmpeg. The Qwen host
tests use the Python standard library; Docker is only needed for the optional
real proxy tests.

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -v
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s models/qwen3.8-27b/tests -v
```

The real HAProxy integration suite is opt-in. It requires only the small pinned
proxy image and local mock HTTP workers; it never pulls an image itself. See
[test_gateway_integration.py](tests/test_gateway_integration.py) and the CI workflow.
Temporary test files are removed automatically. Reports, model caches, keys and
state are git-ignored; actual benchmark reports/caches remain on the server for
inspection/resume and are not automatically deleted.

Reviewed pins and sources, 2026-10-05:

- [Official BF16 model and YaRN guidance](https://huggingface.co/Qwen/Qwen3.8-27B),
  revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`.
- [Official FP8 model](https://huggingface.co/Qwen/Qwen3.8-27B-FP8),
  revision `017b9c7af6b5689d5dd426a76e0bc077eb5ca20a`.
- [vLLM Qwen recipe](https://recipes.vllm.ai/Qwen/Qwen3.8-27B) and
  [vLLM 0.30.0 source](https://github.com/vllm-project/vllm/tree/v0.30.0).
- [HAProxy 3.2 LTS](https://www.haproxy.org/) at 3.2.25.
- [Official Codex provider settings](https://learn.chatgpt.com/docs/config-file/config-reference)
  and [gateway compatibility requirements](https://learn.chatgpt.com/docs/enterprise/gateway-compatibility).

Image digests are checked into `config.example.json` and runtime defaults. The
two official Qwen checkpoints are Apache-2.0 licensed; preserve their notices
when redistributing. No model files are included in this repository.
