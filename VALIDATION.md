# Validation record — 2026-09-23

This revision adds a shared-cache policy, performance-based hardware tuner,
image input and stricter NVLink topology checks.
No A100 benchmark result or universally fastest profile is claimed.

Completed for this update (29 passing CPU tests):

- CPU-only unit tests for shared cache with more than two active slots, context
  guards, all config modes and draft settings, and preservation of GPU UUIDs,
  port and existing keys on reconfiguration.
- Ranking tests showing that performance determines the winner among eligible
  configurations; failed profiles cannot win, inefficient context expansion
  is rejected, and a failed boundary probe falls back to the next candidate.
- Recovery tests for interruptions and restoration of original config/stopped
  service state.
- Real HTTP and SSE parsing against a local mock server, including delayed
  arrival of extra sessions, reasoning/content events, usage and timestamps,
  and calibration of the generation-prefix token count.
- Regression tests rejecting the previous false passes: sequential one-token
  responses, truncated prompt counts, and an extra session that only starts
  after the long sessions finish.
- Health probe checks requiring the engine health endpoint, not just metadata.
- Vision profile migration and explicit opt-out; health fails if vision is
  configured but the loaded model reports it disabled.
- Image probes reject ignored images; mixed HTTP fixtures verify uncached image
  requests overlap long-session decoding and count image tokens.
- NVLink pair detection at every pair position, noncontiguous indices, UUID
  mapping, and rejection of asymmetric/missing links or the wrong pair.
- Real FFmpeg extraction from a tiny locally generated synthetic video, testing
  bounded sampling, timestamped embedded frames, and cleanup. This loads no model.
- Python syntax and CLI plan checks, plus upstream configuration/schema review.

The existing immutable upstream pins remain unchanged. The initial review also
verified the Tabby source archive hash and all 31 model-manifest entries against
upstream metadata, including every declared large-file hash. This did not
involve downloading model weights.

Not performed for this update:

- Docker image build or inference-container startup.
- Model weight download or load, A100 CUDA kernels, real NVLink/NCCL throughput.
- Actual 260k concurrency, long-output or 512k/1M boundary runs.
- Actual image understanding, vision GPU memory peaks, native video or audio.
- Long-context reasoning quality, calibrated quantization comparisons, or soak tests.

Run autotune.py on the target server to obtain actual evidence. It saves
candidate results and the chosen configuration locally under reports/.
A model-loading success, free VRAM measurement or passing mock test alone is
not evidence that a deployment meets the intended latency or quality target.

The video helper sends sampled still images; it does not enable native temporal
video inference. NVLink detection proves topology, not measured use of the bridge.
The pinned native TP reductions use host memory; NCCL also has native fallbacks.
