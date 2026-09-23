# Validation record — 2026-09-23

Completed locally:

- Parsed all Python scripts and YAML/JSON files.
- Validated every TabbyAPI setting against the configuration model in commit
  `f07131cd8fe34e449fe87cdd3a066b52b96d3cac`; no unknown settings.
- Validated Compose against the official Compose specification JSON schema,
  before and after environment-variable substitution.
- Exercised the configuration CLI with a simulated three-card inventory whose
  NVLink pair is GPU 0 and GPU 2. Verified UUID order consistently reaches the
  Docker device reservation and CUDA environment.
- Checked rejection of 40GB cards, enabled MIG, an older driver, and a missing
  bridge. Checked user-key uniqueness, file permissions, retained keys on
  reconfiguration, and all three allocation modes.
- Checked dependency rendering selects exactly one Linux/Python 3.12 PyTorch
  2.9.0 CUDA 12.8 wheel and the SHA256-verified ExLlamaV3 1.5.1 wheel.
- Checked model manifest paths, sizes and SHA256 formats. Exercised detection
  of same-size file corruption with a small fixture.
- Exercised the API client against a local mock HTTP server: smoke requests,
  token counting, two concurrent streams, reasoning/content events, final
  usage, completion status, report writing and rejection of truncated input
  counts. This was not an LLM benchmark.
- Confirmed the inspected allocator, MLA and KDA code match the pinned
  ExLlamaV3 1.5.1 release. Its release notes explicitly include GLM-5.3 tensor
  parallel support.

Not performed here:

- Docker image build or startup (no Docker runtime available).
- CUDA kernel execution, actual NVLink/P2P/NCCL operation or throughput.
- Download/loading of the 165 GB weights and measurement of peak host/GPU RAM.
- Real 250k-token inference, concurrent decoding, or long-context quality tests.

Run the commands in README.md on the target server before accepting the
capacity and latency of this deployment. The package contains the deployment
recipe and checks, not a prebuilt image or model weights.
