# GLM-5.3-Flash

The original GLM deployment remains at the repository root so an existing
installation can continue using its current `docker compose`, configuration,
model cache, API keys and autotuner commands without migration.

- [Deployment guide](../../README.md#existing-glm-53-flash-deployment)
- [Docker Compose](../../compose.yaml)
- [Configurator](../../configure.py)
- [Autotuner](../../autotune.py)
- [Runtime and validation scripts](../../scripts/)

This folder is the GLM entry in the model catalogue. It intentionally references
the original implementation instead of maintaining a second copy. The separate
[Qwen3.8-27B deployment](../qwen3.8-27b/README.md) uses different containers,
ports, files and an explicit server-side download step.
