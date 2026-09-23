# syntax=docker/dockerfile:1
FROM nvidia/cuda:12.8.1-runtime-ubuntu24.04@sha256:ebef3c171eeef0298e4eb2e4be843105edf3b8b0ac45e0b43acee358e8046867

ARG TABBY_COMMIT=f07131cd8fe34e449fe87cdd3a066b52b96d3cac
ARG TABBY_ARCHIVE_SHA256=78452c8a412cfe8ba19294a73bc0d2358a8cc85e6c79e6bddee74938daa3e4cb
ENV DEBIAN_FRONTEND=noninteractive \
    PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/cache/huggingface \
    XDG_CACHE_HOME=/cache \
    TRITON_CACHE_DIR=/cache/triton \
    TORCH_EXTENSIONS_DIR=/cache/torch_extensions \
    MAX_JOBS=4

# Python headers and a C compiler are needed by Triton's runtime JIT.
# ExLlama itself uses the upstream prebuilt CUDA 12.8 wheel.
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential ca-certificates curl python3.12 python3.12-dev \
      python3.12-venv \
    && rm -rf /var/lib/apt/lists/* \
    && python3.12 -m venv /opt/venv

WORKDIR /app
RUN curl --fail --location --retry 3 \
      "https://codeload.github.com/theroyallab/tabbyAPI/tar.gz/${TABBY_COMMIT}" \
      -o /tmp/tabby.tar.gz \
    && echo "${TABBY_ARCHIVE_SHA256}  /tmp/tabby.tar.gz" | sha256sum --check - \
    && tar -xzf /tmp/tabby.tar.gz --strip-components=1 -C /app \
    && rm /tmp/tabby.tar.gz

COPY scripts/ /deploy/
COPY versions.json model-manifest.json /deploy/
RUN python /deploy/install_requirements.py \
    && pip install --no-cache-dir -r /tmp/requirements.in \
    && pip check \
    && python -c "import importlib.metadata as m; import torch; assert m.version('exllamav3').split('+')[0] == '1.5.1'; assert torch.__version__.split('+')[0] == '2.9.0'; assert torch.version.cuda == '12.8'" \
    && pip freeze > /opt/runtime-requirements.txt

LABEL org.opencontainers.image.source="https://github.com/theroyallab/tabbyAPI" \
      org.opencontainers.image.revision="f07131cd8fe34e449fe87cdd3a066b52b96d3cac"
EXPOSE 5000
ENTRYPOINT ["python", "/deploy/serve.py"]
