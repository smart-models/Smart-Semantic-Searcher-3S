# syntax=docker/dockerfile:1.7

# Middleware dependency environment.
FROM python:3.11-slim-bookworm@sha256:b18992999dbe963a45a8a4da40ac2b1975be1a776d939d098c647482bcad5cba AS middleware-builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN python3 -m venv --copies /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
WORKDIR /app
COPY requirements/ requirements/
RUN pip install --upgrade \
        -c requirements/locks/middleware-py311-linux.txt \
        pip==26.2.0 \
        setuptools==84.2.1 \
        wheel==0.46.2 \
    && pip install --no-compile \
        -r requirements/middleware.txt \
        -c requirements/locks/middleware-py311-linux.txt

# Middleware runtime.
FROM python:3.11-slim-bookworm@sha256:b18992999dbe963a45a8a4da40ac2b1975be1a776d939d098c647482bcad5cba AS middleware

LABEL org.opencontainers.image.title="hybrid-manager-middleware" \
      org.opencontainers.image.description="Smart Semantic Searcher FastAPI middleware" \
      org.opencontainers.image.licenses="AGPL-3.0-only"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_COMPILE=1 \
    PATH="/opt/venv/bin:$PATH"

RUN groupadd --gid 1000 appuser \
    && useradd --uid 1000 --gid appuser --shell /bin/bash --create-home appuser

WORKDIR /app/middleware-qdrant-manager
COPY --from=middleware-builder --chown=appuser:appuser /opt/venv /opt/venv
COPY --chown=appuser:appuser middleware-qdrant-manager/qdrant_middleware/ qdrant_middleware/
COPY --chown=appuser:appuser middleware-qdrant-manager/frontend/ frontend/
# LICENSE, NOTICE and the licence texts NOTICE refers to travel with the code.
COPY --chown=appuser:appuser LICENSE NOTICE /app/
COPY --chown=appuser:appuser LICENSES/ /app/LICENSES/
RUN mkdir -p data && chown -R appuser:appuser /app/middleware-qdrant-manager

USER appuser
EXPOSE 6336
CMD ["python3", "-m", "uvicorn", "qdrant_middleware.main:app", "--host", "0.0.0.0", "--port", "6336"]

# MCP server dependency environment.
FROM python:3.11-slim-bookworm@sha256:b18992999dbe963a45a8a4da40ac2b1975be1a776d939d098c647482bcad5cba AS smart-qdrant-mcp-builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN python3 -m venv --copies /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
WORKDIR /app
COPY requirements/ requirements/
RUN pip install --upgrade \
        -c requirements/locks/mcp-py311-linux.txt \
        pip==26.2.0 \
        setuptools==84.2.1 \
        wheel==0.46.2 \
    && pip install --no-compile \
        -r requirements/mcp.txt \
        -c requirements/locks/mcp-py311-linux.txt

# MCP server runtime.
FROM python:3.11-slim-bookworm@sha256:b18992999dbe963a45a8a4da40ac2b1975be1a776d939d098c647482bcad5cba AS smart-qdrant-mcp

# Labels the image only; must equal smart_qdrant_mcp.__version__, which the
# server reports in serverInfo (smart-qdrant-mcp-server/tests/test_server.py).
ARG SMART_QDRANT_MCP_VERSION=1.0.1
LABEL org.opencontainers.image.title="hybrid-manager-smart-qdrant-mcp" \
      org.opencontainers.image.description="MCP server exposing the Qdrant middleware" \
      org.opencontainers.image.version="${SMART_QDRANT_MCP_VERSION}" \
      org.opencontainers.image.licenses="AGPL-3.0-only"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_COMPILE=1 \
    PATH="/opt/venv/bin:$PATH"

RUN groupadd --gid 1000 appuser \
    && useradd --uid 1000 --gid appuser --shell /bin/bash --create-home appuser

WORKDIR /app/smart-qdrant-mcp-server
COPY --from=smart-qdrant-mcp-builder --chown=appuser:appuser /opt/venv /opt/venv
COPY --chown=appuser:appuser smart-qdrant-mcp-server/smart_qdrant_mcp/ smart_qdrant_mcp/
# LICENSE, NOTICE and the licence texts NOTICE refers to travel with the code.
COPY --chown=appuser:appuser LICENSE NOTICE /app/
COPY --chown=appuser:appuser LICENSES/ /app/LICENSES/

USER appuser
EXPOSE 6337
CMD ["python3", "-m", "smart_qdrant_mcp"]

# Smart Embedder CPU dependency environment.
FROM python:3.11-slim-bookworm@sha256:b18992999dbe963a45a8a4da40ac2b1975be1a776d939d098c647482bcad5cba AS smart-embedder-cpu-builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN python3 -m venv --copies /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
WORKDIR /app
COPY requirements/ requirements/
RUN pip install --upgrade \
        -c requirements/locks/embedder-cpu-py311-linux.txt \
        pip==26.2.0 \
        setuptools==84.2.1 \
        wheel==0.46.2
RUN --mount=type=cache,target=/root/.cache/pip \
    PIP_NO_CACHE_DIR=0 pip install --no-compile \
        -r requirements/embedder-cpu.txt \
        -c requirements/locks/embedder-cpu-py311-linux.txt

# Smart Embedder CPU runtime.
FROM python:3.11-slim-bookworm@sha256:b18992999dbe963a45a8a4da40ac2b1975be1a776d939d098c647482bcad5cba AS smart-embedder-cpu

LABEL org.opencontainers.image.title="smart-embedder-cpu" \
      org.opencontainers.image.description="Smart Embedder FastAPI server (CPU)" \
      org.opencontainers.image.licenses="AGPL-3.0-only"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_COMPILE=1 \
    HF_HOME=/app/model_cache \
    HF_HUB_CACHE=/app/model_cache/hub \
    PATH="/opt/venv/bin:$PATH"

RUN groupadd --gid 1000 appuser \
    && useradd --uid 1000 --gid appuser --shell /bin/bash --create-home appuser

WORKDIR /app
COPY --from=smart-embedder-cpu-builder --chown=appuser:appuser /opt/venv /opt/venv
COPY --chown=appuser:appuser \
    server-embedder-reranker/smart-embedder.py \
    server-embedder-reranker/request_budget.py \
    server-embedder-reranker/vram_budget.py \
    server-embedder-reranker/basic_autotune.py \
    server-embedder-reranker/advanced_autotune.py \
    server-embedder-reranker/advanced_autotune_runner.py \
    server-embedder-reranker/advanced_autotune_worker.py \
    server-embedder-reranker/advanced_autotune_quality.py \
    server-embedder-reranker/advanced_milco_runtime.py \
    server-embedder-reranker/advanced_qwen_onnx_backend.py \
    server-embedder-reranker/basic_autotune_tokens.py \
    server-embedder-reranker/basic_onnx_backend.py \
    server-embedder-reranker/basic_autotune_runner.py \
    server-embedder-reranker/model_revisions.py \
    server-embedder-reranker/model_bootstrap.py \
    server-embedder-reranker/model_prepare.py \
    server-embedder-reranker/autotune_entrypoint.py \
    ./
# LICENSE, NOTICE and the licence texts NOTICE refers to travel with the code.
COPY --chown=appuser:appuser LICENSE NOTICE /app/
COPY --chown=appuser:appuser LICENSES/ /app/LICENSES/
RUN mkdir -p /app/model_cache && chown -R appuser:appuser /app/model_cache

USER appuser
EXPOSE 8000
CMD ["python3", "autotune_entrypoint.py"]

# Smart Embedder GPU dependency environment.
FROM nvidia/cuda:12.6.3-runtime-ubuntu22.04@sha256:63a18dd805367dacfb077aeced8384ab2fb569598ec5f5f5220c3f90a5c23650 AS smart-embedder-gpu-builder

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        python3 \
        python3-pip \
        python3-dev \
        python3-venv \
        ca-certificates \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv --copies /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
WORKDIR /app
COPY requirements/ requirements/
RUN pip install --upgrade \
        -c requirements/locks/embedder-gpu-py310-linux.txt \
        pip==26.2.0 \
        setuptools==84.2.1 \
        wheel==0.46.2
RUN --mount=type=cache,target=/root/.cache/pip \
    PIP_NO_CACHE_DIR=0 pip install --no-compile \
        -r requirements/embedder-gpu.txt \
        -c requirements/locks/embedder-gpu-py310-linux.txt

# Smart Embedder GPU runtime.
FROM nvidia/cuda:12.6.3-runtime-ubuntu22.04@sha256:63a18dd805367dacfb077aeced8384ab2fb569598ec5f5f5220c3f90a5c23650 AS smart-embedder-gpu

LABEL org.opencontainers.image.title="smart-embedder-gpu" \
      org.opencontainers.image.description="Smart Embedder FastAPI server (GPU/CUDA)" \
      org.opencontainers.image.licenses="AGPL-3.0-only"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_COMPILE=1 \
    HF_HOME=/app/model_cache \
    HF_HUB_CACHE=/app/model_cache/hub \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    CC=/usr/bin/gcc \
    CXX=/usr/bin/g++ \
    PATH="/opt/venv/bin:$PATH"

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        python3 \
        python3-dev \
        ca-certificates \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd --gid 1000 appuser \
    && useradd --uid 1000 --gid appuser --shell /bin/bash --create-home appuser

WORKDIR /app
COPY --from=smart-embedder-gpu-builder --chown=appuser:appuser /opt/venv /opt/venv
COPY --chown=appuser:appuser \
    server-embedder-reranker/smart-embedder.py \
    server-embedder-reranker/request_budget.py \
    server-embedder-reranker/vram_budget.py \
    server-embedder-reranker/basic_autotune.py \
    server-embedder-reranker/advanced_autotune.py \
    server-embedder-reranker/advanced_autotune_runner.py \
    server-embedder-reranker/advanced_autotune_worker.py \
    server-embedder-reranker/advanced_autotune_quality.py \
    server-embedder-reranker/advanced_milco_runtime.py \
    server-embedder-reranker/advanced_qwen_onnx_backend.py \
    server-embedder-reranker/basic_autotune_tokens.py \
    server-embedder-reranker/basic_onnx_backend.py \
    server-embedder-reranker/basic_autotune_runner.py \
    server-embedder-reranker/model_revisions.py \
    server-embedder-reranker/model_bootstrap.py \
    server-embedder-reranker/model_prepare.py \
    server-embedder-reranker/autotune_entrypoint.py \
    ./
# LICENSE, NOTICE and the licence texts NOTICE refers to travel with the code.
COPY --chown=appuser:appuser LICENSE NOTICE /app/
COPY --chown=appuser:appuser LICENSES/ /app/LICENSES/
RUN mkdir -p /app/model_cache && chown -R appuser:appuser /app/model_cache

USER appuser
EXPOSE 8000
CMD ["python3", "autotune_entrypoint.py"]
