# syntax=docker/dockerfile:1
FROM python:3.12.14-slim-bookworm
LABEL org.opencontainers.image.source="https://github.com/L0ckR/seif-pii"
WORKDIR /app
# TOKENIZERS_PARALLELISM controls threading, not a credential.
# hadolint ignore=DL3064
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
    HF_HOME=/tmp/huggingface NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    SEIF_NER_HOST=0.0.0.0 SEIF_NER_PORT=8770 SEIF_NER_WORKERS=1 \
    SEIF_NER_BACKEND=rubert SEIF_RUBERT_MODEL_PATH=/models/rubert \
    SEIF_RUBERT_DECODER=word SEIF_RUBERT_PROFILE=native
# Pinning libgomp1 to one Debian patch release would prevent security updates
# and make rebuilds fail when that release leaves the package repository.
# hadolint ignore=DL3008
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --uid 10001 --create-home seif
COPY deploy/ner/requirements-rubert-tensorrt.txt /tmp/runtime-constraints.txt
COPY deploy/rubert/requirements-runtime.txt /tmp/runtime-requirements.txt
# CUDA 12.8 user-space libraries come from the pinned Torch dependency wheels;
# the host supplies the NVIDIA driver through NVIDIA Container Toolkit.
# Select Torch from its official index without mixing package indexes.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-deps --index-url https://download.pytorch.org/whl/cu128 torch==2.8.0+cu128 \
    && pip install -c /tmp/runtime-constraints.txt -r /tmp/runtime-requirements.txt \
    && pip check
COPY scripts/ner_service.py ./scripts/ner_service.py
COPY seif ./seif
COPY third_party ./third_party
USER 10001:10001
EXPOSE 8770
HEALTHCHECK --interval=10s --timeout=4s --start-period=180s --retries=6 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8770/health', timeout=3).close()"]
CMD ["python", "-m", "scripts.ner_service"]
