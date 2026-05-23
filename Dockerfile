# syntax=docker/dockerfile:1.7
#
# Non-streaming voice-design HTTP server (Qwen3-TTS-12Hz-1.7B-VoiceDesign).
# Drop-in replacement for the Chutes /speak endpoint. Runs on RTX 4090.
#
# Build:
#   docker build -t docker.io/<ns>/voice-design-non-streaming:latest .
# Run (on the rented box):
#   docker run -d --gpus all --restart=unless-stopped \
#     -p 8112:8112 \
#     -e QWEN3_VD_API_KEY=<key> \
#     docker.io/<ns>/voice-design-non-streaming:latest

FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/cache/hf \
    TRANSFORMERS_CACHE=/cache/hf/transformers \
    QWEN3_VD_HOST=0.0.0.0 \
    QWEN3_VD_PORT=8112

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-dev \
        git curl ca-certificates \
        libsndfile1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Same torch/CUDA stack as fast-tts-streaming (both servers can share a
# GPU and we want the same kernels in VRAM).
RUN pip install --upgrade pip \
    && pip install --index-url https://download.pytorch.org/whl/cu128 \
        "torch>=2.5.1" torchaudio

RUN pip install "faster-qwen3-tts>=0.2.6" "qwen-tts>=0.1.1"

WORKDIR /app
COPY pyproject.toml ./
COPY qwen3_voice_design/ ./qwen3_voice_design/
COPY samples/ ./samples/

RUN pip install --no-deps -e .

VOLUME /cache/hf

EXPOSE 8112

# 120s start-period: model load + warmup synthesis takes ~30-60s on cold cache.
HEALTHCHECK --interval=15s --timeout=5s --start-period=120s --retries=3 \
    CMD curl -fsS http://127.0.0.1:${QWEN3_VD_PORT:-8112}/healthz || exit 1

CMD ["qwen3-voice-design"]
