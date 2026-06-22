# VieNeu-TTS → OpenAI-compatible TTS shim.
# Default build = local mode: v3 Turbo on CPU via ONNX Runtime (torch-free, no GPU).
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/data/hf \
    VIENEU_MODE=local \
    VIENEU_EMOTION=natural

WORKDIR /app

# libsndfile1 backs soundfile (incl. MP3 encode); ffmpeg/curl are handy for ops + healthcheck.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libsndfile1 ffmpeg curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app.py .

# HF model cache lives on a volume so models download once and survive redeploys.
RUN mkdir -p /data/hf
VOLUME ["/data/hf"]

EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=5 \
    CMD curl -fsS http://localhost:8080/health || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
