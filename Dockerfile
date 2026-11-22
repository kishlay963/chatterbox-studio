FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime
LABEL org.opencontainers.image.source="https://github.com/kishlay963/chatterbox-studio" org.opencontainers.image.description="Chatterbox voice-clone TTS + faster-whisper word-level transcription API"

ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1 \
    HF_HOME=/models/hf HF_HUB_DISABLE_TELEMETRY=1 \
    VOICES_DIR=/data/voices JOBS_DIR=/data/jobs BUILTIN_VOICES=/app/voices_builtin \
    DEFAULT_VOICE= WHISPER_MODEL=large-v3 QC_WHISPER_MODEL=medium.en

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt bake_models.py ./
RUN pip install -r requirements.txt
RUN python bake_models.py chatterbox
RUN python bake_models.py medium.en
RUN python bake_models.py large-v3

COPY voices /app/voices_builtin
COPY app /app/app
RUN mkdir -p /data && JOBS_DIR=/tmp/jobs python -c "import app.server; print('api imports ok')"

VOLUME ["/data"]
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s CMD curl -fs localhost:8000/health || exit 1
CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8000"]
