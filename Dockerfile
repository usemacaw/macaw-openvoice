# Macaw OpenVoice — CPU-only multi-stage build
# Usage:
#   docker build -t Macaw .
#   docker run -p 8000:8000 -v Macaw-models:/root/.Macaw/models Macaw

# ── Stage 1: build ──────────────────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY pyproject.toml .
COPY src/ src/

RUN uv pip install --system --no-cache ".[server,grpc,itn]"

# ── Stage 2: runtime ────────────────────────────────────────────
FROM python:3.12-slim

WORKDIR /app

# Copy installed packages and entry point
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /app/src /app/src

# Models volume mount point
VOLUME /root/.macaw/models

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

ENTRYPOINT ["Macaw", "serve", "--host", "0.0.0.0"]
