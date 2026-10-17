# The official Python build uses an older glibc than Ubuntu 24.04 and is compatible.
FROM python:3.11-slim-bookworm AS python
FROM nvidia/cuda:12.6.3-runtime-ubuntu24.04

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HUB_OFFLINE=1 \
    DATA_DIR=/app/data \
    PATH=/usr/local/bin:$PATH

RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates libbz2-1.0 libexpat1 libffi8 libgomp1 liblzma5 \
      libncursesw6 libreadline8t64 libsqlite3-0 libssl3t64 libtinfo6 zlib1g \
    && rm -rf /var/lib/apt/lists/*
COPY --from=python /usr/local /usr/local
RUN ldconfig && python --version

WORKDIR /app
COPY requirements.txt ./
RUN python -m pip install --no-cache-dir -r requirements.txt \
    && python -c "import cv2, fastapi, ssl, sqlite3; print('Runtime imports OK')"
COPY app ./app
COPY pipeline ./pipeline
COPY rules ./rules
COPY static ./static
COPY scripts ./scripts
RUN groupadd --gid 10001 detector \
    && useradd --uid 10001 --gid detector --no-create-home detector \
    && mkdir -p /app/data \
    && chown -R detector:detector /app/data

USER detector
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
