FROM python:3.11-slim-bookworm

ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    OMP_NUM_THREADS=2 \
    MKL_NUM_THREADS=2 \
    OPENBLAS_NUM_THREADS=2 \
    NUMEXPR_NUM_THREADS=2

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
        swig \
        xauth \
        xvfb \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN python -m pip install --upgrade pip \
    && python -m pip install --index-url "${TORCH_INDEX_URL}" torch \
    && python -m pip install -r requirements.txt

RUN python -c "from metadrive.pull_asset import pull_asset; pull_asset(update=False)"

COPY . .

RUN mkdir -p checkpoints logs model_backup models models_backup outputs videos
RUN chmod +x /app/docker-entrypoint.sh

EXPOSE 4000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:4000/api/health', timeout=3)" || exit 1

CMD ["/app/docker-entrypoint.sh"]
