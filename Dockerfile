# syntax=docker/dockerfile:1
# Lightweight CPU image for training, evaluation and inference.
#
#   docker build -t image-captioning .                      # CPU (default)
#   docker build --target demo -t image-captioning:demo .   # + Gradio web demo
#   docker build --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu126 \
#                -t image-captioning:gpu .                  # CUDA build, run with --gpus all
#
# Data and checkpoints are mounted, never baked into the image:
#   docker run --rm -v "$PWD/data:/app/data" -v "$PWD/checkpoints:/app/checkpoints" \
#              image-captioning python -m src.train --epochs 20

FROM python:3.12-slim AS base

ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MPLBACKEND=Agg

WORKDIR /app

# Dependencies first, so code changes do not invalidate this (large) layer.
COPY requirements.txt .
RUN pip install --extra-index-url "${TORCH_INDEX_URL}" -r requirements.txt

COPY pyproject.toml README.md app.py ./
COPY configs ./configs
COPY src ./src
RUN pip install --no-deps . \
    && useradd --create-home --uid 1000 app \
    && mkdir -p data checkpoints \
    && chown -R app:app /app

USER app
VOLUME ["/app/data", "/app/checkpoints"]

# Fail fast if the image is broken; show the available options by default.
RUN python -m src.train --help > /dev/null
CMD ["python", "-m", "src.train", "--help"]


FROM base AS demo
USER root
RUN pip install gradio==6.32.0
USER app
ENV GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860
EXPOSE 7860
CMD ["python", "app.py", "--checkpoint", "checkpoints/best_model.pth"]
