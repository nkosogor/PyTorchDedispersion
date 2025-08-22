# syntax=docker/dockerfile:1.7

ARG PYTHON_VERSION=3.11

########################################
# CPU image
########################################
FROM python:${PYTHON_VERSION}-slim AS cpu
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends build-essential git && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY pyproject.toml README.md LICENSE mkdocs.yml .readthedocs.yaml ./
COPY pytorch_dedispersion ./pytorch_dedispersion
COPY tests ./tests
RUN python -m pip install --upgrade pip \
 && pip install "torch==2.*" --index-url https://download.pytorch.org/whl/cpu \
 && pip install .[dev,fil]
ENTRYPOINT ["pydedisp"]
CMD ["--help"]

########################################
# GPU image (requires NVIDIA Container Toolkit on host)
########################################
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime AS gpu
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY pyproject.toml README.md LICENSE mkdocs.yml .readthedocs.yaml ./
COPY pytorch_dedispersion ./pytorch_dedispersion
COPY tests ./tests
RUN python -m pip install --upgrade pip \
 && pip install .[dev,fil]
ENTRYPOINT ["pydedisp"]
CMD ["--help"]
