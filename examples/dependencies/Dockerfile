#################### BUILD STAGE ####################
FROM nvidia/cuda:12.6.3-devel-ubuntu22.04 AS builder

ARG PYTHON_VERSION=3.12

RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    wget \
    curl \
    bzip2

RUN curl -sL https://micro.mamba.pm/api/micromamba/linux-64/1.1.0 \
    | tar -xvj -C /usr/local bin/micromamba

ENV MAMBA_EXE=/usr/local/bin/micromamba \
    MAMBA_ROOT_PREFIX=/opt/micromamba \
    CONDA_PREFIX=/opt/micromamba \
    PATH=/opt/micromamba/bin:$PATH

RUN micromamba create -y -n base && \
    micromamba shell init --shell=bash --prefix="$MAMBA_ROOT_PREFIX"
    
RUN micromamba install python=${PYTHON_VERSION} pip -c conda-forge -y && \
    python -m pip install --upgrade pip

# Why not do all at once? 
# Because vLLM runs faster with FlashInfer.
# To install FlashInfer, we need to install torch first.
RUN python -m pip install --no-cache-dir torch==2.6.0
RUN python -m pip install --no-cache-dir flashinfer-python -i https://flashinfer.ai/whl/cu126/torch2.6/
RUN python -m pip install --no-cache-dir vllm openai httpx

#################### RUNTIME STAGE ####################
FROM nvidia/cuda:12.6.3-devel-ubuntu22.04

# Add your micromamba setup on top of CoreWeave's RDMA base
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    bzip2 \
    && rm -rf /var/lib/apt/lists/*

RUN curl -sL https://micro.mamba.pm/api/micromamba/linux-64/1.1.0 \
    | tar -xvj -C /usr/local bin/micromamba

ENV MAMBA_EXE=/usr/local/bin/micromamba \
    MAMBA_ROOT_PREFIX=/opt/micromamba \
    CONDA_PREFIX=/opt/micromamba \
    PATH=/opt/micromamba/bin:$PATH

COPY --from=builder /opt/micromamba /opt/micromamba

ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

ENTRYPOINT ["python", "-m", "vllm.entrypoints.openai.api_server"]
