FROM nvidia/cuda:12.8.1-runtime-ubuntu22.04

ENV NVIDIA_DRIVER_CAPABILITIES=graphics,utility,compute,display
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_NO_CACHE_DIR=1
ENV QT_X11_NO_MITSHM=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    git \
    ffmpeg \
    libegl1 \
    libxext6 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libx11-6 \
    libxrandr2 \
    libxi6 \
    libxinerama1 \
    libxcursor1 \
    libxxf86vm1 \
    libxfixes3 \
    libvulkan1 \
    libgomp1 \
    x11-apps \
    mesa-utils \
    ca-certificates \
    curl \
    unzip \
    vim \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3 /usr/bin/python

WORKDIR /workspace

COPY requirements.txt /workspace/requirements.txt
RUN python3 -m pip install --upgrade pip setuptools wheel && \
    python3 -m pip install -r /workspace/requirements.txt

COPY . /workspace
ENV PYTHONPATH=/workspace/src

CMD ["bash"]
