FROM ubuntu:22.04

# Avoid tzdata interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    unzip \
    curl \
    gdb \
    libglew-dev \
    libopencv-dev \
    libeigen3-dev \
    libssl-dev \
    libboost-serialization-dev \
    python3-dev \
    python3-numpy \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libgtk2.0-dev \
    pkg-config \
    sudo \
    && rm -rf /var/lib/apt/lists/*

# Add a non-root user (good practice for vscode dev containers)
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd -s /bin/bash --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# Install Pangolin
WORKDIR /tmp
RUN git clone -b v0.6 https://github.com/stevenlovegrove/Pangolin.git && \
    cd Pangolin && \
    mkdir build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release && \
    make -j$(nproc) && \
    make install && \
    cd /tmp && rm -rf Pangolin

USER $USERNAME
WORKDIR /workspaces/ORB_SLAM3
