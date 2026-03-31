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
    python3 \
    python3-pip \
    python3-dev \
    python3-numpy \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libgtk2.0-dev \
    pkg-config \
    sudo \
    && rm -rf /var/lib/apt/lists/*

# Install ROS2 Humble
RUN apt-get update && apt-get install -y software-properties-common curl \
    && add-apt-repository universe \
    && curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null \
    && apt-get update \
    && apt-get install -y ros-humble-ros-base ros-humble-cv-bridge ros-humble-sensor-msgs ros-humble-image-transport ros-humble-rmw-cyclonedds-cpp python3-colcon-common-extensions \
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
    sed -i '1i#include <limits>' include/pangolin/gl/colour.h && \
    mkdir build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_PANGOLIN_PYTHON=OFF && \
    make -j$(nproc) && \
    make install && \
    cd /tmp && rm -rf Pangolin

RUN pip install "rerun-sdk==0.16.1" "numpy<2"

ENV RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

USER $USERNAME
WORKDIR /workspaces/ORB_SLAM3

# Source ROS2 automatically in the terminal
RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc \
    && echo "if [ -f /workspaces/ORB_SLAM3/Examples_ROS2/install/setup.bash ]; then source /workspaces/ORB_SLAM3/Examples_ROS2/install/setup.bash; fi" >> ~/.bashrc
