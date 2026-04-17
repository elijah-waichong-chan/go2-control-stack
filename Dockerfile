FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Etc/UTC
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ >/etc/timezone

EXPOSE 8501 8765

RUN apt update && apt install -y \
    software-properties-common \
    tzdata && \
    add-apt-repository universe && \
    apt update && apt install -y \
    curl \
    git \
    locales \
    python3 \
    python3-empy \
    python3-pip \
    software-properties-common

RUN locale-gen en_US en_US.UTF-8 && \
    update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
ENV LANG=en_US.UTF-8

RUN set -eux; \
    ROS_APT_SOURCE_VERSION="$(curl -fsSL https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest | grep -F "tag_name" | awk -F\" '{print $4}')"; \
    UBUNTU_CODENAME="$(. /etc/os-release && echo "${UBUNTU_CODENAME:-${VERSION_CODENAME}}")"; \
    curl -fL -o /tmp/ros2-apt-source.deb "https://github.com/ros-infrastructure/ros-apt-source/releases/download/${ROS_APT_SOURCE_VERSION}/ros2-apt-source_${ROS_APT_SOURCE_VERSION}.${UBUNTU_CODENAME}_all.deb"; \
    dpkg -i /tmp/ros2-apt-source.deb

RUN apt update && apt upgrade -y && apt install -y \
    libasio-dev \
    libboost-test-dev \
    libwebsocketpp-dev \
    nlohmann-json3-dev \
    ros-dev-tools \
    ros-jazzy-foxglove-bridge \
    ros-jazzy-pinocchio \
    ros-jazzy-rmw-cyclonedds-cpp \
    ros-jazzy-ros-base \
    ros-jazzy-rosidl-generator-dds-idl \
    ros-jazzy-rosbag2-storage-mcap \
    ros-jazzy-xacro

RUN git config --global --add safe.directory /home/go2-control-stack/ros2_ws/foxglove-sdk

ENV PIP_DEFAULT_TIMEOUT=120 \
    PIP_RETRIES=10

RUN python3 -m pip install --break-system-packages --no-cache-dir --prefer-binary --ignore-installed streamlit onnxruntime==1.18.1 "numpy<2" casadi

ENV ONNXRUNTIME_VERSION=1.18.1
ENV ONNXRUNTIME_ROOT=/root/onnxruntime
ENV LD_LIBRARY_PATH=${ONNXRUNTIME_ROOT}/lib

RUN mkdir -p ${ONNXRUNTIME_ROOT} && \
    curl -L https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-linux-aarch64-${ONNXRUNTIME_VERSION}.tgz \
    | tar -xz -C ${ONNXRUNTIME_ROOT} --strip-components=1

ENV DRAKE_URL=https://drake-packages.csail.mit.edu/drake/nightly/drake-latest-noble-aarch64.tar.gz \
    DRAKE_INSTALL=/opt/drake

RUN mkdir -p ${DRAKE_INSTALL} && \
    curl -fL ${DRAKE_URL} -o /tmp/drake.tar.gz && \
    tar -xzf /tmp/drake.tar.gz -C ${DRAKE_INSTALL} --strip-components=1 && \
    ${DRAKE_INSTALL}/share/drake/setup/install_prereqs -y

ENV PYTHONPATH=/opt/drake/lib/python3.12/site-packages \
    CMAKE_PREFIX_PATH=/opt/drake \
    LD_LIBRARY_PATH=/opt/drake/lib:/opt/onnxruntime/lib
