FROM ubuntu:22.04

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
    ros-humble-foxglove-bridge \
    ros-humble-pinocchio \
    ros-humble-rmw-cyclonedds-cpp \
    ros-humble-ros-base \
    ros-humble-rosidl-generator-dds-idl \
    ros-humble-rosbag2-storage-mcap \
    ros-humble-xacro

RUN git config --global --add safe.directory /home/go2-control-stack/ros2_ws/foxglove-sdk

ENV PIP_DEFAULT_TIMEOUT=120 \
    PIP_RETRIES=10

RUN python3 -m pip install --no-cache-dir --prefer-binary --ignore-installed streamlit onnxruntime==1.18.1 "numpy<2" casadi