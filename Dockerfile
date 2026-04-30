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
    python3-pip && \
    rm -rf /var/lib/apt/lists/*

RUN locale-gen en_US en_US.UTF-8 && \
    update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
ENV LANG=en_US.UTF-8

RUN set -eux; \
    ROS_APT_SOURCE_VERSION="$(curl -fsSL https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest | grep -F "tag_name" | awk -F\" '{print $4}')"; \
    UBUNTU_CODENAME="$(. /etc/os-release && echo "${UBUNTU_CODENAME:-${VERSION_CODENAME}}")"; \
    curl -fL -o /tmp/ros2-apt-source.deb "https://github.com/ros-infrastructure/ros-apt-source/releases/download/${ROS_APT_SOURCE_VERSION}/ros2-apt-source_${ROS_APT_SOURCE_VERSION}.${UBUNTU_CODENAME}_all.deb"; \
    dpkg -i /tmp/ros2-apt-source.deb

RUN apt update && apt install -y \
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
    ros-humble-xacro && \
    rm -rf /var/lib/apt/lists/*

ENV PIP_DEFAULT_TIMEOUT=120 \
    PIP_RETRIES=10

RUN python3 -m pip install --no-cache-dir --prefer-binary --ignore-installed streamlit onnxruntime==1.18.1 "numpy<2" casadi

ENV ONNXRUNTIME_VERSION=1.18.1
ENV ONNXRUNTIME_ROOT=/opt/onnxruntime
ENV LD_LIBRARY_PATH=${ONNXRUNTIME_ROOT}/lib:${LD_LIBRARY_PATH}

RUN mkdir -p ${ONNXRUNTIME_ROOT} && \
    curl -L https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-linux-aarch64-${ONNXRUNTIME_VERSION}.tgz \
    | tar -xz -C ${ONNXRUNTIME_ROOT} --strip-components=1

ENV UNITREE_SDK2_REPO=https://github.com/unitreerobotics/unitree_sdk2.git \
    UNITREE_SDK2_REF=main \
    UNITREE_SDK2_ROOT=/opt/unitree_robotics

RUN apt update && apt install -y \
    build-essential \
    cmake \
    libboost-all-dev \
    libeigen3-dev \
    libfmt-dev \
    libspdlog-dev \
    libyaml-cpp-dev && \
    rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 --branch ${UNITREE_SDK2_REF} ${UNITREE_SDK2_REPO} /tmp/unitree_sdk2 && \
    cmake -S /tmp/unitree_sdk2 -B /tmp/unitree_sdk2/build -DCMAKE_INSTALL_PREFIX=${UNITREE_SDK2_ROOT} && \
    cmake --build /tmp/unitree_sdk2/build -j"$(nproc)" && \
    cmake --install /tmp/unitree_sdk2/build && \
    rm -rf /tmp/unitree_sdk2

ENV LD_LIBRARY_PATH=${UNITREE_SDK2_ROOT}/lib:${LD_LIBRARY_PATH}
