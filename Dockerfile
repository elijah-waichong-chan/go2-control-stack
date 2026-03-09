FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ >/etc/timezone

EXPOSE 8501 8765

RUN apt update
RUN apt install -y software-properties-common tzdata
RUN add-apt-repository universe
RUN apt update

RUN apt install -y python3.11 pip locales software-properties-common curl

RUN locale-gen en_US en_US.UTF-8
RUN update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
ENV LANG=en_US.UTF-8

RUN set -eux; \
    ROS_APT_SOURCE_VERSION="$(curl -fsSL https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest | grep -F "tag_name" | awk -F\" '{print $4}')"; \
    UBUNTU_CODENAME="$(. /etc/os-release && echo "${UBUNTU_CODENAME:-${VERSION_CODENAME}}")"; \
    curl -fL -o /tmp/ros2-apt-source.deb "https://github.com/ros-infrastructure/ros-apt-source/releases/download/${ROS_APT_SOURCE_VERSION}/ros2-apt-source_${ROS_APT_SOURCE_VERSION}.${UBUNTU_CODENAME}_all.deb"; \
    dpkg -i /tmp/ros2-apt-source.deb

RUN apt update
RUN apt upgrade -y
RUN apt install -y ros-humble-ros-base ros-dev-tools 
RUN apt install -y ros-humble-rosidl-generator-dds-idl libboost-test-dev
RUN apt install -y ros-humble-pinocchio
RUN apt install -y ros-humble-foxglove-bridge

RUN pip install streamlit eigenpy onnxruntime
RUN pip install "numpy<2"
ENV CMAKE_PREFIX_PATH=/usr/local/lib/python3.10/dist-packages/cmeel.prefix
