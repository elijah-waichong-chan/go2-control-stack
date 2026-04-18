# HQ-PCOT (Human-Quadruped Prioceptive Co-Transport via Hierarchical Control)

Developed as part of the UC Berkeley Master of Engineering capstone project in Mechanical Engineering.

## Overview

This repository contains the HQ-PCOT ROS 2 Humble workspace for the Unitree Go2 platform. It is intended to run on the robot's Jetson Orin and is packaged around a Docker-based workflow.

The stack includes:

- a Streamlit telemetry dashboard
- reinforcement-learning based Unitree Go2 locomotion
- Unitree D1 arm feedback parsing and arm-control utilities
- intent estimation nodes backed by ONNX models
- robot description and vendor dependencies as submodules

## Repository Layout

| Path | Contents |
| --- | --- |
| `ros2_ws/src/core` | Core HQ-PCOT packages, launch files, and message definitions. |
| `ros2_ws/src/tools` | Tooling packages, including the telemetry dashboard. |
| `ros2_ws/src/descriptions` | Robot description and integration packages for the Go2 and D1 arm. |
| `ros2_ws/src/vendor` | Third-party and forked dependencies such as InEKF and Unitree packages. |
| `docs` | Project documentation, including status topic references. |

## Installation

This project is designed to run inside the provided Docker image on the Go2 onboard computer.

### 1. Clone the repository

Clone with submodules:

```bash
git clone --recurse-submodules https://github.com/elijah-waichong-chan/hq-pcot
cd hq-pcot
```

### 2. Build the Docker image

```bash
docker build -t hq-pcot .
```

### 3. Start the container

Start a container with the repository mounted into `/home/hq-pcot`:

```bash
docker run -d \
  --net=host \
  --mount type=bind,src="$(pwd)",dst=/home/hq-pcot \
  --name hq-pcot \
  hq-pcot sleep infinity
```

### 4. Open a shell in the container

```bash
docker exec -it hq-pcot bash
```

### 5. Build the ROS workspace

Inside the container:

```bash
cd /home/hq-pcot/ros2_ws
source /opt/ros/humble/setup.bash
colcon build --parallel-workers 4 --cmake-clean-cache \
  --packages-skip unitree_hg unitree_ros2_example unitree_api
source install/setup.bash
```

For iterative development on the Python packages, you can build with symlinks instead:

```bash
cd /home/hq-pcot/ros2_ws
source /opt/ros/humble/setup.bash
colcon build --symlink-install --parallel-workers 4 --cmake-clean-cache \
  --packages-skip unitree_hg unitree_ros2_example unitree_api
source install/setup.bash
```

To start the same container again later:

```bash
docker start hq-pcot
docker exec -it hq-pcot bash
```

If your Docker installation requires elevated privileges, prepend `sudo` to the commands above.

## Submodules

| Submodule Path | URL |
| --- | --- |
| `ros2_ws/src/core/go2_odometry` | `https://github.com/elijah-waichong-chan/go2_odometry.git` |
| `ros2_ws/src/vendor/unitree_ros2` | `https://github.com/unitreerobotics/unitree_ros2.git` |
| `ros2_ws/src/descriptions/unitree_description` | `https://github.com/inria-paris-robotics-lab/unitree_description.git` |
| `ros2_ws/src/vendor/inekf` | `https://github.com/elijah-waichong-chan/invariant-ekf.git` |

## Run

Launch the telemetry dashboard:

```bash
cd /home/hq-pcot/ros2_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch hq_pcot dashboard.launch.py
```

Then:

1. Open `http://localhost:8501`
2. Click `Start Control Stack` to launch the stack-managed nodes
3. Optionally click `Start Foxglove Bridge` to start the Foxglove bridge on port `8765`

## InEKF Debug

```bash
export PYTHONPATH=/usr/local/lib/python3.10/dist-packages:$PYTHONPATH
export LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/cmeel.prefix/lib:$LD_LIBRARY_PATH
```

## Onnxruntime Debug

```bash
export ONNXRUNTIME_ROOT=/opt/onnxruntime
export LD_LIBRARY_PATH="$ONNXRUNTIME_ROOT/lib:${LD_LIBRARY_PATH}"
```

## Workspace Packages

### Core Packages

| Package | Description |
| --- | --- |
| `arm_controller` | Arm feedback parsing, arm-control nodes, and D1 reference generation utilities. |
| `coordination_module` | Coordination logic built around HQ-PCOT custom messages. |
| `estimator_bridge` | Converts estimator outputs into the formats consumed elsewhere in the stack. |
| `go2_odometry` | Go2 odometry and InEKF-based state-estimation package. |
| `hq_pcot` | Central launch package for the HQ-PCOT stack. |
| `hq_pcot_msgs` | Custom ROS 2 message definitions shared across the workspace. |
| `intent_estimator` | Left-right, front-back, and up-down intent estimation nodes backed by ONNX models. |
| `locomotion_controller_cpp` | C++ locomotion controller nodes, including stand-up initialization and wireless command bridging. |
| `telemetry_dashboard` | Streamlit dashboard for monitoring and launching stack components. |

### Description Packages

| Package | Description |
| --- | --- |
| `d1_description` | D1 arm description assets and visualization support. |
| `go2_d1_integration` | Integration package that combines the Go2 and D1 descriptions. |
| `unitree_description` | External Unitree robot description dependency used by the estimator and state publisher path. |

### Vendor Packages

| Package | Description |
| --- | --- |
| `inekf` | Invariant EKF library dependency used by `go2_odometry`. |
| `unitree_arm` | Unitree arm interfaces and message definitions. |
| `unitree_ros2` | Unitree ROS 2 SDK sources and example packages that provide dependencies such as `unitree_go`. |

## Additional Documentation

- Status topic and dashboard status-code reference: [docs/status_codes.md](docs/status_codes.md)
