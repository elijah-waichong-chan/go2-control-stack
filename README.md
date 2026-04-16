# ROS 2 Control Stack for Unitree Go2

Developed as part of the UC Berkeley Master of Engineering capstone project in Mechanical Engineering.

## Overview

This repository contains a ROS 2 Humble control stack for the Unitree Go2, including:

- a Streamlit telemetry dashboard
- stand-up initialization
- an InEKF-based state estimator path
- an RL locomotion controller
- arm feedback parsing
- direction intent estimation
- Unitree SDK and description dependencies as submodules

The dashboard can start and stop:

- the main control stack
- the Foxglove bridge

## Clone

Clone with submodules:

```bash
git clone --recurse-submodules https://github.com/elijah-waichong-chan/go2-control-stack
```

## Submodules

| Submodule Path | URL |
| --- | --- |
| `ros2_ws/src/core/go2_odometry` | `https://github.com/elijah-waichong-chan/go2_odometry.git` |
| `ros2_ws/src/vendor/unitree_ros2` | `https://github.com/unitreerobotics/unitree_ros2` |
| `ros2_ws/src/descriptions/unitree_description` | `https://github.com/inria-paris-robotics-lab/unitree_description.git` |
| `ros2_ws/src/vendor/inekf` | `https://github.com/elijah-waichong-chan/invariant-ekf.git` |

## Docker

Build the image:

```bash
sudo docker build -t go2-ros2-control .
```

Start and run the container:

```bash
sudo docker run -d \
  --net=host \
  --mount type=bind,src="$(pwd)",dst=/home/go2-control-stack \
  --name go2-ros2-control \
  go2-ros2-control sleep infinity
```

Open a shell in the container:

```bash
sudo docker exec -it <container_id> bash
```
To start an existing container later:

```bash
sudo docker start go2-ros2-control
```

Useful Docker commands:

```bash
sudo docker image ls
sudo docker ps
```

## Build

Inside the container or on a ROS 2 Humble host:

```bash
cd /home/go2-control-stack/ros2_ws
source /opt/ros/humble/setup.bash
colcon build --parallel-workers 4 --cmake-clean-cache
source install/setup.bash
```

## Run

Launch the dashboard:

```bash
cd /home/go2-control-stack/ros2_ws
source /opt/ros/humble/setup.bash
source /home/go2-control-stack/ros2_ws/install/setup.bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

source install/setup.bash
ros2 launch locomotion_controller dashboard.launch.py
```

Then:

1. Open the dashboard on `http://localhost:8501`
2. Click `Start Control Stack` to launch the control stack
3. Optionally click `Start Foxglove Bridge` to launch Foxglove socket node

ROS Bag collection

```bash
mkdir -p /home/unitree/go2-control-stack/bag_data
ros2 bag record -s mcap -o "/home/unitree/go2-control-stack/bag_data/go2_data_$(date +%Y%m%d_%H%M%S)" /data/push_event /lowstate /arm_angles
```

`/arm_angles` now includes both `angle_deg` and `current`, so this recording already captures arm motor current.

## InEKF Debug

```bash
export PYTHONPATH=/usr/local/lib/python3.10/dist-packages:$PYTHONPATH
export LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/cmeel.prefix/lib:$LD_LIBRARY_PATH

``

## Onnxruntime Debug
export ONNXRUNTIME_ROOT=/root/onnxruntime
export LD_LIBRARY_PATH="$ONNXRUNTIME_ROOT/lib:${LD_LIBRARY_PATH}"


## Workspace Packages

### Project Packages

| Package | Description |
| --- | --- |
| `arm_controller` | Parses `/arm_Feedback`, publishes `/arm_angles`, and exposes arm parser status. |
| `intent_estimator` | Direction intent estimation nodes using 1-D CNN models. |
| `estimator_bridge` | Converts estimator outputs into the `/qdq_est` format used by this stack. |
| `go2_msgs` | Custom ROS 2 message definitions used across the stack. |
| `locomotion_controller` | Stand-up initialization, RL policy control, safety stop handling, and launch files. |
| `telemetry_dashboard` | Streamlit dashboard for module status, topic monitoring, and launch controls. |

`/direction_intent/forward_backward` uses label semantics `0=idle`, `1=forward`, `2=backward`. The current topic publisher applies a 2-consecutive confirmation filter, so the published label only changes after the same new class is predicted twice in a row. The dashboard intent-estimator module card currently monitors only `/status/intent_estimator/forward_backward`, and that status topic now includes loop timing stats in the same format as the locomotion RL controller.

### Third-Party Packages

| Package | Description |
| --- | --- |
| `go2_odometry` | Forked odometry and state-estimation package, including InEKF runtime nodes. |
| `inekf` | Invariant EKF library dependency used by the estimator path. |
| `unitree_arm` | Unitree arm interfaces and related dependencies. |
| `unitree_description` | Robot description package used by the odometry and state publisher path. |
| `unitree_ros2` | Unitree ROS 2 SDK and example packages. |
