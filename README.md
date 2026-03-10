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
| `ros2_ws/src/go2_odometry` | `https://github.com/elijah-waichong-chan/go2_odometry.git` |
| `ros2_ws/src/unitree_ros2` | `https://github.com/unitreerobotics/unitree_ros2` |
| `ros2_ws/src/unitree_description` | `https://github.com/inria-paris-robotics-lab/unitree_description.git` |
| `ros2_ws/src/inekf` | `https://github.com/elijah-waichong-chan/invariant-ekf.git` |

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
colcon build --parallel-workers 4
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
ros2 bag record -s mcap -o "direction_estimator_bag_$(date +%Y%m%d_%H%M%S)" /data/push_event /lowstate /arm_angles
```

## Workspace Packages

### Project Packages

| Package | Description |
| --- | --- |
| `arm_controller` | Parses `/arm_Feedback`, publishes `/arm_angles`, and exposes arm parser status. |
| `direction_intent_estimator` | Direction intent estimation nodes using 1-D CNN models. |
| `estimator_bridge` | Converts estimator outputs into the `/qdq_est` format used by this stack. |
| `go2_msgs` | Custom ROS 2 message definitions used across the stack. |
| `locomotion_controller` | Stand-up initialization, RL policy control, safety stop handling, and launch files. |
| `telemetry_dashboard` | Streamlit dashboard for module status, topic monitoring, and launch controls. |

### Third-Party Packages

| Package | Description |
| --- | --- |
| `go2_odometry` | Forked odometry and state-estimation package, including InEKF runtime nodes. |
| `inekf` | Invariant EKF library dependency used by the estimator path. |
| `unitree_arm` | Unitree arm interfaces and related dependencies. |
| `unitree_description` | Robot description package used by the odometry and state publisher path. |
| `unitree_ros2` | Unitree ROS 2 SDK and example packages. |
