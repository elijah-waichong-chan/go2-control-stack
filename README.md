# ROS 2 Control Stack for Unitree Go2 Quadruped Robot

Developed as part of the **UC Berkeley Master of Engineering (MEng)** capstone project in Mechanical Engineering.

## Updates

03/01/2026
- Added locomotion RL controller policy
- Added MuJoCo simulation option to validate the controller
- Added extended kalman filter state estimator

## Run Instructions

1. Build and source the workspace:
   ```bash
   cd ros2_ws
   source /opt/ros/humble/setup.bash
   colcon build --parallel-workers 2
   source install/setup.bash
   ```

2. Launch the dashboard:
   ```bash
   ros2 launch locomotion_controller dashboard.launch.py
   ```
   - At this point, a web-based dashboard should open on your browser

3. If you want to run in simulation, click **Start MuJoCo** in the dashboard. If not, skip this step.

4. Click **Start Control Stack** in the dashboard. The stack will:
   - first run the stand-up motion sequence using a PD controller,
   - then initialize the state estimator,
   - then start the locomotion RL controller, which publishes commands to `/lowcmd`.

## Repository Packages

| Package | Primary Language(s) | Description |
| --- | --- | --- |
| `locomotion_controller` | Python | Stand-up initialization and RL policy controller nodes, plus launch files for dashboard/simulation/control stack. |
| `estimator_bridge` | Python | Converts estimator-related topics (odometry + joint states) into the `/qdq_est` format used by this stack. |
| `telemetry_dashboard` | Python | Streamlit-based dashboard for status/plots and start-stop controls for MuJoCo and control stack. |
| `xbox_controller_bridge` | Python | Maps joystick input (`/joy`) to locomotion command messages (`/locomotion_cmd`). |
| `mujoco_robot` | C++ | MuJoCo simulation node for Go2 that publishes robot state and consumes low-level motor commands. |
| `go2_odometry` | C++ and Python | Odometry and state-estimation package (including InEKF integration and state conversion utilities). |
| `go2_msgs` | ROS 2 interface definitions (`.msg`) | Custom message definitions used across the control stack (for example `LocomotionCmd` and `QDq`). |

## Dependencies (Git Submodules)

This repository depends on the following Git submodules:

| Submodule Path | Upstream URL |
| --- | --- |
| `ros2_ws/src/go2_odometry` | `https://github.com/elijah-waichong-chan/go2_odometry.git` |
| `ros2_ws/src/unitree_ros2` | `https://github.com/unitreerobotics/unitree_ros2` |
| `ros2_ws/src/unitree_description` | `https://github.com/inria-paris-robotics-lab/unitree_description.git` |
| `ros2_ws/src/inekf` | `https://github.com/inria-paris-robotics-lab/invariant-ekf` |

Note: the `go2_odometry` submodule in this repository is a fork from the INRIA Paris Robotics Lab ecosystem: `https://github.com/inria-paris-robotics-lab`.

To clone with submodules:

```bash
git clone --recurse-submodules <repo_url>
```

If already cloned:

```bash
git submodule update --init --recursive
```
