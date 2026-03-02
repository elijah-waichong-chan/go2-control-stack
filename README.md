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
