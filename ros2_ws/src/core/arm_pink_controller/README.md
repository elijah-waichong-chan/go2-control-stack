# arm_pink_controller

Pink-based `z_ref` controller for the D1 arm using:

- feedback topic: `/arm/servo_feedback`
- command input topic: `/arm/servo_command_input`
- repeated command topic: `/arm/servo_command`
- z-reference topic: `/d1_pink/z_ref`

Runtime Python dependencies are not vendored in this repo. Install them separately before running:

- `pin-pink`
- `pinocchio` (commonly via the `pin` Python package)
- `qpsolvers`
- one supported QP backend such as `quadprog` or `osqp`

This package no longer provides a GUI. Another node is expected to publish
`/d1_pink/z_velocity` and `/d1_pink/enabled`.

`/d1_pink/z_ref` is published by the controller as its internal z target state.

Example:

```bash
ros2 launch arm_pink_controller arm_pink_controller_mode_switch.launch.py
```
