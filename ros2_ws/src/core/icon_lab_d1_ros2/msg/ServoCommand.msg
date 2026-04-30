# Per-joint command mode for the 7 D1 arm servos.
uint8 MODE_IGNORE=0
uint8 MODE_POSITION_INTERVAL=1
uint8 MODE_POSITION_VELOCITY=2
uint8 MODE_DAMPING=3

std_msgs/Header header
uint8[7] mode
float32[7] angle_deg
float32[7] velocity_deg_s
float32[7] damping_power_mw
uint16 interval_ms
