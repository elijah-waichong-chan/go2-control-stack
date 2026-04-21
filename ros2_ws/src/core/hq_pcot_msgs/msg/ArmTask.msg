std_msgs/Header header

# Task selected by the motion coordinator for the mounted arm.
uint8 TASK_UNSPECIFIED=0
uint8 TASK_FRONT_BACK_TRACK=1
uint8 TASK_UP_DOWN_TRACK=2
uint8 TASK_FRONT_BACK_PRESET=3
uint8 TASK_UP_DOWN_PRESET=4
uint8 task

# Signed vertical motion command used for TASK_UP_DOWN_TRACK.
# Values are -1, 0, or +1.
int8 z_motion_direction

# Optional preset joint-space command in degrees for the 6 arm joints.
# Used for TASK_FRONT_BACK_PRESET and TASK_UP_DOWN_PRESET.
float64[6] preset_joint_configuration_deg

# Per-joint execution mask for preset_joint_configuration_deg.
# Values are 0 or 1 for the 6 arm joints.
uint8[6] preset_joint_update_mask
