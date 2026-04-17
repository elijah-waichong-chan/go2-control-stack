std_msgs/Header header

# Target interpretation
uint8 COMMAND_TYPE_UNSPECIFIED=0
uint8 COMMAND_TYPE_END_EFFECTOR_POSE=1
uint8 COMMAND_TYPE_JOINT_CONFIGURATION=2
uint8 command_type

# End-effector position in meters, expressed in header.frame_id.
# Used when command_type == COMMAND_TYPE_END_EFFECTOR_POSE.
float64[3] position_m

# End-effector orientation as quaternion [x, y, z, w].
# Used when command_type == COMMAND_TYPE_END_EFFECTOR_POSE.
float64[4] orientation_xyzw

# Joint-space command in degrees for the 6 arm joints, excluding the gripper.
# Used when command_type == COMMAND_TYPE_JOINT_CONFIGURATION.
float64[6] joint_configuration_deg
