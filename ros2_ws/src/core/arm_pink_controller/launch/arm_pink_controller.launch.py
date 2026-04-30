from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.actions import IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription(
        [
            DeclareLaunchArgument("feedback_topic", default_value="/arm/servo_feedback"),
            DeclareLaunchArgument(
                "command_input_topic", default_value="/arm/servo_command_input"
            ),
            DeclareLaunchArgument("command_topic", default_value="/arm/servo_command"),
            DeclareLaunchArgument("z_ref_topic", default_value="/d1_pink/z_ref"),
            DeclareLaunchArgument("z_velocity_topic", default_value="/d1_pink/z_velocity"),
            DeclareLaunchArgument("current_z_topic", default_value="/d1_pink/current_z"),
            DeclareLaunchArgument("enabled_topic", default_value="/d1_pink/enabled"),
            DeclareLaunchArgument("startup_complete_topic", default_value="/d1_pink/startup_complete"),
            DeclareLaunchArgument("network_interface", default_value="enx9c69d3284803"),
            DeclareLaunchArgument("arm_ip", default_value="192.168.123.100"),
            DeclareLaunchArgument("domain_id", default_value="0"),
            DeclareLaunchArgument("control_rate_hz", default_value="20.0"),
            DeclareLaunchArgument("z_ref_min_m", default_value="0.20"),
            DeclareLaunchArgument("z_ref_max_m", default_value="0.50"),
            DeclareLaunchArgument("max_abs_z_velocity_mps", default_value="0.20"),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    PathJoinSubstitution(
                        [
                            FindPackageShare("icon_lab_d1_ros2"),
                            "launch",
                            "icon_lab_d1_ros2.launch.py",
                        ]
                    )
                ),
                launch_arguments={
                    "ros_feedback_topic": LaunchConfiguration("feedback_topic"),
                    "ros_command_input_topic": LaunchConfiguration("command_input_topic"),
                    "ros_command_topic": LaunchConfiguration("command_topic"),
                    "network_interface": LaunchConfiguration("network_interface"),
                    "command_udp_address": LaunchConfiguration("arm_ip"),
                    "domain_id": LaunchConfiguration("domain_id"),
                    "launch_gui": "false",
                }.items(),
            ),
            Node(
                package="arm_pink_controller",
                executable="d1_pink_z_ref_controller",
                name="d1_pink_z_ref_controller",
                output="screen",
                parameters=[
                    {
                        "feedback_topic": LaunchConfiguration("feedback_topic"),
                        "command_topic": LaunchConfiguration("command_input_topic"),
                        "z_ref_topic": LaunchConfiguration("z_ref_topic"),
                        "z_velocity_topic": LaunchConfiguration("z_velocity_topic"),
                        "current_z_topic": LaunchConfiguration("current_z_topic"),
                        "enabled_topic": LaunchConfiguration("enabled_topic"),
                        "startup_complete_topic": LaunchConfiguration("startup_complete_topic"),
                        "control_rate_hz": LaunchConfiguration("control_rate_hz"),
                        "z_ref_min_m": LaunchConfiguration("z_ref_min_m"),
                        "z_ref_max_m": LaunchConfiguration("z_ref_max_m"),
                        "max_abs_z_velocity_mps": LaunchConfiguration("max_abs_z_velocity_mps"),
                    }
                ],
            ),
        ]
    )
