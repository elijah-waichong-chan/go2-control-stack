from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription(
        [
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
                    "ros_feedback_topic": "/arm/servo_feedback",
                    "ros_command_input_topic": "/arm/servo_command_input",
                    "ros_command_topic": "/arm/servo_command",
                    "network_interface": "enx9c69d3284803",
                    "command_udp_address": "192.168.123.100",
                    "domain_id": "0",
                    "launch_gui": "false",
                }.items(),
            ),
            Node(
                package="arm_pink_controller",
                executable="d1_pink_mode_switch_controller",
                name="d1_pink_mode_switch_controller",
                output="screen",
            ),
            Node(
                package="arm_pink_controller",
                executable="d1_pink_z_ref_controller",
                name="d1_pink_z_ref_controller",
                output="screen",
                parameters=[
                    {
                        "defer_startup_until_enabled": True,
                    }
                ],
            ),
        ]
    )
