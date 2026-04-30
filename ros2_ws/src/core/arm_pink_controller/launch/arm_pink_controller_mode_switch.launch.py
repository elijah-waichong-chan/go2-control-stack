from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription(
        [
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
