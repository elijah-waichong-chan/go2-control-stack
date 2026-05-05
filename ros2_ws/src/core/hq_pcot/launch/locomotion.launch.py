from launch import LaunchDescription
from launch.actions import SetEnvironmentVariable  # type: ignore
from launch_ros.actions import Node  # type: ignore


def generate_launch_description():
    return LaunchDescription([
        SetEnvironmentVariable(
            "RCUTILS_CONSOLE_OUTPUT_FORMAT",
            "[{severity}] [{name}]: {message}",
        ),
        Node(
            package="locomotion_controller_cpp",
            executable="stand_up_init",
            name="stand_up_init",
            output="screen",
        ),
        Node(
            package="locomotion_controller_cpp",
            executable="policy_controller",
            name="policy_controller",
            output="screen",
        ),
    ])
