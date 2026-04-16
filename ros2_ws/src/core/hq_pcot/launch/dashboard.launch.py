from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package="telemetry_dashboard",
            executable="telemetry_dashboard",
            name="telemetry_dashboard",
            output="screen",
        ),
    ])
