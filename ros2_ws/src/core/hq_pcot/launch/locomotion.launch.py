from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable  # type: ignore
from launch.conditions import IfCondition  # type: ignore
from launch.substitutions import LaunchConfiguration  # type: ignore
from launch_ros.actions import Node  # type: ignore


def generate_launch_description():
    enable_wireless_cmd_bridge = LaunchConfiguration("enable_wireless_cmd_bridge")

    return LaunchDescription([
        DeclareLaunchArgument(
            "enable_wireless_cmd_bridge",
            default_value="true",
            description="Launch wireless_cmd_bridge node",
        ),
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
        Node(
            package="locomotion_controller_cpp",
            executable="wireless_cmd_bridge",
            name="wireless_cmd_bridge",
            output="screen",
            condition=IfCondition(enable_wireless_cmd_bridge),
            parameters=[{
                "push_event_hz": 10.0,
                "publish_hz": 50.0,
                "cmd_timeout_s": 0.5,
                "deadzone": 0.1,
                "scale_x": 1.0,
                "scale_y": -0.5,
                "scale_yaw": -1.0,
                "z_pos": 0.27,
            }],
        ),
    ])
