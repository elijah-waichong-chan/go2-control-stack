from launch import LaunchDescription
from launch.actions import (  # type: ignore
    DeclareLaunchArgument,
    SetEnvironmentVariable,
)
from launch.substitutions import LaunchConfiguration  # type: ignore
from launch_ros.actions import Node  # type: ignore


def generate_launch_description():
    intent_topic = LaunchConfiguration("intent_topic")
    locomotion_cmd_topic = LaunchConfiguration("locomotion_cmd_topic")
    publish_hz = LaunchConfiguration("publish_hz")
    intent_hold_s = LaunchConfiguration("intent_hold_s")
    backward_x_vel = LaunchConfiguration("backward_x_vel")
    forward_x_vel = LaunchConfiguration("forward_x_vel")
    y_vel = LaunchConfiguration("y_vel")
    yaw_rate = LaunchConfiguration("yaw_rate")
    z_pos = LaunchConfiguration("z_pos")
    gait_hz = LaunchConfiguration("gait_hz")

    return LaunchDescription([
        DeclareLaunchArgument(
            "intent_topic",
            default_value="/direction_intent/forward_backward",
            description="Forward/backward intent label topic",
        ),
        DeclareLaunchArgument(
            "locomotion_cmd_topic",
            default_value="/locomotion_cmd",
            description="Locomotion command topic to publish",
        ),
        DeclareLaunchArgument(
            "publish_hz",
            default_value="50.0",
            description="Command publish frequency in Hz",
        ),
        DeclareLaunchArgument(
            "intent_hold_s",
            default_value="1.0",
            description="How long to hold a commanded x velocity after intent",
        ),
        DeclareLaunchArgument(
            "backward_x_vel",
            default_value="-0.5",
            description="x velocity for intent label 1",
        ),
        DeclareLaunchArgument(
            "forward_x_vel",
            default_value="0.5",
            description="x velocity for intent label 2",
        ),
        DeclareLaunchArgument(
            "y_vel",
            default_value="0.0",
            description="Fixed y velocity to publish",
        ),
        DeclareLaunchArgument(
            "yaw_rate",
            default_value="0.0",
            description="Fixed yaw rate to publish",
        ),
        DeclareLaunchArgument(
            "z_pos",
            default_value="0.27",
            description="Fixed base height to publish",
        ),
        DeclareLaunchArgument(
            "gait_hz",
            default_value="3.0",
            description="Fixed gait frequency to publish",
        ),
        SetEnvironmentVariable(
            "RCUTILS_CONSOLE_OUTPUT_FORMAT",
            "[{severity}] [{name}]: {message}",
        ),
        Node(
            package="direction_intent_estimator",
            executable="forward_backward_intent_cmd_publisher",
            name="forward_backward_intent_cmd_publisher",
            output="screen",
            parameters=[{
                "intent_topic": intent_topic,
                "locomotion_cmd_topic": locomotion_cmd_topic,
                "publish_hz": publish_hz,
                "intent_hold_s": intent_hold_s,
                "backward_x_vel": backward_x_vel,
                "forward_x_vel": forward_x_vel,
                "y_vel": y_vel,
                "yaw_rate": yaw_rate,
                "z_pos": z_pos,
                "gait_hz": gait_hz,
            }],
        ),
    ])
