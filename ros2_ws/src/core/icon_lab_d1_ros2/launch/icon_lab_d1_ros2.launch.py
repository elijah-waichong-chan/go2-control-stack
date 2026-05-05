from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import AnyLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    feedback_udp_address_arg = DeclareLaunchArgument(
        "feedback_udp_address",
        default_value="0.0.0.0",
    )
    command_udp_address_arg = DeclareLaunchArgument(
        "command_udp_address",
        default_value="192.168.123.100",
    )

    ros_feedback_topic = "/arm/servo_feedback"
    ros_command_input_topic = "/arm/servo_command_input"
    ros_command_topic = "/arm/servo_command"
    feedback_udp_address = LaunchConfiguration("feedback_udp_address")
    command_udp_address = LaunchConfiguration("command_udp_address")

    ros_feedback_publisher = Node(
        package="icon_lab_d1_ros2",
        executable="udp_feedback_ros_publisher",
        name="udp_feedback_ros_publisher",
        output="screen",
        parameters=[
            {
                "feedback_topic": ros_feedback_topic,
                "udp_address": feedback_udp_address,
            }
        ],
    )

    ros_command_repeater = Node(
        package="icon_lab_d1_ros2",
        executable="ros_command_topic_repeater",
        name="ros_command_topic_repeater",
        output="screen",
        parameters=[
            {
                "input_topic": ros_command_input_topic,
                "output_topic": ros_command_topic,
            }
        ],
    )

    ros_command_forwarder = Node(
        package="icon_lab_d1_ros2",
        executable="ros_command_udp_forwarder",
        name="ros_command_udp_forwarder",
        output="screen",
        parameters=[
            {
                "command_topic": ros_command_input_topic,
                "udp_address": command_udp_address,
            }
        ],
    )

    return LaunchDescription(
        [
            feedback_udp_address_arg,
            command_udp_address_arg,
            ros_feedback_publisher,
            ros_command_repeater,
            ros_command_forwarder,
        ]
    )
