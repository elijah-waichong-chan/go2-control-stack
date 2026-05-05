from launch import LaunchDescription
from launch.substitutions import Command, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    robot_desc = ParameterValue(
        Command(
            [
                "xacro ",
                PathJoinSubstitution(
                    [
                        FindPackageShare("go2_d1_integration"),
                        "urdf",
                        "go2_d1_combined.xacro",
                    ]
                ),
            ]
        ),
        value_type=str,
    )

    return LaunchDescription(
        [
            Node(
                package="robot_state_publisher",
                executable="robot_state_publisher",
                name="robot_state_publisher",
                output="screen",
                parameters=[{"robot_description": robot_desc}],
                ros_arguments=[
                    "--log-level", "robot_state_publisher:=warn",
                    "--log-level", "kdl_parser:=error",
                ],
            ),
            Node(
                package="go2_state_converter",
                executable="state_converter_node",
                name="state_converter_node",
                output="screen",
            ),
        ]
    )
