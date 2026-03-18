from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os
import xacro


def generate_launch_description():
    combined_xacro_path = os.path.join(
        get_package_share_directory("go2_d1_integration"),
        "urdf",
        "go2_d1_combined.xacro",
    )
    robot_desc = xacro.process_file(
        combined_xacro_path,
        mappings={"include_fingers": "false"},
    ).toxml()

    return LaunchDescription(
        [
            Node(
                package="robot_state_publisher",
                executable="robot_state_publisher",
                name="robot_state_publisher",
                output="screen",
                parameters=[{"robot_description": robot_desc}],
                arguments=[combined_xacro_path],
                ros_arguments=[
                    "--log-level", "robot_state_publisher:=warn",
                    "--log-level", "kdl_parser:=error",
                ],
            ),
            Node(
                package="go2_odometry",
                executable="state_converter_node",
                name="state_converter_node",
                parameters=[],
                output="screen",
            ),
        ]
    )
