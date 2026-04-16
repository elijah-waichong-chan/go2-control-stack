import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():
    go2_odom_launch = os.path.join(
        get_package_share_directory("hq_pcot"),
        "launch",
        "go2_inekf_odometry.launch.py",
    )

    return LaunchDescription(
        [
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(go2_odom_launch),
            ),
            Node(
                package="estimator_bridge",
                executable="qdq_est_bridge",
                name="qdq_est_bridge",
                output="screen",
            ),
        ]
    )
