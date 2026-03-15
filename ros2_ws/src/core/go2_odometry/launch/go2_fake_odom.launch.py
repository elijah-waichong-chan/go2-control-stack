from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch_ros.actions import Node
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution, TextSubstitution, LaunchConfiguration
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    minimal_state_publisher_launch_file = PathJoinSubstitution(
        [FindPackageShare("go2_odometry"), "launch", "go2_state_publisher.launch.py"]
    )

    fake_odom_base_height_arg = DeclareLaunchArgument(
        "base_height",
        default_value=TextSubstitution(text="0.30"),
        description="[IF FAKE ODOM] Height of the robot base.",
    )

    return LaunchDescription(
        [
            fake_odom_base_height_arg,
            Node(
                package="go2_odometry",
                executable="fake_odom.py",
                name="fake_odom",
                parameters=[{"base_height": LaunchConfiguration("base_height")}],
            ),
            IncludeLaunchDescription(PythonLaunchDescriptionSource([minimal_state_publisher_launch_file])),
        ]
    )
