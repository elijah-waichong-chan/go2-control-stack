from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, SetEnvironmentVariable # type: ignore
from launch.launch_description_sources import PythonLaunchDescriptionSource # type: ignore
from launch_ros.actions import Node # type: ignore
from ament_index_python.packages import get_package_share_directory # type: ignore


def generate_launch_description():
    go2_odom_launch = (
        get_package_share_directory('go2_odometry')
        + '/launch/go2_inekf_odometry.launch.py'
    )
    return LaunchDescription([
        SetEnvironmentVariable(
            'RCUTILS_CONSOLE_OUTPUT_FORMAT',
            '[{severity}] [{name}]: {message}'
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(go2_odom_launch),
        ),
        Node(
            package='locomotion_controller',
            executable='stand_up_init',
            name='stand_up_init',
            output='screen',
        ),
        Node(
            package='estimator_bridge',
            executable='qdq_est_bridge',
            name='qdq_est_bridge',
            output='screen',
            parameters=[{
                'odom_topic': '/odometry/filtered',
                'joint_states_topic': '/joint_states',
                'qdq_topic': '/qdq_est',
            }],
        ),
        Node(
            package='locomotion_controller',
            executable='policy_controller',
            name='policy_controller',
            output='screen',
            parameters=[{
                'require_standing_init': True,
                'control_hz': 50.0,
            }],
        ),
    ])
