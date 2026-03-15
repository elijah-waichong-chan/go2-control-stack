from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, SetEnvironmentVariable # type: ignore
from launch.conditions import IfCondition # type: ignore
from launch.launch_description_sources import PythonLaunchDescriptionSource # type: ignore
from launch.substitutions import LaunchConfiguration # type: ignore
from launch_ros.actions import Node # type: ignore
from ament_index_python.packages import get_package_share_directory # type: ignore


def generate_launch_description():
    enable_estimator = LaunchConfiguration("enable_estimator")
    enable_arm_parser = LaunchConfiguration("enable_arm_parser")
    enable_wireless_cmd_bridge = LaunchConfiguration("enable_wireless_cmd_bridge")
    enable_forward_backward_estimator = LaunchConfiguration("enable_forward_backward_estimator")
    enable_left_right_estimator = LaunchConfiguration("enable_left_right_estimator")
    go2_odom_launch = (
        get_package_share_directory('go2_odometry')
        + '/launch/go2_inekf_odometry.launch.py'
    )
    return LaunchDescription([
        DeclareLaunchArgument(
            "enable_estimator",
            default_value="true",
            description="Launch estimator stack (inekf_odom.py + qdq_est_bridge)",
        ),
        DeclareLaunchArgument(
            "enable_arm_parser",
            default_value="true",
            description="Launch arm_feedback_parser node",
        ),
        DeclareLaunchArgument(
            "enable_wireless_cmd_bridge",
            default_value="true",
            description="Launch wireless_cmd_bridge node",
        ),
        DeclareLaunchArgument(
            "enable_forward_backward_estimator",
            default_value="true",
            description="Launch forward_backward_intent_estimator node",
        ),
        DeclareLaunchArgument(
            "enable_left_right_estimator",
            default_value="true",
            description="Launch left_right_intent_estimator node",
        ),
        SetEnvironmentVariable(
            'RCUTILS_CONSOLE_OUTPUT_FORMAT',
            '[{severity}] [{name}]: {message}'
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(go2_odom_launch),
            launch_arguments={"run_inekf": enable_estimator}.items(),
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
            condition=IfCondition(enable_estimator),
        ),
        Node(
            package='intent_estimator',
            executable='forward_backward_intent_estimator',
            name='forward_backward_intent_estimator',
            output='screen',
            condition=IfCondition(enable_forward_backward_estimator),
        ),
        Node(
            package='intent_estimator',
            executable='left_right_intent_estimator',
            name='left_right_intent_estimator',
            output='screen',
            condition=IfCondition(enable_left_right_estimator),
        ),
        Node(
            package='arm_controller',
            executable='arm_feedback_parser',
            name='arm_feedback_parser',
            output='screen',
            condition=IfCondition(enable_arm_parser),
        ),
        Node(
            package='locomotion_controller',
            executable='policy_controller',
            name='policy_controller',
            output='screen',
        ),
        Node(
            package='locomotion_controller',
            executable='wireless_cmd_bridge',
            name='wireless_cmd_bridge',
            output='screen',
            condition=IfCondition(enable_wireless_cmd_bridge),
            parameters=[{
                'push_event_hz': 10.0,
                'publish_hz': 50.0,
                'cmd_timeout_s': 0.5,
                'deadzone': 0.1,
                'scale_x': 1.0,
                'scale_y': -0.5,
                'scale_yaw': -1.0,
                'z_pos': 0.27,
            }],
        ),
    ])
