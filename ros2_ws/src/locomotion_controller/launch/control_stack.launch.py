from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, LogInfo, SetEnvironmentVariable # type: ignore
from launch.conditions import IfCondition, UnlessCondition # type: ignore
from launch.launch_description_sources import PythonLaunchDescriptionSource # type: ignore
from launch.substitutions import LaunchConfiguration # type: ignore
from launch_ros.actions import Node # type: ignore
from ament_index_python.packages import get_package_share_directory # type: ignore


def generate_launch_description():
    enable_inekf = LaunchConfiguration("enable_inekf")
    require_standing_init = LaunchConfiguration("require_standing_init")
    go2_odom_launch = (
        get_package_share_directory('go2_odometry')
        + '/launch/go2_inekf_odometry.launch.py'
    )
    return LaunchDescription([
        DeclareLaunchArgument(
            "enable_inekf",
            default_value="true",
            description="Launch inekf_odom.py estimator node",
        ),
        DeclareLaunchArgument(
            "require_standing_init",
            default_value="true",
            description="Run stand_up_init and require its ready status before locomotion starts",
        ),
        SetEnvironmentVariable(
            'RCUTILS_CONSOLE_OUTPUT_FORMAT',
            '[{severity}] [{name}]: {message}'
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(go2_odom_launch),
            launch_arguments={"run_inekf": enable_inekf}.items(),
        ),
        Node(
            package='locomotion_controller',
            executable='stand_up_init',
            name='stand_up_init',
            output='screen',
            condition=IfCondition(require_standing_init),
        ),
        LogInfo(
            msg='stand_up_init skipped',
            condition=UnlessCondition(require_standing_init),
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
            package='direction_intent_estimator',
            executable='forward_backward_intent_estimator',
            name='forward_backward_intent_estimator',
            output='screen',
            parameters=[{
                'lowstate_topic': '/lowstate',
                'qdq_topic': '/qdq_est',
                'output_topic': '/direction_intent/forward_backward',
                'status_topic': '/status/intent_estimator/forward_backward',
                'status_hz': 10.0,
            }],
        ),
        Node(
            package='direction_intent_estimator',
            executable='left_right_intent_estimator',
            name='left_right_intent_estimator',
            output='screen',
            parameters=[{
                'arm_angles_topic': '/arm_angles',
                'output_topic': '/direction_intent/left_right',
                'status_topic': '/status/intent_estimator/left_right',
                'status_hz': 10.0,
            }],
        ),
        Node(
            package='arm_controller',
            executable='arm_feedback_parser',
            name='arm_feedback_parser',
            output='screen',
        ),
        Node(
            package='locomotion_controller',
            executable='policy_controller',
            name='policy_controller',
            output='screen',
            parameters=[{
                'require_standing_init': require_standing_init,
                'control_hz': 50.0,
            }],
        ),
        Node(
            package='locomotion_controller',
            executable='wireless_cmd_bridge',
            name='wireless_cmd_bridge',
            output='screen',
            parameters=[{
                'wireless_topic': '/wirelesscontroller',
                'locomotion_cmd_topic': '/locomotion_cmd',
                'push_event_topic': '/data/push_event',
                'push_event_hz': 10.0,
                'publish_hz': 50.0,
                'cmd_timeout_s': 0.5,
                'deadzone': 0.05,
                'scale_x': 0.5,
                'scale_y': -0.5,
                'scale_yaw': -1.0,
                'z_pos': 0.27,
                'gait_hz': 3.0,
            }],
        ),
    ])
