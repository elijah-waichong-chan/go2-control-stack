from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    mujoco_share = get_package_share_directory('mujoco_robot')
    xml_path = f"{mujoco_share}/models/MJCF/go2/scene.xml"

    return LaunchDescription([
        Node(
            package='mujoco_robot',
            executable='mujoco_robot_node',
            name='mujoco_robot_node',
            output='screen',
            parameters=[{
                'xml_path': xml_path,
                'freeze_base': False,
                'enable_viewer': True,
                'render_hz': 30.0,
                'sim_hz': 500.0,
                'pub_hz': 500.0,
                'debug_print': True,
                'debug_publish': False,
                'foot_force_lpf_alpha': 0.0,
                'imu_gyro_noise_std': 0.005,
                'imu_acc_noise_std': 0.1,
                'imu_noise_seed': 1,
            }],
        )
    ])
