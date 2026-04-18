from setuptools import setup

package_name = 'arm_controller'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='unitree',
    maintainer_email='chanwaichong0352@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'arm_feedback_parser = arm_controller.arm_feedback_parser:main',
            'arm_controller = arm_controller.arm_controller:main',
            'd1_z_ref_node = arm_controller.d1_z_ref_node:main',
            'd1_drake_z_ref_node = arm_controller.d1_drake_z_ref_node:main',
            'd1_pink_arm_controller = arm_controller.d1_pink_arm_controller:main',
            'drake_arm_controller = arm_controller.drake_arm_controller:main',
        ],
    },
)
