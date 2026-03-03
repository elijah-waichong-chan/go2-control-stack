import os

from setuptools import find_packages, setup

package_name = 'locomotion_controller'

data_files = [
    ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
    ('share/' + package_name, ['package.xml']),
]

for root, _, files in os.walk('config'):
    if not files:
        continue
    install_dir = os.path.join('share', package_name, root)
    file_paths = [os.path.join(root, f) for f in files]
    data_files.append((install_dir, file_paths))

for root, _, files in os.walk('launch'):
    launch_files = [f for f in files if f.endswith('.py')]
    if not launch_files:
        continue
    install_dir = os.path.join('share', package_name, root)
    file_paths = [os.path.join(root, f) for f in launch_files]
    data_files.append((install_dir, file_paths))

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=data_files,
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='elijah',
    maintainer_email='chanwaichong0352@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'stand_up_init = locomotion_controller.standup_init:main',
            'policy_controller = locomotion_controller.policy_controller:main',
            'qdq_plotter = locomotion_controller.qdq_plotter:main',
        ],
    },
)
