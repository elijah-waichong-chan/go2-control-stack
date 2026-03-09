import os

from setuptools import find_packages, setup

package_name = 'direction_intent_estimator'

data_files = [
    (
        'share/ament_index/resource_index/packages',
        ['resource/' + package_name],
    ),
    ('share/' + package_name, ['package.xml']),
]

for root, _, files in os.walk('config'):
    if not files:
        continue
    install_dir = os.path.join('share', package_name, root)
    file_paths = [os.path.join(root, f) for f in files]
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
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'forward_backward_intent_estimator = direction_intent_estimator.forward_backward_intent_estimator:main',
            'left_right_intent_estimator = direction_intent_estimator.left_right_intent_estimator:main',
        ],
    },
)
