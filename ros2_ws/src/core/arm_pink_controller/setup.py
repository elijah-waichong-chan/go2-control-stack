from setuptools import setup


package_name = "arm_pink_controller"


setup(
    name=package_name,
    version="0.0.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (
            "share/" + package_name + "/launch",
            [
                "launch/arm_pink_controller_mode_switch.launch.py",
            ],
        ),
        ("share/" + package_name, ["README.md"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="elijah",
    maintainer_email="elijah@example.com",
    description="Pink-based z-reference controller for the D1 arm.",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "d1_pink_z_ref_controller = arm_pink_controller.d1_pink_z_ref_controller:main",
            "d1_pink_mode_switch_controller = arm_pink_controller.d1_pink_mode_switch_controller:main",
        ],
    },
)
