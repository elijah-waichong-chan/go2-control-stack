from setuptools import find_packages, setup


package_name = "coordination_module"


setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        (
            "share/ament_index/resource_index/packages",
            ["resource/" + package_name],
        ),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="elijah",
    maintainer_email="chanwaichong0352@gmail.com",
    description="Coordination nodes for intent-driven locomotion.",
    license="TODO: License declaration",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "intent_command_coordinator = coordination_module.intent_command_coordinator:main",
        ],
    },
)
