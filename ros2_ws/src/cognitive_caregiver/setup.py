from setuptools import find_packages, setup


package_name = "cognitive_caregiver"


setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="iqrashabbir1",
    maintainer_email="iqraaaaashabbir@gmail.com",
    description="ROS2 package skeleton for the cognitive caregiving robot digital twin.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "digital_twin_node = cognitive_caregiver.digital_twin_node:main",
            "risk_router_node = cognitive_caregiver.risk_router_node:main",
            "physiology_replay_node = cognitive_caregiver.physiology_replay_node:main",
        ],
    },
)
