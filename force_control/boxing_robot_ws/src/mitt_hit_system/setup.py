from glob import glob
from setuptools import find_packages, setup


package_name = "mitt_hit_system"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools", "PyYAML"],
    zip_safe=True,
    maintainer="rokey",
    maintainer_email="rokey@todo.local",
    description="Force-based mitt hit analysis and M0609 safety control.",
    license="Apache-2.0",
    extras_require={"test": ["pytest"]},
    entry_points={
        "console_scripts": [
            "compliance_preflight = mitt_hit_system.compliance_preflight_node:main",
            "hit_analyzer = mitt_hit_system.hit_analyzer_node:main",
            "mitt_positioner = mitt_hit_system.mitt_positioner_node:main",
            "rt_force_diagnostic = mitt_hit_system.rt_force_diagnostic_node:main",
        ]
    },
)
