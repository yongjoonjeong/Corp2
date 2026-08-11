from setuptools import find_packages, setup


setup(
    name="boxing_integration",
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/boxing_integration"]),
        ("share/boxing_integration", ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="rokey",
    maintainer_email="rokey@todo.local",
    description="KO UI to mitt session integration bridge.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "session_bridge = boxing_integration.session_bridge:main",
        ]
    },
)
