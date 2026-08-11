from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from pathlib import Path


def generate_launch_description() -> LaunchDescription:
    config = (
        Path(get_package_share_directory("mitt_hit_bringup"))
        / "config"
        / "rt_force_diagnostic.params.yaml"
    )
    return LaunchDescription(
        [
            Node(
                package="mitt_hit_system",
                executable="rt_force_diagnostic",
                name="rt_force_diagnostic",
                output="screen",
                parameters=[str(config)],
            )
        ]
    )
