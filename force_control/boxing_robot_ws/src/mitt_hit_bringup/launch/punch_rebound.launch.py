from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from pathlib import Path


def generate_launch_description() -> LaunchDescription:
    config_directory = Path(get_package_share_directory("mitt_hit_bringup")) / "config"
    return LaunchDescription(
        [
            Node(
                package="mitt_hit_system",
                executable="hit_analyzer",
                name="hit_analyzer",
                output="screen",
                parameters=[
                    str(config_directory / "hit_analyzer.params.yaml"),
                    str(config_directory / "compliance_session_base.params.yaml"),
                    str(config_directory / "punch_rebound.params.yaml"),
                ],
            )
        ]
    )
