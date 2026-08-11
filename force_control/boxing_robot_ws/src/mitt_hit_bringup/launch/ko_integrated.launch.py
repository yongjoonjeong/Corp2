from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import EmitEvent, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch_ros.actions import Node


def _shutdown_if_node_exits(node: Node, label: str) -> RegisterEventHandler:
    """Treat every force-control process as critical.

    ros2 launch normally keeps sibling processes alive when one node exits.
    For KO that is unsafe because SessionBridge, force sampling, hit analysis and
    mitt positioning are one session-level unit.  Any unexpected exit therefore
    shuts down this entire launch so the top-level launcher can fail-fast too.
    """
    return RegisterEventHandler(
        OnProcessExit(
            target_action=node,
            on_exit=[EmitEvent(event=Shutdown(reason=f"critical node exited: {label}"))],
        )
    )


def generate_launch_description() -> LaunchDescription:
    config_dir = Path(get_package_share_directory("mitt_hit_bringup")) / "config"

    rt_force = Node(
        package="mitt_hit_system",
        executable="rt_force_diagnostic",
        name="rt_force_diagnostic",
        output="screen",
        parameters=[str(config_dir / "rt_force_diagnostic.params.yaml")],
    )
    hit_analyzer = Node(
        package="mitt_hit_system",
        executable="hit_analyzer",
        name="hit_analyzer",
        output="screen",
        parameters=[
            str(config_dir / "hit_analyzer.params.yaml"),
            str(config_dir / "compliance_session_base.params.yaml"),
            str(config_dir / "punch_rebound.params.yaml"),
        ],
    )
    mitt_positioner = Node(
        package="mitt_hit_system",
        executable="mitt_positioner",
        name="mitt_positioner",
        output="screen",
        parameters=[str(config_dir / "mitt_positioner.params.yaml")],
    )
    session_bridge = Node(
        package="boxing_integration",
        executable="session_bridge",
        name="boxing_session_bridge",
        output="screen",
    )

    critical_nodes = [
        (rt_force, "rt_force_diagnostic"),
        (hit_analyzer, "hit_analyzer"),
        (mitt_positioner, "mitt_positioner"),
        (session_bridge, "boxing_session_bridge"),
    ]

    actions = [node for node, _ in critical_nodes]
    actions.extend(_shutdown_if_node_exits(node, label) for node, label in critical_nodes)
    return LaunchDescription(actions)
