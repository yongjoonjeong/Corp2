#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

import numpy as np

from robot_calibration.common import load_yaml
from robot_calibration.world_transform import RobotWorldTransformer


def values_in_mm(values: list[float], unit: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    return array * 1000.0 if unit == "m" else array


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transform camera points/vectors into Doosan BASE coordinates"
    )
    parser.add_argument("--calibration", default="calibration/results/robot_world.yaml")
    subparsers = parser.add_subparsers(dest="command", required=True)

    point = subparsers.add_parser("point")
    point.add_argument("--camera", required=True)
    point.add_argument("--xyz", nargs=3, type=float, required=True)
    point.add_argument("--unit", choices=["mm", "m"], default="mm")

    vector = subparsers.add_parser("vector")
    vector.add_argument("--camera", required=True)
    vector.add_argument("--xyz", nargs=3, type=float, required=True)

    mitt = subparsers.add_parser("mitt")
    mitt.add_argument("--camera", required=True)
    mitt.add_argument("--point", nargs=3, type=float, required=True)
    mitt.add_argument("--direction", nargs=3, type=float, required=True)
    mitt.add_argument("--unit", choices=["mm", "m"], default="mm")
    mitt.add_argument("--mitt-config", default="config/mitt.yaml")
    mitt.add_argument("--stand-off-mm", type=float, default=0.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    transformer = RobotWorldTransformer(args.calibration)
    if args.camera not in transformer.camera_names():
        raise RuntimeError(
            f"Camera '{args.camera}' not calibrated. Available: {transformer.camera_names()}"
        )
    if args.command == "point":
        point_base = transformer.point_camera_to_base(
            args.camera, values_in_mm(args.xyz, args.unit)
        )
        print(json.dumps({"point_base_mm": point_base.tolist()}, indent=2))
        return 0
    if args.command == "vector":
        vector_base = transformer.vector_camera_to_base(args.camera, args.xyz)
        print(json.dumps({"vector_base": vector_base.tolist()}, indent=2))
        return 0
    mitt_config = load_yaml(args.mitt_config)
    pose = transformer.make_mitt_pose(
        camera_name=args.camera,
        impact_point_camera_mm=values_in_mm(args.point, args.unit),
        punch_direction_camera=args.direction,
        surface_normal_axis=str(mitt_config.get("surface_normal_axis", "+Z")),
        local_up_axis=str(mitt_config.get("local_up_axis", "+Y")),
        base_up=mitt_config.get("base_up_axis", [0, 0, 1]),
        stand_off_mm=args.stand_off_mm,
    )
    print(
        json.dumps(
            {
                "point_base_mm": pose.position_base_mm.tolist(),
                "punch_direction_base": pose.punch_direction_base.tolist(),
                "mitt_surface_normal_base": pose.desired_surface_normal_base.tolist(),
                "R_base_tcp": pose.rotation_base_tcp.tolist(),
                "doosan_target_posx_mm_deg": pose.doosan_posx_mm_deg.tolist(),
                "warning": (
                    "ABC is one valid ZYZ representation. Validate the configured mitt "
                    "normal-axis sign at low speed before robot motion."
                ),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
