#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import yaml


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from sandbag_vision.calibration import load_robot_world  # noqa: E402


def resolve(config_path: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (config_path.parent / path).resolve()


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate models, calibration, and projected training ROIs")
    parser.add_argument("--config", default=str(PROJECT / "config" / "runtime.yaml"))
    args = parser.parse_args()
    config_path = Path(args.config).expanduser().resolve()
    with config_path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    calibration_path = resolve(config_path, config["calibration"]["robot_world"])
    calibration = load_robot_world(calibration_path)
    missing = []
    for section in ("target", "pose"):
        model = resolve(config_path, config[section]["model"])
        print(f"{section:>12} model: {model} ({'OK' if model.is_file() else 'MISSING'})")
        if not model.is_file():
            missing.append(model)
    print(f" calibration: {calibration_path} (OK)")
    workspace = config["target"]["workspace_base_mm"]
    for camera in ("left", "front", "right"):
        roi = calibration.project_workspace(camera, workspace)
        print(f"{camera:>12} workspace ROI: {roi}")
    if missing:
        print(f"[ERROR] missing files: {missing}")
        return 1
    print("[OK] runtime configuration is internally consistent")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
