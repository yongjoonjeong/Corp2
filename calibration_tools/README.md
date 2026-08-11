# 3카메라 → 로봇 BASE 캘리브레이션 도구

ZIP에서 캘리브레이션 전용 코드만 가져왔습니다. 현재 비전 런타임과 기존
`calibration/intrinsics`, `calibration/results/robot_world.yaml`은 덮어쓰지 않았습니다.

모든 명령은 프로젝트 루트에서 실행합니다.

```bash
python3 calibration_tools/00_check_cameras.py --list-c270
python3 calibration_tools/00_check_cameras.py --assign-c270
python3 calibration_tools/00_check_cameras.py

python3 calibration_tools/01_intrinsic_calibration.py --camera front
python3 calibration_tools/01_intrinsic_calibration.py --camera left
python3 calibration_tools/01_intrinsic_calibration.py --camera right

python3 calibration_tools/02_collect_external_samples.py --camera front
python3 calibration_tools/03_solve_external_calibration.py
python3 calibration_tools/04_validate_robot_world.py --camera front
python3 calibration_tools/05_robot_world_transform.py point --camera front --xyz 0 0 1000
python3 calibration_tools/06_enable_manual_teaching.py
```

`01`~`04`는 `config/board.yaml`, `config/cameras.yaml`과 현재
`config/camera_roles.yaml`을 사용합니다. `02`, `04`, `06`은 Doosan ROS 2 환경이
필요합니다. 실제 로봇을 움직이거나 직접교시 모드를 바꾸기 전에는 작업 셀을 비우고
저속에서 검증하세요.
