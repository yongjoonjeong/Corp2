# KO UI용 3카메라 3D 비전 브리지

이 폴더에는 KO UI와 ROS 비전 토픽을 연결하는 `ko_ui_bridge.py`가 있습니다.
기존 단일 웹캠 MediaPipe MVP 파일은 참고용으로만 남아 있으며 통합 실행에서는 사용하지 않습니다.

실제 비전 런타임은 통합본 최상위의 `vision/` 폴더입니다.

```text
FRONT RealSense + LEFT/RIGHT C270
→ YOLO11n-Pose 동일 복서 매칭
→ 3D 관절·손목 궤적·접촉점
→ ROS score/status/preview/evidence 토픽
→ ko_ui_bridge.py
→ KO UI
```

개별 실행 대신 통합본 루트에서 실행하세요.

```bash
./setup_integrated.sh
./run_integrated.sh
```

로봇 없이 UI와 3D 비전만 확인하려면:

```bash
KO_WITHOUT_ROBOT=1 ./run_integrated.sh
```

`run_ros_mvp.sh`는 하위 호환을 위한 얇은 래퍼이며 받은 인자를 수정하지 않고
`vision/run_ros_3d_mvp.sh`로 전달합니다.
