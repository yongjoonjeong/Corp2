# 현재 U자 위빙 모션 설정

이 문서는 V3 자동 시작 통합본의 실제 로봇 흐름을 설명합니다.

```text
노드 시작
→ HOME [-180, 0, 90, -180, -90, 0]
→ 위빙 준비 [-180, 0, 90, -270, 90, 0]
→ U자 위빙 자동 시작
→ Wake Word 인식
→ Soft Stop
→ 위빙 준비 자세 복귀
```

별도 접근 자세는 사용하지 않습니다. U자 경로는 위빙 준비 TCP를 오른쪽 상단으로 사용하며 X=0~-170 mm, Z=0~-68 mm 범위에서 동작합니다.

```python
AUTO_START_WEAVING_ON_STARTUP = True
U_HALF_WIDTH_MM = 85.0
U_DEPTH_MM = 68.0
WEAVE_VEL = [450.0, 15.0]
WEAVE_ACC = [900.0, 30.0]
SMOOTH_ROUND_TRIPS_PER_MOVESX = 4
SMOOTH_POINTS_PER_ROUND_TRIP = 20
```

한 번의 `movesx()`에 4왕복, 총 80개 점을 전달해 상단마다 명령이 끊기는 현상을 줄입니다.
