# ADMIN MODE · Three-camera training view

실행:

```bash
./run_integrated.sh --admin-mode
```

훈련 준비와 훈련 화면에서는 현재 vision runtime의 `/api/vision/preview.jpg`를 우선 사용합니다.
이 프리뷰는 LEFT / FRONT / RIGHT annotated view를 하나의 composite로 제공합니다.

ADMIN MODE에서 함께 보존되는 진단 항목:
- LEFT / RIGHT fist BASE position
- LEFT / RIGHT fist BASE velocity
- Guard readiness
- READY / ACTIVE / IMPACT / COOLDOWN
- Target / mitt tracking 상태
- 관리자 전용 시스템 설정 및 연결 상태

USER MODE에서는 같은 비전 런타임이 동작하지만 전면 카메라와 종합 인식 상태만 표시합니다.
