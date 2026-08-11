# `/mitt/hit_result` Phase 2 연동

Phase 2는 사용자가 제공한 미완성 `boxing_robot_ws`의 `HitResult.msg` 계약을 기준으로 구현했습니다.

```text
builtin_interfaces/Time stamp
uint32 hit_id
bool valid_hit
string invalid_reason
string hit_direction
float64 hit_x_mm
float64 hit_y_mm
float64 center_error_mm
float64 peak_force_n
float64 peak_normal_force_n
float64 impulse_ns
float64 contact_duration_ms
float64 accuracy_score
float64 power_score
float64 total_score
bool force_warning
bool safety_stop
```

`run_ui_bridge.sh`는 기본적으로 `${BOXING_ROBOT_WS:-$HOME/boxing_robot_ws}/install/setup.bash`를 찾습니다.
존재하면 source 후 `/mitt/hit_result`를 구독하고, 없으면 force 기능만 비활성 상태로 남습니다.

다른 위치라면:

```bash
BOXING_ROBOT_WS=/원하는/경로/boxing_robot_ws ./run_integrated.sh --admin-mode
```

현재 미완성 analyzer 코드에서 `power_score`와 `safety_stop`은 고정값이므로 AI 보고서의 발전 판정에는 사용하지 않습니다.
