# Force Interface Preparation

Reference workspace: uploaded `boxing_robot_ws`

This document records the force-data contract that later KO integration should consume. Phase 1 does **not** modify or run the force-control workspace.

## Primary per-hit interface

The current robot-force workspace already publishes a useful per-hit message on:

```text
/mitt/hit_result
```

Message: `boxing_interfaces/msg/HitResult`

Fields currently available:

```text
stamp
hit_id
valid_hit
invalid_reason
hit_direction
hit_x_mm
hit_y_mm
center_error_mm
peak_force_n
peak_normal_force_n
impulse_ns
contact_duration_ms
accuracy_score
power_score
total_score
force_warning
safety_stop
```

For the next KO integration phase, `/mitt/hit_result` should be the primary report/database interface instead of inventing a second force-result format.

## Raw RT interface

The analyzer consumes:

```text
/mitt/rt_sample
```

Message: `boxing_interfaces/msg/RtMittSample`

```text
stamp
frame_id
corrected_wrench[6]
tcp_pose_mm_deg[6]
tcp_velocity_mm_deg_s[6]
robot_state
singularity
```

The analyzer configuration expects:

```text
frame_id = mitt_tool_corrected
```

If a future report requires the full force-vector history (Fx/Fy/Fz), it should be derived from the contact interval of `corrected_wrench`, not guessed from `HitResult`.

## Recommended KO mapping

### USER MODE — only when real data exists

- 최대/대표 타격 힘: `peak_force_n` or `peak_normal_force_n` after the team confirms which value is the product metric
- 타격 위치/방향: `hit_direction`, `center_error_mm`
- 접촉 특성 when useful: `impulse_ns`, `contact_duration_ms`

If the force source is unavailable, USER MODE should omit the force section rather than showing zero as a real measurement.

### ADMIN MODE

Expose raw diagnostic values from `/mitt/hit_result`, plus RT connection/state when available.

## Important current limitations in the uploaded workspace

The current analyzer publishes:

```text
power_score = 0.0
safety_stop = false
```

in `hit_analyzer_node.py`, so those fields must not yet be presented as implemented power scoring or certified robot safety-stop results.

The analyzer parameters also currently keep compliance fail-closed/disabled while several operational limits remain unset or unverified. Therefore the UI/report integration may prepare to receive force results, but must not imply that force compliance or robot safety control is complete.

## Phase 2 target

Later integration can add a KO-side adapter that subscribes to `/mitt/hit_result`, maps hits to the current training session/punch record, persists supported fields to SQLite, and makes the data available to ADMIN diagnostics and the final coaching report.
