# KO Final Integration · 2026-08-11

## Final ownership

- `ui/`, `sandbag_vision/`: KO 최신 UI, 3-camera vision, user DB, reports, Wakeword/STT
- `robot_control/robot_weaving_node.py`: M0609 HOME + automatic BASE-XZ weaving + safe action handoff
- `robot_control/ui_robot_bridge.py`: UI command queue ↔ ROS command/status adapter
- `force_control/boxing_robot_ws`: mitt positioning, latest RT wrench processing, compliance/rebound, hit analysis
- `boxing_integration/session_bridge.py`: final robot/training state owner and the only owner of `/mitt/start_test` / `/mitt/stop_test`

## Authoritative robot poses

- HOME: `[0, 0, 90, 0, 90, 0]`
- Weaving ready: `[-180, 0, 90, 90, 90, 0]`
- Punching ready (`reference_joint_deg`): `[-90, 60, 30, -90, -90, 0]`
- Weaving plane: BASE `XZ`
- Weaving extent: `X=-85..+85 mm`, `Z=0..-68 mm`, relative `Y=0`

Weaving ready and punching ready are intentionally different postures.

## Final runtime flow

1. `run_final.sh` starts the integrated runtime.
2. M0609 moves to the weaving-ready posture and continuously performs the XZ U-shaped weave.
3. The user starts training from the UI. The UI first performs front-camera body/alignment checking; weaving continues during this stage.
4. Only after camera alignment is complete does the UI send `training_start`.
5. The weaving node soft-stops, returns to weaving-ready, then publishes `/robot_boxing/action_ready`.
6. SessionBridge asks MittPositioner to move from the weaving side into the separate punching-ready/user-specific mitt pose.
7. **1st calibration — reach correction:** the user holds the jab-side/non-dominant fist extended and still. The mitt advances slowly along Tool +Z, the mitt-face normal, until force contact, saves the actual reach correction, then waits for force release.
8. **2nd calibration — five-hit force centering:** StartHitTest enables the verified Force/Compliance session. The UI displays and speaks `영점 조정 중입니다. 잠시 기다려주세요.` while the wrench baseline is being adjusted. When ready, it speaks `다시 펀치하세요.`
9. Each valid calibration hit uses the force-derived contact offset/direction (`hit_x_mm`, `hit_y_mm`) to post-correct the mitt center. Intentional mitt motion is guarded so the Compliance watchdog does not mistake it for an uncontrolled motion. After the move the settled pose is recaptured and wrench zero is recalibrated before the next calibration punch is admitted.
10. After five valid hits, the final personalized mitt correction is stored in the user DB and SessionBridge publishes `TRAINING_READY`.
11. The UI countdown starts only after `TRAINING_READY`. Current production scope is **jab and straight**. Hook/uppercut extension is intentionally deferred until jab/straight physical validation is complete.
12. During training, Vision fist state feeds the bounded punch-target predictor; intentional tracking moves use the same motion guard while the Force hit result is stored through the single SessionBridge route.
13. Training end stops the Force session and returns the robot to weaving-ready, then automatic weaving resumes.
14. In parallel, the UI sessionizes the training data, generates the coaching report, stores the report/session in SQLite, and later reports can compare against the user's previous DB sessions.

## Coaching report reference

`ui/vision_coach.py` supplies the OpenAI coaching request with measured KO evidence plus a concise reference guide derived from the **England Boxing Level 1 Coaching Handbook**. The prompt explicitly keeps measured data/image evidence authoritative and does not infer unmeasured motion. The generated result also records the handbook title/publisher/reference URL in `coaching_reference`.

## Pause / resume

Pause keeps the personalized mitt pose but stops HitTest/Compliance and freezes the UI timer. Resume restarts HitTest and waits for the robot/Force stack to become ready before the timer resumes.

## Combination code

Existing combination code is preserved. This final integration does not add hook or uppercut mechanics; physical validation is first limited to jab/straight as requested.

## Final run

The final launcher also checks a force-workspace build stamp. If the project was moved to a different path or `src/` is newer than the last force build, it automatically rebuilds the workspace. Relocated/unstamped legacy colcon `build/install/log` artifacts are cleared first so old absolute paths cannot leak into the new runtime; same-root source changes use an incremental rebuild.

One-time setup/build:

```bash
./setup.sh --build-force
```

After `roboton` and Doosan services are ready, target-rig preflight without project robot motion:

```bash
./test_final.sh --hardware
```

USER mode:

```bash
./run_final.sh
```

ADMIN mode:

```bash
./run_final.sh --admin-mode
```

## Physical verification boundary

Software regression and static contracts cannot certify physical reach, collision clearance, TP safety configuration, real compliance feel, actual punch impact, or M0609 motion on the target rig. The first physical validation should therefore verify the flow in order: XZ weaving → camera-alignment handoff → punching ready → reach contact calibration → five-hit/zeroing calibration → jab/straight tracking → training stop → weaving return.
