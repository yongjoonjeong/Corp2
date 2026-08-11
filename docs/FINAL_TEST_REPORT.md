# KO Final Integration · Test Report · 2026-08-11

## Current checks completed in the integration workspace

- Final integration contract checker: **PASS**
  - separate weaving/punch-ready poses
  - robot BASE XZ weaving
  - SessionBridge ownership of Start/StopHitTest
  - Tool +Z mitt-face reach calibration → 5-hit Force centering → post-move wrench zero
  - UI `TRAINING_READY` gate
  - calibration zeroing / punch-ready UI + TTS contract
  - England Boxing Level 1 coaching-reference contract
- UI/API + ROS-independent Force/rebound/mitt regression: **154 PASS + 18 subtests PASS**
- ROS-independent Vision regression: **21 PASS**
- Python compile check: **PASS**
- YAML parse: **15 files / 0 errors**
- ROS `package.xml` parse: **4 files / 0 errors**
- JavaScript syntax: **PASS**

## Environment-limited checks

This execution environment does not provide the target ROS 2 Humble / Doosan runtime (`rclpy`, `dsr_msgs2`, ROS message packages), so tests that import live ROS node dependencies cannot be collected here. Those are environment limitations rather than observed assertion failures in the ROS-independent regression set.

The actual M0609 hardware, real Doosan services, physical force sensor behavior, camera devices and target-gym clearances are also unavailable here. Therefore this report does **not** claim physical robot validation.

On the target Ubuntu/ROS environment, run:

```bash
./setup.sh --build-force
./test_final.sh
./test_final.sh --hardware
```

The hardware preflight does not replace low-speed real-motion validation.

## Integration-specific changes verified

1. KO remains the base for UI, Vision, DB/reporting, Wakeword/STT and the final launcher.
2. Standalone's newer RT wrench fusion and reach-calibration logic are merged without replacing KO's higher-level UI/session flow.
3. Camera alignment completes before `training_start`; the training handoff is not sent merely by entering the alignment screen.
4. Weaving remains the user-confirmed robot BASE XZ U motion.
5. Weaving-ready and punching-ready poses remain separate.
6. First calibration advances the mitt along its face-normal Tool +Z and stores the force-contact reach correction.
7. Second calibration uses five valid Force hits for mitt-center post-correction.
8. Intentional calibration/training mitt moves use the motion guard so Compliance watchdog handling can distinguish commanded motion from uncontrolled drift.
9. After each 2nd-calibration correction move, the settled reference is recaptured and wrench zero is recalibrated before the next punch is admitted.
10. UI/TTS announces `영점 조정 중입니다. 잠시 기다려주세요.` and, when ready, `다시 펀치하세요.`
11. The UI countdown waits for `TRAINING_READY`, which is emitted only after both calibration stages are complete.
12. Current production scope remains jab/straight; hook/uppercut behavior was not newly implemented.
13. Coaching report generation now receives the England Boxing Level 1 Coaching Handbook reference while retaining measured-evidence-first rules.
14. Normal training stop ends the Force session, returns the robot to weaving-ready and resumes weaving; report/DB processing remains on the UI side.

## Physical validation boundary

Before normal punching, the target M0609 rig must verify at low speed: actual mitt-face Tool +Z approach direction, reach contact threshold/release behavior, 5-hit correction sign and magnitude, post-move zero timing, Compliance/rebound feel, bounded Vision target-follow motion, collision clearance and the full stop/return-to-weaving path.
