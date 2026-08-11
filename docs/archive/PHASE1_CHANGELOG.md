# KO Phase 1 Integration — USER / ADMIN mode

Base: `sandbag_vision_realtime_v3_deploy_20260807_r2`

## Scope

This phase changes the presentation/diagnostic layer only. Existing vision runtime, impact detection, wake-word one-command policy, robot weaving node, and OpenAI image coaching flow are preserved.

### USER MODE

Run:

```bash
./run_integrated.sh --user-mode
```

`--user-mode` is also the default when no mode flag is supplied.

- Hides developer-only BASE P/V telemetry.
- Hides System Settings.
- Shows one simplified VISION STATUS card:
  - `정상 인식`
  - `인식 불안정`
  - `인식 불가`
- Uses recent vision frames to avoid status flicker.
- Keeps user-facing training values and controls.
- Shows contextual voice-command examples for the current screen.

### ADMIN MODE

Run:

```bash
./run_integrated.sh --admin-mode
```

- Shows an `ADMIN MODE` badge.
- Keeps the detailed real-time vision telemetry used for field testing.
- Enables the System Settings menu.
- System Settings exposes camera/target/mitt/fist-3D/robot/wake/STT/database status plus a live diagnostic snapshot and recent event.

Compatibility aliases are also accepted:

```text
-user_mode  --user_mode  -admin_mode  --admin_mode
```

## Wake Word policy

This phase intentionally preserves the current test policy: one command per wake activation. It does not restore the earlier continuous voice session.

## Not included yet

- Previous-session feedback tracking
- Force result ingestion/report integration
- Improved final OpenAI longitudinal report
- Combination 1–5 execution model
- New robot force-control or mitt-motion behavior

These are scheduled for later phases after Phase 1 is checked in the target environment.

## CPU/GPU 자동 설치 보완

- `setup.sh`가 NVIDIA GPU 존재 여부를 자동 감지합니다.
- GPU 없는 PC에서는 정상 import되는 기존 torch/torchvision을 CPU용으로 재사용합니다.
- CPU PC에서 CUDA 12.8 PyTorch를 강제로 다운로드하던 문제를 제거했습니다.
- GPU PC에서는 기존 CUDA PyTorch가 실제 사용 가능할 때 재사용하고, 아니면 cu128 조합을 설치합니다.
- `--cpu`, `--cuda`, `--auto`, `KO_TORCH_MODE`, `KO_FORCE_TORCH_REINSTALL`을 지원합니다.
- `requirements.txt`에서 PyTorch 고정 핀을 제거하고 `setup.sh`가 먼저 관리하도록 변경했습니다.
