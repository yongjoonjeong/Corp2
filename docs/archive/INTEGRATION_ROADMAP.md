# KO Stepwise Integration Roadmap

## Phase 1 — UI modes and operator UX

Status: implemented in this package.

- USER / ADMIN runtime mode
- USER simplified recognition status
- ADMIN detailed diagnostics + System Settings
- contextual voice-command help
- preserve one-command-per-wake policy

## Phase 2 — Longitudinal data + force interface

After Phase 1 is checked on the target machine:

- store structured improvement targets from each session
- compare prior/current measurable metrics locally
- map statuses such as `improved / maintained / declined`
- add KO-side adapter/schema for `/mitt/hit_result`
- store real force/hit fields only when present
- USER hides unavailable force data; ADMIN reports source status

## Phase 3 — Final OpenAI coaching report

- preserve representative-image analysis
- add current numeric vision results
- add prior feedback target + deterministic trend result
- add real force/hit results when available
- enforce “do not infer unmeasured facts”
- output previous-feedback progress + current strength + next focus

## Phase 4 — Combination 1–5 command/UI model

- STT parse of combination number/name
- UI sequence/progress display
- database representation
- admin view of parsed sequence
- actual robot mitt-motion execution remains disconnected until robot motion code is ready

## Final robot/force integration

Connect the completed robot-team control logic to the prepared interfaces after its safety/force-control parameters are finalized and verified on the real system.
