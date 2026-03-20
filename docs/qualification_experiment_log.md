# Qualification Experiment Log

This file is for short, high-signal experiment notes.
It should stay concise enough to scan quickly.

## How To Use

- Add one entry per meaningful experiment batch or milestone completion.
- Prefer one entry that summarizes a group of closely related runs over many tiny entries.
- Always include the representative result path and the current commit hash.
- If the run is not submission-safe, say so explicitly.

## Entry Template

```md
## YYYY-MM-DD HH:MM JST - Short Title

- Commit: `git-sha`
- Milestone: `M0` / `M1` / ...
- Submission-safe: `yes` / `no`
- Policy / branch: `...`
- Run config: `...`
- Score total: `...`
- Score by trial: `t1=...`, `t2=...`, `t3=...`
- Artifacts:
  - `/home/masa/aic_results/.../scoring.yaml`
  - `/home/masa/aic_results/.../bag_trial_...`
  - `/home/masa/ws_aic/tmp/...`
- What worked:
  - `...`
- What failed:
  - `...`
- Next action:
  - `...`
```

## 2026-03-21 04:19 JST - Public-Sample Replay Reference

- Commit: `uncommitted`
- Milestone: `reference`
- Submission-safe: `no`
- Policy / branch: `aic_example_policies.ros.PublicTrialPosePilot`
- Run config: `public sample config`, replay-heavy SC strategy
- Score total: `202.68080794003197`
- Score by trial: `t1=61.08252149993895`, `t2=97.97300578637278`, `t3=43.62528065372023`
- Artifacts:
  - `/home/masa/aic_results/codex_publictrialposepilot_sc_replay_20260321_041916/scoring.yaml`
- What worked:
  - `SFP public-sample trials scored strongly`
  - `SC reached near-insertion distance`
- What failed:
  - `strategy is not submission-safe because it depends on public-sample structure`
- Next action:
  - `replace replay assumptions with target-local perception and controller milestones`

## 2026-03-21 05:21 JST - M0 Observability Harness

- Commit: `uncommitted`
- Milestone: `M0`
- Submission-safe: `yes`
- Policy / branch: `aic_example_policies.ros.QualPhasePilot`
- Run config: `AIC_QUAL_STAGE=m0`, headless eval via `/entrypoint.sh`, `ground_truth:=false`
- Score total: `3`
- Score by trial: `t1=1`, `t2=1`, `t3=1`
- Artifacts:
  - `/home/masa/aic_results/scoring.yaml`
  - `/home/masa/ws_aic_runtime/qualification_debug/20260321_052118_m0_task_1`
  - `/home/masa/ws_aic_runtime/qualification_debug/20260321_052126_m0_task_1`
  - `/home/masa/ws_aic_runtime/qualification_debug/20260321_052133_m0_task_1`
- What worked:
  - `all three trials completed without a policy crash`
  - `initial/final image montages, metadata, and phase timelines were saved`
  - `the task mix included both SFP and SC, so the harness already captured both views`
- What failed:
  - `this milestone intentionally does not move toward insertion, so only Tier 1 scored`
  - `the current scoring output path is overwritten in-place, so later milestones must copy or record it immediately`
- Next action:
  - `commit M0, then add the M1 development-only target provider and phase-by-phase controller logging`
