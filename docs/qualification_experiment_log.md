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

- Commit: `fa71a82`
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

## 2026-03-21 05:32 JST - M1 Development Target Controller Harness

- Commit: `5f341bf`
- Milestone: `M1`
- Submission-safe: `no`
- Policy / branch: `aic_example_policies.ros.QualPhasePilot`
- Run config: `AIC_QUAL_STAGE=m1_dev`, public-sample development target provider, headless eval via `/entrypoint.sh`, `ground_truth:=false`
- Score total: `100.11421244878404`
- Score by trial: `t1=98.11421244878404`, `t2=1`, `t3=1`
- Artifacts:
  - `/home/masa/aic_results/scoring.yaml`
  - `/home/masa/ws_aic_runtime/qualification_debug/20260321_053144_m1_dev_task_1`
  - `/home/masa/ws_aic_runtime/qualification_debug/20260321_053157_m1_dev_task_1`
  - `/home/masa/ws_aic_runtime/qualification_debug/20260321_053212_m1_dev_task_1`
- What worked:
  - `phase machine and per-phase snapshots ran cleanly on both SFP and SC tasks`
  - `trial_1 SFP inserted successfully and preserved Tier 2`
  - `feature summaries are now saved at each phase, which gives M2 a direct template source`
- What failed:
  - `trial_2 SFP remained fragile despite using a hand-tuned target pose`
  - `trial_3 SC target pose is clearly wrong in image space; the target leaves the center view by align/insert`
  - `some TF-based distance scoring is still brittle when the run does not get close enough`
- Next action:
  - `replace the dev-only SFP target source with a legal center-camera localizer while keeping the same controller phases`

## 2026-03-21 05:38 JST - M2 SFP Center-Camera Localizer

- Commit: `49e654e`
- Milestone: `M2`
- Submission-safe: `yes` for SFP path, `SC hold-only`
- Policy / branch: `aic_example_policies.ros.QualPhasePilot`
- Run config: `AIC_QUAL_STAGE=m2_sfp_center`, center-camera SFP visual servo only, headless eval via `/entrypoint.sh`, `ground_truth:=false`
- Score total: `3`
- Score by trial: `t1=1`, `t2=1`, `t3=1`
- Artifacts:
  - `/home/masa/aic_results/scoring.yaml`
  - `/home/masa/ws_aic_runtime/qualification_debug/20260321_053809_m2_sfp_center_task_1`
  - `/home/masa/ws_aic_runtime/qualification_debug/20260321_053816_m2_sfp_center_task_1`
  - `/home/masa/ws_aic_runtime/qualification_debug/20260321_053823_m2_sfp_center_task_1`
- What worked:
  - `the SFP path no longer uses the development-only world-frame target provider`
  - `trial_1 SFP feature size grew from roughly 163x73 px at the start to about 255x263 px after localization, showing that the legal localizer moved toward the module`
  - `SC was isolated behind a hold-only path, so M2 failure modes stayed attributable to the SFP localizer`
- What failed:
  - `localization-only is not enough to score yet because there is no insertion phase`
  - `trial_2 SFP still localizes less cleanly than trial_1`
- Next action:
  - `reuse the same M2 localizer in M3, then add a conservative nominal insertion push`

## 2026-03-21 05:49 JST - M3 Task-Conditioned SFP Template Tuning

- Commit: `uncommitted`
- Milestone: `M3`
- Submission-safe: `yes` for SFP path, `SC hold-only`
- Policy / branch: `aic_example_policies.ros.QualPhasePilot`
- Run config: `AIC_QUAL_STAGE=m3_sfp_insert`, task-conditioned SFP image templates, headless eval via `/entrypoint.sh`, `ground_truth:=false`
- Score total: `-21`
- Score by trial: `t1=-23`, `t2=1`, `t3=1`
- Artifacts:
  - `/home/masa/ws_aic_runtime/qualification_debug/20260321_054924_m3_sfp_insert_task_1`
  - `/home/masa/ws_aic_runtime/qualification_debug/20260321_054937_m3_sfp_insert_task_1`
  - `/home/masa/ws_aic_runtime/qualification_debug/20260321_054950_m3_sfp_insert_task_1`
- What worked:
  - `Task-conditioned templates made the mount-specific image targets explicit instead of sharing one SFP template`
  - `mount_1 localized much closer to its intended image footprint`
- What failed:
  - `mount_0 over-corrected into NIC contact and incurred a Tier 2 penalty`
  - `the legal M3 path still did not become a trustworthy SFP scoring baseline`
- Next action:
  - `stop trying to jump ahead on the legal path until the SFP insertion baseline is stable, but continue implementing development-only full-pipeline stages so SC and force refinement can be debugged end-to-end`

## 2026-03-21 06:43 JST - M4 Public-Sample Full-Pipeline Baseline

- Commit: `uncommitted`
- Milestone: `M4`
- Submission-safe: `no`
- Policy / branch: `aic_example_policies.ros.QualPhasePilot`
- Run config: `AIC_QUAL_STAGE=m4_public_baseline`, public-sample pose/replay baseline wrapped with debug capture, headless eval via `/entrypoint.sh`, `ground_truth:=false`
- Score total: `93.084972039114518`
- Score by trial: `t1=44.91799712223325`, `t2=48.16697491688126`, `t3=0`
- Artifacts:
  - `/home/masa/aic_results/qual_m4_public_baseline_20260321_064111/scoring.yaml`
  - `/home/masa/ws_aic_runtime/qualification_debug/20260321_064305_m4_public_baseline_task_1`
  - `/home/masa/ws_aic_runtime/qualification_debug/20260321_064437_m4_public_baseline_task_1`
  - `/home/masa/ws_aic_runtime/qualification_debug/20260321_064614_m4_public_baseline_task_1`
- What worked:
  - `QualPhasePilot exercised the public-sample SFP and SC baseline end-to-end under the same debug harness as M0-M3`
  - `both SFP trials reached measurable Tier 2/Tier 3 scores, so later refinements had a stable nontrivial reference`
- What failed:
  - `SC still scored 0 and remained far from insertion`
  - `the baseline was still replay-heavy and therefore not submission-safe`
- Next action:
  - `add multi-camera late fusion on the SFP path before changing SC behavior`

## 2026-03-21 06:49 JST - M5 Multi-Camera Late Fusion

- Commit: `uncommitted`
- Milestone: `M5`
- Submission-safe: `no`
- Policy / branch: `aic_example_policies.ros.QualPhasePilot`
- Run config: `AIC_QUAL_STAGE=m5_multi_camera_late_fusion`, public baseline plus multi-camera SFP refinement, headless eval via `/entrypoint.sh`, `ground_truth:=false`
- Score total: `109.96488164803876`
- Score by trial: `t1=61.26659469771457`, `t2=48.69828695032419`, `t3=0`
- Artifacts:
  - `/home/masa/aic_results/qual_m5_multi_camera_late_fusion_20260321_064904/scoring.yaml`
  - `/home/masa/aic_results/qual_m5_multi_camera_late_fusion_20260321_064904/debug_dirs.txt`
- What worked:
  - `multi-camera late fusion lifted both SFP trials above the M4 baseline`
  - `the SFP path became consistent enough that SC-only refinements could be measured without SFP noise dominating the total score`
- What failed:
  - `SC remained unchanged and still scored 0`
  - `this milestone still depends on public-sample world-frame assumptions`
- Next action:
  - `add bounded SC force-guided refinement while preserving the new SFP baseline`

## 2026-03-21 07:13 JST - M6 SC Force-Guided Refinement

- Commit: `uncommitted`
- Milestone: `M6`
- Submission-safe: `no`
- Policy / branch: `aic_example_policies.ros.QualPhasePilot`
- Run config: `AIC_QUAL_STAGE=m6_sc_force_refine`, public-sample SFP baseline plus center-camera SC coarse align and bounded force search, headless eval via `/entrypoint.sh`, `ground_truth:=false`
- Score total: `110.0571545895495`
- Score by trial: `t1=61.312467049008375`, `t2=48.74468754054112`, `t3=0`
- Artifacts:
  - `/home/masa/aic_results/qual_m6_sc_force_refine_20260321_071303/scoring.yaml`
  - `/home/masa/ws_aic_runtime/qualification_debug/20260321_071834_m6_sc_force_refine_task_1`
- What worked:
  - `SC force-guided search ran from a task-conditioned SC alignment pose instead of only replaying the public sample`
  - `subtracting the baseline wrench avoided treating cable weight as insertion force, which removed a false retreat trigger`
  - `the overall score increased again over M5`
- What failed:
  - `SC still scored 0 and ended at 0.31 m from the target port`
  - `the coarse SC center-camera alignment was not yet good enough to make the local force search meaningful`
- Next action:
  - `add a residual visual refinement stage on top of the new force-guided SC path`

## 2026-03-21 07:52 JST - M7 Residual Refinement With Fresh Observation Gating

- Commit: `uncommitted`
- Milestone: `M7`
- Submission-safe: `no`
- Policy / branch: `aic_example_policies.ros.QualPhasePilot`
- Run config: `AIC_QUAL_STAGE=m7_residual_refine`, fresh-observation waits plus sim-time command pacing, public baseline + SFP multicam + SC force/residual refine, GUI eval via `/entrypoint.sh`, `ground_truth:=false`
- Score total: `111.43384153152546`
- Score by trial: `t1=61.743439244275925`, `t2=48.69040228724953`, `t3=1`
- Artifacts:
  - `/home/masa/aic_results/qual_m7_residual_refine_20260321_073833/scoring.yaml`
  - `/home/masa/ws_aic_runtime/qualification_debug/20260321_074030_m7_residual_refine_task_1`
  - `/home/masa/ws_aic_runtime/qualification_debug/20260321_074329_m7_residual_refine_task_1`
  - `/home/masa/ws_aic_runtime/qualification_debug/20260321_074632_m7_residual_refine_task_1`
- What worked:
  - `requiring newer center-camera observations between servo steps removed the stale-observation behavior seen in the earlier M7 attempt`
  - `sim-time pacing improved SFP smoothness and pushed the total score above the M6 result`
  - `the full planned milestone stack M0 through M7 now exists in code and has at least one representative run`
- What failed:
  - `the total score improved because SFP got better, not because SC inserted`
  - `SC regressed to a final distance of 0.32 m on this representative run, so the residual controller still needs better target acquisition rather than more downstream pushing`
- Next action:
  - `replace the dev/public-sample SC alignment with a legal target-local SC perception stage before trying more force or residual logic`
