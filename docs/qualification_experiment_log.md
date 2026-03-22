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

## 2026-03-21 15:38 JST - S0 First End-To-End Submission-Safe Baseline

- Commit: `uncommitted`
- Milestone: `S0`
- Submission-safe: `yes`
- Policy / branch: `aic_example_policies.ros.QualPhasePilot`
- Run config: `AIC_QUAL_STAGE=submission_safe_v0`, GUI eval via `/entrypoint.sh`, `ground_truth:=false`, `DISPLAY=:1`
- Score total: `98.041744964269498`
- Score by trial: `t1=48.061318260132214`, `t2=48.980426704591785`, `t3=1.0`
- Artifacts:
  - `/home/masa/aic_results/qual_submission_safe_v0_20260321_151750/scoring.yaml`
  - `/home/masa/aic_results/qual_submission_safe_v0_20260321_151750/eval.log`
  - `/home/masa/aic_results/qual_submission_safe_v0_20260321_151750/model.log`
  - `/home/masa/aic_results/qual_submission_safe_v0_20260321_151750/debug_dirs.txt`
  - `/home/masa/aic_results/bag_trial_1_20260321_151941_882`
  - `/home/masa/aic_results/bag_trial_2_20260321_152407_545`
  - `/home/masa/aic_results/bag_trial_3_20260321_152831_017`
  - `/home/masa/ws_aic_runtime/qualification_debug/20260321_151944_submission_safe_v0_task_1`
  - `/home/masa/ws_aic_runtime/qualification_debug/20260321_152410_submission_safe_v0_task_1`
  - `/home/masa/ws_aic_runtime/qualification_debug/20260321_152833_submission_safe_v0_task_1`
- What worked:
  - `the run completed all 3 trials using only official runtime inputs`
  - `SFP stayed strong without public-sample world targets, scoring about 48 on both trials`
  - `debug artifacts clearly record the legal-only target provider and per-phase snapshots`
- What failed:
  - `SC still earned only Tier 1 and finished 0.20 m from the target port`
  - `the dominant bottleneck is now legal SC target acquisition, not downstream force/residual logic`
- Next action:
  - `preserve the S0 SFP path and replace only the SC acquisition stage with a stronger legal localizer`

## 2026-03-21 18:30 JST - S1 Submission-Safe Triangulated SC Translation-Only Acquisition

- Commit: `uncommitted`
- Milestone: `S1`
- Submission-safe: `yes`
- Policy / branch: `aic_example_policies.ros.QualPhasePilot`
- Run config: `AIC_QUAL_STAGE=submission_safe_v4`, GUI eval via `/entrypoint.sh`, `ground_truth:=false`, `DISPLAY=:1`
- Score total: `122.08548930859087`
- Score by trial: `t1=48.72334879624358`, `t2=48.987096754294114`, `t3=24.375043758053183`
- Artifacts:
  - `/home/masa/aic_results/qual_submission_safe_v4_20260321_181109/scoring.yaml`
  - `/home/masa/aic_results/qual_submission_safe_v4_20260321_181109/eval.log`
  - `/home/masa/aic_results/qual_submission_safe_v4_20260321_181109/model.log`
  - `/home/masa/aic_results/qual_submission_safe_v4_20260321_181109/debug_dirs.txt`
  - `/home/masa/ws_aic_runtime/qualification_debug/20260321_181301_submission_safe_v4_task_1`
  - `/home/masa/ws_aic_runtime/qualification_debug/20260321_181719_submission_safe_v4_task_1`
  - `/home/masa/ws_aic_runtime/qualification_debug/20260321_182141_submission_safe_v4_task_1`
- What worked:
  - `the legal-only path now scores above both S0 and the development-oriented M7 reference`
  - `keeping triangulated SC translation but disabling the SC optical-axis rotation heuristic materially improved the SC trial`
  - `SFP stayed stable while SC rose from Tier 1 only to a 24.38-point trial`
- What failed:
  - `SC still did not insert and finished 0.16 m from the target port`
  - `the SC path still spends too long in force/residual cleanup, which means the pre-insertion pose is better but not yet good enough`
- Next action:
  - `treat S1 as the new legal baseline and focus the next changes on tightening SC pre-insertion distance before adding more downstream complexity`

## 2026-03-21 22:25 JST - S2 Submission-Safe Learned SC Acquisition

- Commit: `uncommitted`
- Milestone: `S2`
- Submission-safe: `yes`
- Policy / branch: `aic_example_policies.ros.QualPhasePilot`
- Run config: `AIC_QUAL_STAGE=submission_safe_v6`, headless eval via `/entrypoint.sh`, `ground_truth:=false`, `DISPLAY=:1`, learned SC model `sc_uvz_v1`
- Score total: `126.58206055565613`
- Score by trial: `t1=48.21030515038397`, `t2=48.982548911699574`, `t3=29.389206493572582`
- Artifacts:
  - `/home/masa/aic_results/qual_submission_safe_v6_20260321_220649/scoring.yaml`
  - `/home/masa/aic_results/qual_submission_safe_v6_20260321_220649/eval.log`
  - `/home/masa/aic_results/qual_submission_safe_v6_20260321_220649/model.log`
  - `/home/masa/ws_aic_runtime/qualification_debug/20260321_220845_submission_safe_v6_task_1`
  - `/home/masa/ws_aic_runtime/qualification_debug/20260321_221302_submission_safe_v6_task_1`
  - `/home/masa/ws_aic_runtime/qualification_debug/20260321_221726_submission_safe_v6_task_1`
- What worked:
  - `the learned multi-view SC acquisition beat the previous legal best S1 and raised the overall score above 126`
  - `both SFP trials stayed near 48 points while the SC trial climbed to 29.39 without using DEV_TARGETS or public-sample world targets`
  - `switching _wait_for_observation to wall-clock timeout removed the sim-time stall that had made the v6/v7 runs hang`
- What failed:
  - `SC still did not insert; the final plug-port distance was 0.13 m`
  - `the learned SC stage still relies on hand-picked desired image/depth templates extracted from GT teacher data`
- Next action:
  - `treat S2 as the new winning legal baseline and test only small legal post-acquisition refinements against it`

## 2026-03-21 22:35 JST - X1 Legal Tool-Frame Force Search Probe

- Commit: `uncommitted`
- Milestone: `X1`
- Submission-safe: `yes`
- Policy / branch: `aic_example_policies.ros.QualPhasePilot`
- Run config: `AIC_QUAL_STAGE=submission_safe_v8`, SC-only fail-fast config `/home/masa/ws_aic/src/aic/aic_engine/config/learn_collect_sc_only.yaml`, headless eval via `/entrypoint.sh`, `ground_truth:=false`, `DISPLAY=:1`
- Score total: `33.404680405165706`
- Score by trial: `t1=33.404680405165706`
- Artifacts:
  - `/home/masa/aic_results/qual_submission_safe_v8_20260321_222558/scoring.yaml`
  - `/home/masa/aic_results/qual_submission_safe_v8_20260321_222558/eval.log`
  - `/home/masa/aic_results/qual_submission_safe_v8_20260321_222558/model.log`
  - `/home/masa/ws_aic_runtime/qualification_debug/20260321_222753_submission_safe_v8_task_1`
- What worked:
  - `adding a legal tool-frame bounded force search improved the isolated SC trial over S2's trial_3 score band`
  - `the SC-only run finished at 0.10 m final distance, which is better than the S2 representative full-run SC result`
- What failed:
  - `the tool-frame force search is too slow to be a good fail-fast refinement`
  - `a follow-up full-run probe started with the same v8 code but reached only 48.949048 + 41.896870 = 90.845918 after the first two SFP trials, meaning trial_3 would have needed 35.73614255565613 points just to match S2; that was already above the SC-only v8 result, so the run was aborted`
- Next action:
  - `keep S2 as the promoted best run and do not advance v8 without reducing its time cost and explaining the unexpected SFP regression`

## 2026-03-22 11:56 JST - P0 `center_uvz` Larger Balanced Model Gate

- Commit: `uncommitted`
- Milestone: `P0`
- Submission-safe: `yes`
- Policy / branch: `aic_example_policies.ros.QualPhasePilot`
- Run config: `AIC_QUAL_STAGE=submission_safe_v11`, `SFP-only` config [sfp_trials_only.yaml](/home/masa/ws_aic/src/aic/aic_engine/config/sfp_trials_only.yaml), headless eval via `/entrypoint.sh`, `ground_truth:=false`, `DISPLAY=:1`, learned SFP model `/home/masa/ws_aic_runtime/learned_port_models/sfp_centeruvz_rand20_allphases_v0_20260322`
- Score total: `71.320068644735088`
- Score by trial: `t1=35.220362710871615`, `t2=36.099705933863473`
- Artifacts:
  - `/home/masa/aic_results/qual_submission_safe_v11_20260322_115655/eval.log`
  - `/home/masa/aic_results/qual_submission_safe_v11_20260322_115655/model.log`
  - `/home/masa/ws_aic_runtime/qualification_debug/20260322_115844_submission_safe_v11_task_1`
  - `/home/masa/ws_aic_runtime/qualification_debug/20260322_120151_submission_safe_v11_task_1`
- What worked:
  - `the 840-sample retrained model moved reliably and completed both SFP trials`
  - `the runtime aux-vector bug was fixed, so the learned branch was no longer stuck at score 2`
- What failed:
  - `the new learned SFP route still scored far below the old SFP-pair baseline of 97.19285406208354`
  - `both trials remained proximity-only and never crossed into insertion scoring`
- Next action:
  - `stop treating center_uvz-only as the mainline and switch the PDCA loop to teacher-feasibility gating`

## 2026-03-22 12:08 JST - T0 Randomized SFP GT Teacher Feasibility v0

- Commit: `uncommitted`
- Milestone: `T0`
- Submission-safe: `no`
- Policy / branch: `aic_example_policies.ros.QualPhasePilot`
- Run config: `AIC_QUAL_STAGE=teacher_feasibility_v0`, randomized `SFP-only` config `/home/masa/ws_aic_runtime/generated_configs/t0_teacher_sfp_rand2_seed91.yaml`, headless eval via `/entrypoint.sh`, `ground_truth:=true`, `DISPLAY=:1`
- Score total: `86.653740214899685`
- Score by trial: `t1=41.779318087841951`, `t2=44.874422127057734`
- Artifacts:
  - `/home/masa/aic_results/qual_teacher_feasibility_v0_20260322_120830/scoring.yaml`
  - `/home/masa/aic_results/qual_teacher_feasibility_v0_20260322_120830/eval.log`
  - `/home/masa/aic_results/qual_teacher_feasibility_v0_20260322_120830/model.log`
  - `/home/masa/ws_aic_runtime/qualification_debug/20260322_121027_teacher_feasibility_v0_task_1`
  - `/home/masa/ws_aic_runtime/qualification_debug/20260322_121403_teacher_feasibility_v0_task_1`
- What worked:
  - `both randomized SFP tasks completed successfully under a ground-truth teacher`
  - `trial_2 reached a final plug-port distance of 0.03 m, showing the teacher can approach the port mouth closely`
- What failed:
  - `the teacher still missed the T0 gate badly, reaching only 86.65 / 200 instead of the required 150 / 200`
  - `the terminal push phase remained proximity-led and did not generate insertion events`
- Next action:
  - `replace the fixed terminal push with a more insertion-aware GT near-contact controller before collecting any student data`

## 2026-03-22 12:18 JST - T0 Randomized SFP GT Teacher Feasibility v1

- Commit: `uncommitted`
- Milestone: `T0`
- Submission-safe: `no`
- Policy / branch: `aic_example_policies.ros.QualPhasePilot`
- Run config: `AIC_QUAL_STAGE=teacher_feasibility_v0`, same randomized `SFP-only` config `/home/masa/ws_aic_runtime/generated_configs/t0_teacher_sfp_rand2_seed91.yaml`, headless eval via `/entrypoint.sh`, `ground_truth:=true`, `DISPLAY=:1`, plus a GT terminal-contact loop before the final push
- Score total: `86.562241559624113`
- Score by trial: `t1=42.659702234323316`, `t2=43.902539325300786`
- Artifacts:
  - `/home/masa/aic_results/qual_teacher_feasibility_v0_20260322_121820/scoring.yaml`
  - `/home/masa/aic_results/qual_teacher_feasibility_v0_20260322_121820/eval.log`
  - `/home/masa/aic_results/qual_teacher_feasibility_v0_20260322_121820/model.log`
  - `/home/masa/ws_aic_runtime/qualification_debug/20260322_122017_teacher_feasibility_v0_task_1`
  - `/home/masa/ws_aic_runtime/qualification_debug/20260322_122244_teacher_feasibility_v0_task_1`
- What worked:
  - `the GT terminal-contact loop slightly improved trial_1 tier-3 proximity`
  - `the revised teacher still completed both randomized SFP tasks cleanly`
- What failed:
  - `the total score was effectively unchanged, so the added GT terminal loop did not solve insertion`
  - `the teacher still stayed proximity-led and therefore still failed the 150 / 200 T0 gate`
- Next action:
  - `redesign the GT teacher around lower-jerk, axis-aware, recovery-capable near-contact insertion instead of collecting more student labels`
