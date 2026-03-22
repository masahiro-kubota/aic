# Qualification Strategy Notes

This document is a working note for developing a submission-oriented policy.
It is intentionally stricter than "what scores well on the public sample."

## Goal

Build a policy that remains valid when:

- the task board pose and yaw are randomized
- NIC cards and SC ports move along their rails
- the task switches between SFP and SC insertions
- the runtime environment only exposes officially supported interfaces

## Hard Constraints

The policy should assume the following:

- Use only officially provided interfaces.
- Do not depend on `/scoring/*`, `gz_server/*`, or other backend-only channels.
- Do not depend on `ground_truth:=true`.
- Do not assume a fixed world pose for the task board, NIC card, or SC port.
- Do not assume the public `sample_config.yaml` will match evaluation trials.

## What Is Allowed

The following inputs are valid and should be treated as the main runtime signals:

- `Task`
- `Observation.left_image`, `center_image`, `right_image`
- `Observation.*_camera_info`
- `Observation.wrist_wrench`
- `Observation.joint_states`
- `Observation.controller_state`
- official `/tf` and `/tf_static` for robot and camera kinematics

Ground-truth task poses may still be useful for offline training and debugging, but
they should not be part of the runtime policy logic.

## What To Avoid

The following approaches are not submission-safe:

- replaying world-frame trajectories extracted from the public sample
- branching on public-trial identity or `sample_config.yaml` specifics
- using task-board or cable ground-truth TF at runtime
- hardcoding fixed world coordinates for pre-insertion or insertion

## Observed Camera Characteristics

I checked the actual wrist-camera streams from a live local launch and saved example
captures under:

- `/home/masa/ws_aic/tmp/camera_capture_20260321_0452/summary.json`
- `/home/masa/ws_aic/tmp/camera_capture_20260321_0453/montage.png`
- `/home/masa/ws_aic/tmp/camera_capture_20260321_0454/montage.png`

I also cross-checked these observations against the qualification and task-board docs.

What the live camera setup looks like:

- three rectified RGB wrist cameras: left, center, and right
- image size is `1152 x 1024` for all three cameras
- intrinsics are identical across cameras: `fx ~= fy ~= 1236.6 px`,
  `cx = 576`, `cy = 512`
- distortion is effectively zero in the provided `CameraInfo`
- approximate field of view is `49.95 deg` horizontal and `44.98 deg`
  vertical

What matters for policy design:

- this is a close-range wrist view, not a global board camera
- the gripper, grasped plug, and sometimes the loose cable occupy a large part
  of the image
- the three cameras provide different viewpoints, but strong overlap should not
  be assumed at all times
- the docs guarantee the target port is within view during qualification, but
  they do not guarantee that the whole board or all rails are visible at once

Practical implication:

- whole-board reconstruction from a single frame is a weak default assumption
- target-local perception is a better fit than world-frame replay
- multi-camera late fusion is more realistic than relying on dense stereo
- final alignment should use local visual error and contact feedback

## Recommended Runtime Strategy

The recommended policy should be built in three stages.

### 1. Target Localization

Use `Task` to decide what must be found:

- SFP: which NIC module and which SFP port
- SC: which SC port family and which target port

Then estimate the target from wrist-camera observations:

- detect the relevant component or port in image space
- estimate a target pose or a lower-dimensional rail-relative parameterization
- combine image observations with camera calibration and robot kinematics

This step should produce a target-local insertion frame, not a world-fixed pose.

### 2. Target-Local Approach

Once a target frame is estimated:

- move to a pre-insertion pose expressed relative to the detected target
- align orientation to the port geometry
- keep the controller low-risk and smooth enough to preserve Tier 2

For early iterations, it is acceptable to use hand-designed relative approach poses
for SFP and SC, as long as they are attached to the detected target frame.

### 3. Final Insertion

The last few millimeters should not be pure open loop.

Use:

- visual alignment near the port opening
- low-speed motion along the insertion axis
- wrist wrench feedback to detect contact and guide small corrective search

Preferred final behaviors:

- axial insertion with low stiffness and sufficient damping
- small lateral search or raster/spiral search when contact indicates misalignment
- time-bounded recovery rather than aggressive pushing

## Recommended Model Shapes

The following designs are worth pursuing, in order.

### Option A: Perception + Analytic Controller

- detect port or target module from images
- estimate target frame
- use hand-designed target-relative motion primitives
- use force feedback for final insertion

Why this is good:

- rule-compliant
- interpretable
- likely fastest path to a robust baseline

### Option B: Rail-Parameter Estimation

Instead of full 6-DoF pose estimation, predict only the structured variables that
actually randomize:

- task board yaw / coarse placement
- NIC rail selection and translation / yaw
- SC rail translation

Why this is good:

- lower-dimensional problem
- matches the task-board structure
- easier to generalize than full absolute replay

### Option C: Task-Conditioned Residual Policy

Use `Task` + images + controller state to predict local motion deltas around the
current TCP pose.

Why this is good:

- directly uses official inputs
- can learn corrections for grasp error and cable deformation
- works well as a refinement layer on top of Option A or B

## Concrete Method Candidates

These are the method candidates that best match the actual wrist-camera
observations and the qualification rules.

### Candidate 1: Rail-Structured Perception + Port-Centered Servo

Runtime idea:

- use `Task` to choose the relevant target family: NIC/SFP or SC
- estimate only the structured variables that actually randomize:
  rail identity, rail translation, and for NIC also local yaw
- convert that estimate into a coarse pre-insertion pose
- switch to image-based visual servoing on the target port opening
- finish with low-speed insertion and wrench-based correction

Why it matches the observed images:

- it does not require the whole board to be visible
- it uses the known rail structure from the docs
- it treats the final stage as a local image-alignment problem, which fits the
  narrow wrist-camera view

Main risk:

- the coarse rail estimate may still need enough board context to distinguish
  similar rails reliably

### Candidate 2: Task-Conditioned Port Heatmap + Axis Estimation

Runtime idea:

- predict a target heatmap for the port center and a local insertion-axis
  direction from the camera images
- optionally predict confidence and an "already aligned" score
- drive the TCP to reduce pixel error in the center camera
- use left/right cameras mainly to disambiguate yaw and lateral offset

Why it matches the observed images:

- the cameras are rectified and calibrated
- pixel-level servoing is natural when the camera is mounted on the wrist
- this avoids brittle full-scene 6-DoF recovery from a partially occluded view

Main risk:

- controller tuning matters a lot; a naive visual servo loop can oscillate or
  overreact to occlusion by the plug

### Candidate 3: Keypoint-Based Target Frame Estimation

Runtime idea:

- detect a small set of target-specific keypoints on the visible component or
  port opening
- fit a target-local frame using known geometry and camera intrinsics
- follow with analytic target-relative motion primitives
- use force-guided micro-search only in the last few millimeters

Why it matches the observed images:

- it uses the strong geometric prior of SFP and SC connectors
- it can work from partial local views rather than requiring a full board image
- it gives an interpretable target frame for debugging

Main risk:

- it requires labeled keypoints or synthetic rendering pipelines

### Candidate 4: Opportunistic Multi-View Triangulation

Runtime idea:

- detect the same target in two or three wrist cameras when overlap exists
- triangulate the target center and local axis using official camera TF and
  calibration
- fall back to Candidate 1, 2, or 3 when only one good view is available

Why it matches the observed images:

- all three cameras are synchronized enough for local geometry fusion
- it can reduce depth ambiguity in the final approach

Main risk:

- view overlap is not guaranteed, so this is not a good sole strategy

### Candidate To Avoid As The Primary Approach

Whole-board segmentation followed by single-shot full board 6-DoF estimation
from one wrist image is probably not the best first bet.

Reason:

- the real wrist-camera view is too local and too self-occluded to count on
  stable full-board visibility
- the challenge randomization is structured enough that lower-dimensional target
  estimation is a better fit

## Current Recommendation

If I had to pick one concrete direction now, I would build:

1. `Task`-conditioned rail-structured coarse localization
2. port-centered visual servo for the last approach
3. wrench-guided insertion and bounded micro-search at contact

That stack maps best to both:

- what the qualification docs randomize
- what the actual wrist-camera images look like

The best learning-heavy variant of this stack is:

- a shared perception model that predicts rail parameter, port center heatmap,
  and insertion-axis cues
- a non-learning target-relative controller
- a small learned residual only near contact if needed

## Chosen Implementation Path

To reduce debugging complexity, the implementation path should be narrower than
the full strategy space.

The chosen default stack is:

1. `Task`-conditioned target perception
2. late fusion across wrist cameras, but only after a single-camera baseline works
3. deterministic target-relative state machine controller
4. low-speed force-guided insertion and bounded recovery
5. learned residuals only after the deterministic stack is measurable

The concrete representation to start with is:

- predict a target heatmap for the port center
- predict insertion-axis cues and confidence
- convert this into a target-local frame
- drive a hand-designed controller from that frame

This means:

- we are not starting with end-to-end imitation learning
- we are not starting with whole-board 6-DoF estimation
- we are not starting with a learned controller

## Development Principles

The implementation should follow these rules.

### 1. Add One Source Of Complexity At A Time

Only one major uncertainty should be introduced per milestone:

- first controller observability
- then target sourcing
- then submission-safe perception
- then SC support
- then multi-camera fusion
- then contact refinement
- then learning-based refinement

### 2. Prefer Low Score With Clear Failure Over High Score With Opaque Failure

A milestone is good if it tells us exactly which subsystem is broken:

- perception
- target-frame construction
- approach control
- insertion control
- recovery logic

### 3. Every Milestone Must Produce Human-Inspectable Artifacts

Each run should save enough evidence to answer:

- what did the cameras see
- what target did the policy think it saw
- which control phase did it enter
- what force/contact behavior happened
- why did it stop or fail

### 4. Build SFP First, Then SC

SFP should be the first submission-safe baseline because:

- the local geometry is simpler
- the public docs describe the structure clearly
- it is easier to separate localization errors from insertion errors

SC should reuse the same pipeline after SFP is measurable.

## Step-By-Step Execution Plan

The point of this plan is not to maximize score immediately.
The point is to create a policy that can be debugged stage by stage.

### M0. Observability Harness

Goal:

- create a run harness that makes failures inspectable

Implement:

- a new policy skeleton with an explicit phase machine:
  `idle -> acquire_target -> approach -> align -> insert -> recover -> done`
- save per-run debug artifacts:
  - left/center/right image montage
  - estimated target metadata as JSON
  - phase transition timeline
  - wrench summary
  - final controller-state summary
- write all debug outputs into the result directory or a sibling debug folder

Expected score:

- no score target
- even a `0` is acceptable if the run completes and artifacts are saved

Exit criteria:

- one complete trial can be run without crashing
- debug files can be inspected after the run
- we can tell whether failure came before or after target acquisition

Commit trigger:

- commit when the harness is stable and artifacts are readable

### M1. Development-Only Controller Harness

Goal:

- validate the controller state machine before trusting perception

Implement:

- plug a temporary target provider into the controller
- this target provider may use a development-only source such as:
  - manually specified relative target offsets
  - existing public-sample reference targets
  - offline-inspected target frames
- keep the controller interface identical to the final one so the target provider
  can later be swapped out

Expected score:

- low but non-zero score is enough
- this stage is for controller debugging, not for submission quality

Exit criteria:

- approach, align, insert, and recover phases all execute in the right order
- controller can reach a pre-insertion pose reproducibly
- obvious controller bugs are removed before adding learned perception

Commit trigger:

- commit when the controller is stable enough that future failures are likely to
  be perception-related

### M2. First Submission-Safe Baseline: SFP, Single Camera

Goal:

- replace the dev-only target provider with the simplest legal perception path

Implement:

- use only the center camera at first
- predict:
  - port center heatmap
  - insertion-axis cue
  - confidence
- construct a coarse target-local frame from that prediction and camera intrinsics
- drive only SFP trials through this path

Expected score:

- still modest
- success means "moves toward the correct target for the right reason"

Exit criteria:

- on repeated SFP trials, the robot visibly approaches the correct NIC port
- failures can be labeled as:
  - missed detection
  - wrong depth/standoff
  - wrong orientation
  - insertion/contact failure

Commit trigger:

- commit when SFP target acquisition is clearly working on more than one layout

### M3. SFP Target-Relative Baseline With Bounded Insertion

Goal:

- get a low-risk, fully legal SFP baseline that can score modestly but reliably

Implement:

- add conservative target-relative approach poses
- add low-speed insertion along the estimated axis
- stop on contact anomalies instead of pushing aggressively
- add bounded retreat and retry once

Expected score:

- moderate SFP score is enough
- reliability matters more than peak score

Exit criteria:

- repeated SFP runs produce a consistent approach pattern
- Tier 2 does not collapse due to chaotic motion
- insertion failures are mostly local alignment issues, not gross targeting errors

Commit trigger:

- commit when this becomes the first trustworthy submission-safe baseline

### M4. SC Support Using The Same Representation

Goal:

- extend the exact same perception/controller stack to SC

Implement:

- add SC-specific heatmap and axis heads or a task-conditioned shared head
- add SC pre-insertion geometry and alignment rules
- keep the controller phases unchanged

Expected score:

- SC may still underperform
- success means SC failures are understandable and isolated

Exit criteria:

- the policy routes SC through the same state machine
- SC reaches a meaningful pre-insertion pose
- failure modes are mostly final-alignment/contact related

Commit trigger:

- commit when both SFP and SC run through the same debuggable pipeline

### M5. Multi-Camera Late Fusion

Goal:

- improve robustness without destroying debuggability

Implement:

- keep the single-camera path as fallback
- fuse left/right camera predictions only after center-camera baseline works
- use multi-view only for:
  - confidence reweighting
  - lateral disambiguation
  - optional depth refinement

Expected score:

- moderate improvement in reliability

Exit criteria:

- fusion beats center-only on a repeated test set
- when fusion is uncertain, the system falls back rather than behaving erratically

Commit trigger:

- commit when fusion helps more often than it hurts

### M6. Force-Guided Final Insertion And Recovery

Goal:

- convert near-miss approaches into stable partial or full insertions

Implement:

- contact-triggered low-speed insertion mode
- small bounded lateral search
- explicit retreat conditions
- per-attempt force and duration caps

Expected score:

- this is where score should start rising meaningfully

Exit criteria:

- contact behavior is smoother
- fewer runs fail at `0.01m`-style near misses
- recovery produces understandable outcomes instead of noise

Commit trigger:

- commit when insertion quality improves without introducing unsafe thrashing

### M7. Learned Residuals Near Contact

Goal:

- add learning only where the deterministic stack is consistently close

Implement:

- keep the coarse perception and state machine fixed
- learn only a small residual around:
  - final lateral correction
  - yaw correction
  - insertion-axis correction

Expected score:

- potentially the first high-score milestone

Exit criteria:

- residual policy improves a stable baseline instead of masking a broken one
- ablations show the residual is actually adding value

Commit trigger:

- commit when the residual reliably beats the non-residual baseline

## Training Strategy

Training and data generation should intentionally cover randomized scenes.

Recommended training setup:

- generate many randomized task-board configurations
- export worlds when useful for labeling or cross-simulator training
- train on both SFP and SC with the same runtime interface
- train with grasp perturbations and small visual variations
- evaluate with `ground_truth:=false`

Do not optimize only against the public three-trial sample.

## Immediate Implementation Plan

The next concrete work items should follow the milestone order above:

### P0. Build M0

- create the policy skeleton and debug artifact writer
- do not optimize score yet

### P1. Build M1

- validate the controller phases with a temporary target provider
- remove obvious controller bugs first

### P2. Build M2 And M3 For SFP

- first legal perception baseline
- first reliable low-risk SFP insertion baseline

### P3. Build M4 For SC

- extend the same representation and controller

### P4. Build M5 Through M7

- add multi-camera fusion
- add force-guided refinement
- add learned residuals only after the baseline is stable

## Concrete Milestone Execution Plan

This section is the execution contract for the actual implementation work.
The policy should move through these milestones in order, without skipping M1
or M2 just because a later idea seems attractive.

Each milestone must satisfy three conditions before moving on:

- the implementation goal is present in code
- one representative run is logged in `qualification_experiment_log.md`
- one commit records the milestone score, artifact path, and dominant failure mode

## Milestone Tracker

This tracker exists to make the current state obvious at a glance.
It should be updated whenever a milestone is completed or blocked.

| Milestone | Focus | Current status | Gate to move on |
| --- | --- | --- | --- |
| `M0` | observability harness | `done` | artifacts saved for all trials |
| `M1` | development-only controller harness | `done` | controller phases behave predictably |
| `M2` | legal SFP center-camera localizer | `done` | legal targeting is visible and repeatable |
| `M3` | legal SFP insertion baseline | `implemented, still weak` | SFP must score above pure Tier 1 without regressing into chaotic motion |
| `M4` | legal SC baseline | `done (development-oriented baseline)` | replace replay assumptions with a legal SC target provider |
| `M5` | multi-camera late fusion | `done` | preserve the M5 SFP gain while making SC more legal |
| `M6` | force-guided final insertion and recovery | `done` | keep the bounded search, but feed it a better SC pre-insertion pose |
| `M7` | learned residual near contact | `done` | upgrade SC target acquisition before tuning the residual loop further |
| `S0` | first end-to-end submission-safe baseline | `done` | keep SFP at this level while improving SC target acquisition without reintroducing dev/public assumptions |
| `S1` | submission-safe triangulated SC acquisition | `done` | preserve the legal SFP path and keep improving SC final distance without reintroducing runtime target leaks |
| `S2` | submission-safe learned SC acquisition | `done` | keep the learned SC gain while replacing hand-picked teacher templates with a cleaner legal target representation |

## Executed Milestone Results

This table records the representative score that currently stands for each
implemented milestone.
The paths below should be the first place to look when a later change regresses.

| Milestone | Representative score | Representative artifact | Key takeaway |
| --- | --- | --- | --- |
| `M0` | `3.0` | `/home/masa/ws_aic_runtime/qualification_debug/20260321_052118_m0_task_1` | observability harness is sound |
| `M1` | `100.11421244878404` | `/home/masa/ws_aic_runtime/qualification_debug/20260321_053144_m1_dev_task_1` | controller phases can score when given a development target |
| `M2` | `3.0` | `/home/masa/ws_aic_runtime/qualification_debug/20260321_053809_m2_sfp_center_task_1` | legal SFP localizer moves toward the right module but does not insert yet |
| `M3` | `-21.0` | `/home/masa/ws_aic_runtime/qualification_debug/20260321_054924_m3_sfp_insert_task_1` | legal fixed-push SFP insertion is still too brittle |
| `M4` | `93.084972039114518` | `/home/masa/aic_results/qual_m4_public_baseline_20260321_064111/scoring.yaml` | first nontrivial full-pipeline reference |
| `M5` | `109.96488164803876` | `/home/masa/aic_results/qual_m5_multi_camera_late_fusion_20260321_064904/scoring.yaml` | multi-camera SFP refinement gives a clear gain |
| `M6` | `110.0571545895495` | `/home/masa/aic_results/qual_m6_sc_force_refine_20260321_071303/scoring.yaml` | bounded SC force search is implemented and measurable |
| `M7` | `111.43384153152546` | `/home/masa/aic_results/qual_m7_residual_refine_20260321_073833/scoring.yaml` | fresh-observation gating improves the full stack, but SC perception is still the bottleneck |
| `S0` | `98.041744964269498` | `/home/masa/aic_results/qual_submission_safe_v0_20260321_151750/scoring.yaml` | first full legal-only baseline; SFP is strong, SC still scores only Tier 1 |
| `S1` | `122.08548930859087` | `/home/masa/aic_results/qual_submission_safe_v4_20260321_181109/scoring.yaml` | triangulated SC translation-only acquisition lifts the legal path well above `S0`, including nontrivial SC scoring |
| `S2` | `126.58206055565613` | `/home/masa/aic_results/qual_submission_safe_v6_20260321_220649/scoring.yaml` | learned multi-view SC acquisition becomes the new legal best while preserving strong SFP performance |

### Current State

As of `2026-03-21`, the milestone stack through `M7` is implemented and has
representative runs, and the submission-safe track now has three concrete
reference points: the first legal baseline (`S0`), the triangulated SC upgrade
(`S1`), and the current best learned-SC run (`S2`).

What is already true:

- `M0` through `M7` all exist in code and have at least one representative run
- the total score increased step by step across the development-oriented stages:
  `M4 -> M5 -> M6 -> M7 = 93.08 -> 109.96 -> 110.06 -> 111.43`
- fresh-observation gating and sim-time pacing improved the reliability of the
  SFP stages
- `S0` proves that the same policy can complete all 3 trials using only
  official runtime inputs and still score `98.04`
- `S0` keeps the two SFP trials near the development-oriented baseline:
  `t1=48.06`, `t2=48.98`
- `S1` raises the submission-safe track to `122.09` while keeping the SFP path
  stable: `t1=48.72`, `t2=48.99`
- `S1` is the first legal-only run where the SC trial scores substantially
  above Tier 1: `trial_3=24.38`
- `S2` raises the legal best to `126.58` with `trial_3=29.39` while keeping the
  two SFP trials in the same strong band: `t1=48.21`, `t2=48.98`
- the learned SC acquisition in `S2` is the first legal-only path in this repo
  that clearly beats the triangulated `S1` baseline on total score

What is not true yet:

- the submission-safe path still does not achieve SC insertion
- `S2` still leaves the SC final plug-port distance at `0.13 m`
- further force/residual tuning is no longer the first blocker; the next gain
  still depends on a better legal SC pre-insertion pose

### Historical Blocking Issue

As of `2026-03-21`, the development milestones are no longer the main blocker.
The current blocker is the gap between `S2` and a submission-safe SC insertion
path.

What is already true:

- `M0` exists and saves usable debug artifacts
- `M1` proved the controller state machine can score on at least one SFP trial
- `M2` proved the legal SFP localizer moves toward the correct module
- `S0` proved that legal-only runtime inputs are sufficient for strong SFP
  scoring on both public sample SFP trials
- `S1` proved that legal-only multi-camera SC acquisition can produce real SC
  score without `_DEV_TARGETS` or public-sample world targets
- `S2` proved that learned legal SC acquisition can beat the best hand-crafted
  legal baseline without sacrificing the strong SFP path

What is not true yet:

- the current legal SC path does not get close enough to reliably convert the
  SC trial into insertion
- the best legal SC run still ends around `0.13 m` from the target port on the
  representative run

Therefore the next work item is not more public-sample tuning.
The next work item is to keep `S2` as the legal baseline and improve only the
SC pre-insertion alignment and final approach.

### Rejected Probe: X1 Tool-Frame Force Search

One small legal follow-up was tested after `S2`:

- `submission_safe_v8` added a bounded tool-frame force search after the
  learned SC acquisition
- in SC-only fail-fast mode it improved the isolated SC score to `33.40`
  (`/home/masa/aic_results/qual_submission_safe_v8_20260321_222558/scoring.yaml`)
- however, the corresponding full-run probe regressed the second SFP trial to
  `41.90`, so the branch was not promoted

Current conclusion:

- keep `S2` as the winning baseline
- do not spend more time on `v8` unless its time cost and cross-trial coupling
  are explained first

### Submission-Safe Track

The development-oriented milestones `M0` through `M7` were useful because they
made the controller phases and failure modes visible.
However, the submission target must now be tracked separately.

#### S0. First End-To-End Submission-Safe Baseline

Purpose:

- prove that the entire 3-trial run can complete using only `Task` and
  `Observation` at runtime
- preserve the strong SFP behavior while removing all dev/public target
  dependencies
- establish the first trustworthy legal baseline for further work

Representative result:

- stage: `submission_safe_v0`
- score total: `98.041744964269498`
- score by trial:
  - `trial_1 = 48.061318260132214`
  - `trial_2 = 48.980426704591785`
  - `trial_3 = 1.0`
- artifacts:
  - `/home/masa/aic_results/qual_submission_safe_v0_20260321_151750/scoring.yaml`
  - `/home/masa/ws_aic_runtime/qualification_debug/20260321_151944_submission_safe_v0_task_1`
  - `/home/masa/ws_aic_runtime/qualification_debug/20260321_152410_submission_safe_v0_task_1`
  - `/home/masa/ws_aic_runtime/qualification_debug/20260321_152833_submission_safe_v0_task_1`

What it uses:

- center-camera SFP servo driven only by image-space features from the current
  observation
- center-camera SC servo driven only by current observation masks
- short tool-axis push derived from the current TCP orientation
- bounded SC force refine and visual residual, but without `_DEV_TARGETS`,
  `PublicTrialPosePilot._TARGETS`, or replay trajectories

What it proved:

- the first two SFP trials do not need public-sample world targets to score
  near `49` each
- the legal-only path can complete all trials and generate debug artifacts
  cleanly
- the dominant remaining bottleneck is isolated to SC acquisition, not SFP

What still fails:

- the SC path still scores only Tier 1 and finishes at `0.20 m`
- the SC residual logic is now downstream of the real problem; it needs a
  better legal pre-insertion pose

Next gate:

- implement a better legal SC acquisition stage while preserving the current
  `S0` SFP score band

#### S1. Submission-Safe Triangulated SC Translation-Only Acquisition

Purpose:

- keep the legal SFP path from `S0`
- replace the weak SC center-camera acquisition with a multi-camera legal
  triangulation step
- isolate whether SC optical-axis rotation correction is helping or hurting

Representative result:

- stage: `submission_safe_v4`
- score total: `122.08548930859087`
- score by trial:
  - `trial_1 = 48.72334879624358`
  - `trial_2 = 48.987096754294114`
  - `trial_3 = 24.375043758053183`
- artifacts:
  - `/home/masa/aic_results/qual_submission_safe_v4_20260321_181109/scoring.yaml`
  - `/home/masa/ws_aic_runtime/qualification_debug/20260321_181301_submission_safe_v4_task_1`
  - `/home/masa/ws_aic_runtime/qualification_debug/20260321_181719_submission_safe_v4_task_1`
  - `/home/masa/ws_aic_runtime/qualification_debug/20260321_182141_submission_safe_v4_task_1`

What it uses:

- the same legal SFP center-camera servo as `S0`
- SC cyan-feature triangulation from the left / center / right wrist cameras
- bounded translation-only SC acquisition with optical-axis rotation disabled
- bounded SC force refine and visual residual on top of the new legal
  pre-insertion pose

What it proved:

- the legal-only track can now beat the development-oriented `M7` result
  without reintroducing `_DEV_TARGETS` or public-sample world targets
- disabling the SC optical-axis rotation heuristic is better than the coupled
  `v3` rotation path for this run family
- the SC trial can score materially (`24.38`) even without insertion when the
  pre-insertion pose is improved

What still fails:

- SC still does not insert
- the representative SC trial still ends at `0.16 m`, so the legal
  pre-insertion pose is better but not yet tight enough
- the SC runtime is still longer than desired (`19.00 s`) because the force and
  residual phases are doing corrective work that should move upstream

Next gate:

- preserve the `S1` SFP scores while tightening SC final distance below the
  current `0.16 m`
- improve SC pre-insertion alignment before adding more downstream force logic

### M0. Observability Harness

Purpose:

- verify that runs complete
- verify that images, phase transitions, and controller summaries are saved

Concrete deliverables:

- `QualPhasePilot` skeleton with explicit phases
- per-trial debug directory under a writable runtime path
- initial/final image montage, metadata JSON, and summary JSON

Validation gate:

- one full 3-trial run completes without policy crash
- artifacts exist for all three trials
- failure timing is visible from saved phase logs

Commit gate:

- commit as soon as the harness is stable, even if score is only Tier 1

### M1. Development-Only Controller Harness

Purpose:

- validate the controller state machine before legal perception is introduced

Concrete deliverables:

- add a `target_provider` abstraction to `QualPhasePilot`
- implement a development-only provider that returns coarse target poses for:
  - SFP `nic_card_mount_0`
  - SFP `nic_card_mount_1`
  - SC `sc_port_*`
- implement the controller phases end-to-end:
  - `acquire_target`
  - `approach`
  - `align`
  - `insert`
  - `recover`
- save target metadata, commanded pose summaries, and per-phase timing

Expected score profile:

- low but clearly above pure `M0`
- correctness of phase execution matters more than raw score

Validation gate:

- all phases execute in the intended order on representative SFP and SC trials
- pre-insertion pose is reached reproducibly
- obvious controller issues are fixed before perception is added

Commit gate:

- commit when controller behavior is predictable enough that future failures are
  likely to be target-localization errors rather than state-machine bugs

### M2. First Legal Baseline: SFP Center-Camera Localizer

Purpose:

- replace the development-only target source with the first submission-safe
  SFP target estimator

Concrete deliverables:

- use only the center camera for SFP
- implement a coarse localizer that estimates:
  - target pixel center
  - slot orientation cue
  - confidence
- convert the localizer output into a target-relative pre-insertion pose
- route SC tasks through a safe no-op or hold path so SFP can be debugged in
  isolation

Expected score profile:

- still modest
- success means the robot approaches the correct NIC for the right reason

Validation gate:

- on repeated SFP trials, the TCP moves toward the correct port family
- failures can be classified as detection, standoff, orientation, or insertion

Commit gate:

- commit when SFP targeting is visibly legal and repeatable, even if insertion
  is still weak

### M3. SFP Target-Relative Insertion Baseline

Purpose:

- turn the legal SFP localizer into a low-risk, score-producing baseline

Concrete deliverables:

- conservative pre-insertion standoff
- slow axial insertion
- one bounded retreat and retry
- explicit force and duration logging for the insertion attempt

Expected score profile:

- moderate SFP score with limited variance

Validation gate:

- repeated SFP runs show a consistent approach pattern
- Tier 2 is preserved by avoiding aggressive pushing
- dominant failures are local alignment misses, not gross targeting mistakes

Commit gate:

- commit when this becomes the first trustworthy submission-safe SFP baseline

### M4. SC Legal Baseline

Purpose:

- extend the same legal target-local stack to SC without changing the overall
  policy architecture

Concrete deliverables:

- add SC-specific localizer logic
- add SC pre-insertion geometry and alignment behavior
- keep the same phase machine and debug outputs used for SFP

Expected score profile:

- total score rises, but SC may still trail SFP

Validation gate:

- SC reaches a meaningful pre-insertion pose
- failure modes are understandable and mostly near-contact
- SFP behavior does not regress while SC support is added

Commit gate:

- commit when both SFP and SC run through the same debuggable legal pipeline

### M5. Multi-Camera Late Fusion

Purpose:

- improve robustness while preserving debuggability

Concrete deliverables:

- keep center-only logic as the fallback
- use left/right cameras only for:
  - confidence reweighting
  - lateral disambiguation
  - optional depth refinement

Expected score profile:

- moderate improvement in reliability, not necessarily a dramatic jump

Validation gate:

- fusion helps more often than it hurts
- when fusion is uncertain, the controller falls back cleanly

Commit gate:

- commit only if fusion measurably improves repeatability over center-only

### M6. Force-Guided Final Insertion And Recovery

Purpose:

- convert near-miss approaches into more stable insertions

Concrete deliverables:

- contact-triggered slow insertion mode
- bounded lateral micro-search near contact
- explicit retreat conditions
- per-attempt force, time, and retry caps

Expected score profile:

- this is the first milestone where score should rise meaningfully for both SFP
  and SC

Validation gate:

- fewer failures stop at "close but not inserted"
- contact behavior is smoother, not more chaotic
- the recovery outcome is legible from saved artifacts

Commit gate:

- commit when force-guided refinement improves insertion quality without
  collapsing Tier 2 through excessive contact

### M7. Learned Residuals Near Contact

Purpose:

- add learning only after the deterministic stack is measurable and stable

Concrete deliverables:

- fixed deterministic perception and state machine
- learned residual only for:
  - final lateral correction
  - yaw correction
  - insertion-axis correction

Expected score profile:

- potential high-score milestone, but only after M6 is trustworthy

Validation gate:

- residual improves a stable baseline
- ablations show the residual is adding value instead of hiding regressions

Commit gate:

- commit only if the learned residual reliably beats the non-residual M6 system

## Run And Commit Protocol

Each milestone should be executed with the same operational discipline so that
results are comparable.

### Before Every Validation Run

- rebuild the modified package:
  - `cd /home/masa/ws_aic/src/aic && pixi reinstall ros-kilted-aic-example-policies`
- restart the eval container if previous processes are still alive
- launch eval with an explicit display export:
  - `docker exec -i aic_eval bash -lc 'export DISPLAY=:1; /entrypoint.sh ground_truth:=false start_aic_engine:=true shutdown_on_aic_engine_exit:=true launch_rviz:=false gazebo_gui:=false'`
- launch the model with an explicit display export as well, even if the run is
  headless, to keep the command path consistent:
  - `cd /home/masa/ws_aic/src/aic && export DISPLAY=:1 && export RMW_IMPLEMENTATION=rmw_zenoh_cpp && export AIC_QUAL_STAGE=... && pixi run ros2 run aic_model aic_model --ros-args -p use_sim_time:=true -p policy:=aic_example_policies.ros.QualPhasePilot`

### After Every Validation Run

- record:
  - total score
  - per-trial scores
  - representative artifact paths
  - dominant failure mode
- add one short entry to `qualification_experiment_log.md`
- make one milestone commit with the same score summary in the commit body

### Non-Negotiable Rule For This Work

Do not jump directly from `M0` to `M4` or `M6`.
If a milestone exposes a failure, fix that failure before moving on.
The goal is a debuggable path to SC and force-guided refinement, not a pile of
partially overlapping ideas.

## Commit And Experiment Discipline

Commits should help debugging, not just checkpoint code.

### What To Commit

- commit at the end of each milestone
- also commit when a failed direction teaches something important
- do not commit every tiny parameter tweak unless it changes a conclusion

### What Each Commit Should Record

Each milestone commit should record:

- milestone name
- whether it is submission-safe or development-only
- best score observed for that milestone
- artifact path for the representative run
- a short note about the dominant failure mode

Recommended commit message style:

- `qual:M0 add run observability harness`
- `qual:M2 center-camera sfp heatmap baseline`
- `qual:M6 add force-guided insertion retry`

Recommended commit body fields:

- `Submission-safe: yes/no`
- `Score total: ...`
- `Score by trial: ...`
- `Artifacts: /home/masa/aic_results/...`
- `Failure mode: ...`

### Experiment Logging

In addition to commit messages, maintain a lightweight experiment log in the repo.
This log should contain:

- date and time
- current commit hash
- milestone
- run command or config
- score summary
- artifact paths
- 3 to 5 bullet observations
- next action

## Success Criteria

A strategy should be considered promising only if it:

- uses only submission-safe runtime inputs
- remains functional under randomized board and component placement
- works for both SFP and SC with the same policy framework
- improves score without relying on public-sample overfitting

## Pivot Plan If `center_uvz` Stalls

If the `center_uvz`-only route fails to beat the best legal baseline, do not
keep tuning it with small parameter changes. The next mainline should switch to
teacher-guided insertion.

### Why This Pivot Exists

The current `center_uvz` path only improves target acquisition. It does not
directly represent:

- insertion axis
- plug roll about the insertion axis
- near-contact correction conditioned on wrench and image evidence
- recovery when the plug drifts or loses the mouth of the port

That limitation makes it hard to cross the scoring cliff from `Tier 3 proximity`
to `Tier 3 insertion success`. A policy that predicts only a point in the image
is unlikely to deliver consistent `80+` scores per trial.

### Pivot Goal

The pivot target is not "a slightly better proximity score".

The pivot target is:

- `SFP >= 80` per trial
- `SC >= 80` per trial
- insertion-driven scores, not Tier 2 polishing

This means the new route must focus on producing reliable insertion events.

### Core Strategy

Use learning for the part that matters most:

- estimate a local insertion frame from official observations
- then run a teacher-guided closed-loop insertion policy in that local frame

The runtime stack should become:

1. coarse legal acquisition
2. local port-frame estimation
3. bounded pre-insertion motion
4. teacher-guided insertion near contact
5. recovery and retry if confidence or contact pattern is wrong

This keeps the system debuggable while moving the learned part to the place
where the score cliff actually is.

### What The New Model Should Predict

The new learned representation should not stop at `center_uvz`.

It should predict a task-conditioned local insertion frame:

- a point on the port mouth or insertion axis
- insertion axis direction
- roll or a roll proxy for plug orientation
- confidence for whether near-contact insertion should start

Near contact, a second learned head or second model should predict bounded
teacher-guided local actions:

- lateral correction in the port-local frame
- insertion-axis correction
- small rotational correction
- optional backoff / retry trigger

### Data Collection Plan

Do not train this route on sparse target labels alone. Collect dense
teacher-guided insertion trajectories.

For each randomized trial, save:

- three wrist camera images
- task fields
- current TCP pose
- current wrench
- legal feature summary
- ground-truth port frame for offline labels only
- teacher hover pose
- teacher pre-insert pose
- teacher insert pose
- dense intermediate teacher waypoints near contact
- success / failure / recovery outcome

The dataset must include:

- successful insertions
- near-miss insertions
- recoverable bad approaches
- explicit negative examples where pushing should stop

### Teacher Definition

The teacher should be deterministic and stage-based, not an opaque policy.

The teacher controller should use ground-truth offline only to generate:

- stable hover
- aligned pre-insertion
- low-speed insertion with bounded force
- backoff and reacquire when insertion diverges

This has two advantages:

- it makes the training target interpretable
- it exposes whether the controller is capable of `80+` even with perfect state

If the teacher itself cannot score well on randomized trials, fix the teacher
before training the student.

### Stage Gates

The pivot should move through these gates in order.

#### T0. Teacher Feasibility

Goal:

- prove that a ground-truth teacher can score at an insertion level on
  randomized SFP

Gate:

- randomized `SFP` teacher trials must reach at least `150 / 200` combined
  before any student training is trusted

If this fails, the problem is controller design, not learning.

Current status on 2026-03-22:

- `center_uvz` rerun with the larger balanced model still failed the pre-pivot
  gate, scoring only `71.320068644735088 / 200` on `SFP-only`
- `teacher_feasibility_v0` was then introduced to test a runtime GT teacher on
  randomized `SFP`
- the first GT-teacher run scored `86.653740214899685 / 200`
- a second run with an added GT terminal-contact loop scored
  `86.562241559624113 / 200`
- a third run, `teacher_feasibility_v1`, switched to GT pre-insert followed by
  tool-frame force refinement and scored only `70.903064258712277 / 200`
- bag/debug analysis from the failed GT runs shows that the current teacher can
  get close, but either jams while commanding too-deep reference poses
  (`T0 v0/v1`) or becomes too conservative and abort-heavy when handed off to a
  pure force search too early (`T0 v2`)

Interpretation:

- `center_uvz-only` is no longer the mainline
- the current GT teacher also fails the `150 / 200` feasibility gate
- therefore the next bottleneck is still teacher/controller design, not student
  learning
- the next T0 redesign should keep the stronger GT approach from `T0 v0` longer
  and add progress-gated recovery based on observed advance / tracking error,
  rather than switching immediately to a pure force-search handoff

#### T1. Dense Teacher Dataset

Goal:

- collect a balanced randomized dataset with successful and failed insertion
  traces for both `SFP` and `SC`

Gate:

- enough data to cover all SFP rails repeatedly
- enough `SC` variation to prevent memorizing one public configuration

Practical first target:

- `SFP`: at least `100` successful teacher trials
- `SC`: at least `100` successful teacher trials

#### T2. Local Frame Estimator

Goal:

- replace `center_uvz` with a learned local insertion-frame estimator

Gate:

- online pre-insertion should beat the old `center_uvz` route on SFP
- if not, do not proceed to insertion learning

Expected online target:

- `SFP-only` score should clearly beat the `submission_safe_v11` bug-fixed
  result and approach or exceed the old `S2` SFP pair

#### T3. Teacher-Guided SFP Insertion

Goal:

- train a bounded local policy that starts only after pre-insertion is good

Gate:

- `SFP` should cross from proximity into repeated insertion events
- target combined score should exceed `160 / 200`

If this gate fails, the problem is likely the local action target or teacher
labels, not more image data.

#### T4. Teacher-Guided SC Insertion

Goal:

- carry the same framework to `SC`, with a plug-specific head if needed

Gate:

- randomized `SC` should first reach `50+`
- then push to `80+`

Do not mix `SC` into the mainline until `SFP` insertion is already strong.

#### T5. Unified Submission Policy

Goal:

- integrate SFP and SC under one legal runtime policy framework

Gate:

- all three trials should repeatedly exceed the old legal baseline
- at least one representative run should show the system is now insertion-led,
  not proximity-led

### Fail-Fast Conditions

Abort the `center_uvz` route as the mainline if any of these remain true after
the larger balanced dataset:

- `SFP-only` still cannot beat the old SFP pair baseline
- the model still helps acquisition but not insertion
- the learned route still collapses once push begins

Abort the teacher-guided student route if:

- the offline teacher cannot itself insert reliably
- student behavior stays at the teacher-hover level and never learns insertion
- success depends on one fixed public-looking geometry

### Immediate Next Actions For The Pivot

If the current `center_uvz` rerun still fails to beat the legal baseline, do
these next:

1. build a ground-truth teacher insertion controller for randomized `SFP`
2. collect dense successful insertion traces, not just target labels
3. train a local insertion-frame estimator
4. train a bounded teacher-guided insertion policy that only activates after
   pre-insertion
5. require `SFP >= 160 / 200` combined before starting serious `SC` work

This is the first route in this repo that has a realistic path to repeated
`80+` trial scores.

### Immediate Outcome Of The First Pivot Loop

The first pivot loop completed and gave a clear answer:

- `center_uvz-only` failed its gate
- the current GT teacher also failed its gate

So the next PDCA cycle should not collect more student data yet. It should
instead redesign the GT teacher so that near-contact motion is:

- lower-jerk
- axis-aware
- explicitly recovery-capable
- demonstrably insertion-led before any student retraining begins
