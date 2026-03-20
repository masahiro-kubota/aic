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
