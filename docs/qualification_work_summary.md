# Qualification Work Summary

## Purpose

This document summarizes the qualification-phase work so far at a higher level
than the detailed experiment log.

The goal of this summary is to make it easy to answer the following questions:

- what has been tried so far
- which runs improved the score and which did not
- what conclusions were actually supported by experiments
- what the current blocker is
- what should happen next

This file is intentionally redundant with
[qualification_experiment_log.md](/home/masa/ws_aic/src/aic/docs/qualification_experiment_log.md)
and
[qualification_strategy_notes.md](/home/masa/ws_aic/src/aic/docs/qualification_strategy_notes.md).
The experiment log is the raw chronological record. The strategy notes are the
design and planning document. This file is the narrative bridge between them.

## Current Bottom Line

The current state is:

- the best submission-safe score is still `126.58206055565613 / 300`
- that best run is `S2`, stored at
  `/home/masa/aic_results/qual_submission_safe_v6_20260321_220649/scoring.yaml`
- `SFP` is reasonably strong and repeatedly lands in the high-40-point band
- `SC` is still the main bottleneck
- `center_uvz-only` has now failed as the mainline strategy
- the first ground-truth teacher feasibility loop also failed its gate

The current project conclusion is therefore:

- do not keep tuning `center_uvz-only`
- do not collect more student data yet
- redesign the teacher/controller first
- only resume teacher-guided student learning after the teacher itself becomes
  insertion-led

## Important Operational Facts

The following environment facts turned out to matter in practice:

- GUI-capable runs should explicitly set `DISPLAY=:1`
- changing packages under `src/aic` requires rebuilding before validating
- `distrobox enter -r aic_eval` was awkward in this environment because it
  internally tried to use interactive `sudo`
- `docker exec` was a workable replacement, and the docs were adjusted in that
  direction

These are also reflected in local workflow notes and docs updates.

## What Was Verified Early

Before serious policy iteration, several environment and scoring issues were
checked:

1. The end-to-end pipeline from docs could run.
2. GUI launch via `DISPLAY=:1` worked.
3. The earlier `Tier 3 tf between cable and port not found` issue was traced to
   stale evaluation artifacts rather than only to source state.
4. Rebuilding the relevant packages in the evaluation path was necessary for
   some upstream fixes to actually take effect.

This mattered because it separated environment problems from actual policy
performance problems.

## Scoring Reality

One important clarification that shaped the later plan:

- scoring is `100` per trial, not `100` total
- qualification sample runs discussed here usually have `3` trials, so the
  meaningful total scale is `300`
- a score around `120` is therefore not "pretty good"; it is still far from the
  `80+` per-trial regime needed for a serious submission

That realization is what pushed the work away from "small improvements" and
toward "cross the insertion cliff".

## High-Level Timeline

The work naturally split into several phases:

1. Public-sample experimentation and development scaffolding
2. First submission-safe baseline
3. Better legal `SC` acquisition
4. Learned acquisition experiments
5. Pivot planning after `center_uvz` stagnation
6. First teacher-feasibility experiments

Each phase is described below.

## Phase 1: Public-Sample And Development Scaffolding

### Public-sample replay reference

Early on, a strong public-sample-oriented replay policy was built as a reference.

- Score: `202.68080794003197`
- This was useful as an upper reference for "what a tuned public-sample-specific
  policy can do locally"
- This was not considered submission-safe

The key value of this phase was not the submission value of the score, but the
fact that it showed the environment was capable of much higher local scores
than the initial legal baselines.

### M0 through M7

The `M0-M7` sequence was the development-oriented milestone ladder.

The main point of `M0-M7` was to build observability and to test ideas in a
controlled way before insisting on submission safety.

Representative scores:

| Milestone | Purpose | Score |
| --- | --- | ---: |
| `M0` | observability harness | `3` |
| `M1` | development target controller harness | `100.11421244878404` |
| `M2` | first SFP center-camera localizer | `3` |
| `M3` | first task-conditioned SFP template tuning | `-21` |
| `M4` | public-sample full pipeline baseline | `93.084972039114518` |
| `M5` | multi-camera late fusion | `109.96488164803876` |
| `M6` | SC force-guided refinement | `110.0571545895495` |
| `M7` | residual refinement with fresh-observation gating | `111.43384153152546` |

Main takeaways from `M0-M7`:

- observability and debug artifacts were worth the effort
- stale observation handling and sim-time pacing mattered
- `SFP` could be pushed up fairly reliably
- `SC` kept resisting insertion
- development-only paths were useful for diagnosis but not for final policy

The final conclusion of this phase was that development scaffolding was useful,
but the project still needed a real submission-safe track.

## Phase 2: First Submission-Safe Baselines

### S0: first legal-only end-to-end run

`S0` was the first run treated as a real submission-safe baseline.

- Score: `98.041744964269498`
- By trial:
  - `t1=48.061318260132214`
  - `t2=48.980426704591785`
  - `t3=1.0`

What `S0` proved:

- it was possible to complete all 3 trials using only legal runtime inputs
- the legal `SFP` path was already fairly stable
- `SC` was clearly the weak point

This was the first major turning point. The problem was no longer "can we build
a legal pipeline?" but "why does legal `SC` acquisition still fail so badly?"

### S1: triangulated SC translation-only acquisition

`S1` improved legal `SC` acquisition with a triangulated translation-only stage.

- Score: `122.08548930859087`
- By trial:
  - `t1=48.72334879624358`
  - `t2=48.987096754294114`
  - `t3=24.375043758053183`

What `S1` proved:

- legal `SC` acquisition could contribute meaningful score
- some earlier `SC` orientation heuristics were making things worse
- even without insertion, better legal acquisition could move `SC` far beyond
  Tier 1 only

This established that the legal framework itself was not the problem.

### S2: learned SC acquisition

`S2` became the current best submission-safe result.

- Score: `126.58206055565613`
- By trial:
  - `t1=48.21030515038397`
  - `t2=48.982548911699574`
  - `t3=29.389206493572582`

What `S2` proved:

- learned legal `SC` acquisition can beat the best hand-crafted legal version
- the legal path is not stuck at `~100`
- but the whole system is still proximity-led rather than insertion-led

This is the current official best submission-safe reference.

## Phase 3: Small Legal Follow-Ups That Did Not Change The Main Picture

### X1: tool-frame force search probe

An isolated `SC` tool-frame force search was tested.

- Score: `33.404680405165706` on the isolated probe

What it showed:

- the force search idea was not useless
- but it was too slow and too brittle to promote immediately
- it did not justify replacing `S2`

The lesson here was that local force search alone was not the missing piece.

## Phase 4: `center_uvz` Learned SFP Route

After the legal `SC` acquisition work, the next large branch focused on learned
`SFP` acquisition, particularly a `center_uvz` target representation.

### Data collection

A larger randomized `SFP` dataset was collected:

- `20` randomized trials
- `840` total samples
- balanced across all five `SFP` rails
- phases included initial, teacher hover, and teacher insert sweeps

This was meant to test whether `center_uvz` could become a serious legal
mainline.

### Training

A larger `center_uvz` model was trained on that dataset.

- training ran to `160` epochs
- best checkpoint was at `epoch 152`
- best validation loss was `0.006016482987130682`
- best validation metrics were in the rough range of:
  - `u ~= 4.18 px`
  - `v ~= 11.59 px`
  - `depth ~= 2.42 mm`

This was a nontrivial improvement on paper.

### Runtime bug and fix

An important runtime bug was found:

- the learned model expected runtime auxiliary features
- runtime inference was not actually feeding them
- this caused nonsense predictions or no motion

That bug was fixed, and the learned branch was re-evaluated.

### P0 gate result

Even after retraining and the runtime bugfix, the larger `center_uvz` route
failed the mainline gate:

- Score: `71.320068644735088 / 200` on `SFP-only`
- By trial:
  - `t1=35.220362710871615`
  - `t2=36.099705933863473`

Why this mattered:

- this was well below the older `SFP` pair baseline from `S2`
- the model improved acquisition enough to move, but not enough to cross into
  insertion-driven scoring
- this is why `center_uvz-only` is now considered failed as the mainline route

This was the second major turning point.

## Phase 5: Pivot To Teacher-Guided Insertion

Once `center_uvz-only` failed, the strategy pivoted away from "predict a point
better" and toward "make insertion itself learnable".

The core pivot idea was:

- use learning to estimate a local insertion frame
- then use a teacher-guided near-contact policy in that frame

The important conceptual shift was:

- the problem is not only acquisition
- the real bottleneck is crossing from proximity to insertion

This pivot was written explicitly into
[qualification_strategy_notes.md](/home/masa/ws_aic/src/aic/docs/qualification_strategy_notes.md)
under the `Pivot Plan If center_uvz Stalls` section.

## Phase 6: First Teacher Feasibility Loop

The pivot plan started with `T0`: before training a student, check whether a
ground-truth teacher can itself score well enough on randomized `SFP`.

This was deliberate. If the teacher cannot insert reliably, collecting more
student labels is premature.

### T0 v0

The first GT teacher feasibility run used a randomized `SFP-only` config.

- Score: `86.653740214899685 / 200`
- By trial:
  - `t1=41.779318087841951`
  - `t2=44.874422127057734`

What this showed:

- the teacher could complete both tasks
- the teacher could get very near the mouth of the port
- but it still did not produce insertion events

So even with GT, the controller was still proximity-led.

### T0 v1

A second teacher feasibility run added a GT terminal-contact loop before the
final push.

- Score: `86.562241559624113 / 200`
- By trial:
  - `t1=42.659702234323316`
  - `t2=43.902539325300786`

This was effectively unchanged from `T0 v0`.

What this means:

- replacing the last fixed push with a slightly more GT-aware loop did not
  change the result enough
- the problem is not solved by a tiny local patch
- the teacher itself still needs a more fundamental redesign

This was the third major turning point.

## What The Current Evidence Actually Says

At this point, the evidence is fairly strong on several points.

### 1. Legal runtime inputs are not the limiting factor

`S0`, `S1`, and `S2` proved that legal runtime inputs can support nontrivial
performance.

### 2. `SFP` is not solved, but it is not the worst part

Repeated legal runs keep `SFP` around the high-40-point band. That is not good
enough, but it is materially stronger than the current `SC` path.

### 3. `SC` is still the main blocker on the submission-safe path

The best legal total is still capped by `SC` not reaching insertion.

### 4. `center_uvz-only` is not the right mainline

Even with better data and a fixed runtime bug, it failed the `SFP-only` gate.

### 5. The current GT teacher is also not good enough

This is the most important recent conclusion.

The current GT teacher:

- reaches near-contact
- completes tasks
- but does not insert

That means the next bottleneck is teacher/controller design, not student model
capacity.

## What Has Been Inspected And What Still Needs More Work

### What has already been inspected

The work so far has used:

- scoring outputs
- debug snapshots under `/home/masa/ws_aic_runtime/qualification_debug`
- per-run logs
- bag existence and bag metadata

For the latest `T0` runs, bag metadata confirms the presence of useful signals:

- `/aic_controller/pose_commands`
- `/aic_controller/controller_state`
- `/fts_broadcaster/wrench`
- `/tf`
- `/scoring/tf`

Example bag metadata:

- `trial_1` bag duration: `60.869616868 s`
- `trial_1` pose commands: `213`
- `trial_1` controller_state messages: `1098`
- `trial_1` wrench messages: `109`
- `trial_2` bag duration: `76.147619956 s`
- `trial_2` pose commands: `426`
- `trial_2` controller_state messages: `2215`
- `trial_2` wrench messages: `222`

### What still needs deeper work

The next loop should be more explicitly bag-first than some of the recent
iterations.

In particular, the next controller redesign should be based on:

- the actual relationship between pose commands and realized motion near contact
- where the wrench rises and whether that rise corresponds to useful insertion
  or useless collision
- whether the end-effector advances along the intended insertion axis after
  contact
- whether the final teacher phases are too jerky or too misaligned

This is a real next-step requirement, not a vague "maybe inspect later".

## Why More Student Data Is Not The Immediate Answer

It would be easy to keep collecting more training data because the infrastructure
for it already exists. That is not the current priority.

The current reasoning is:

1. `center_uvz-only` already received a serious data increase and still failed
   its gate.
2. The GT teacher also failed its own gate.
3. Therefore, collecting more student labels right now is unlikely to address
   the real bottleneck.

So the current data policy is:

- do not prioritize more point-label acquisition right now
- only resume large student-data collection after the GT teacher becomes
  insertion-led
- once that happens, collect dense teacher-guided insertion traces rather than
  only sparse target labels

## What The Next PDCA Loop Should Focus On

The next loop should not be "try another small tweak".

It should be:

1. analyze the latest `T0` bags and debug artifacts at the near-contact phase
2. redesign the GT teacher to be:
   - lower-jerk
   - axis-aware
   - recovery-capable
   - explicitly insertion-oriented
3. rerun randomized `SFP-only` teacher feasibility
4. require `>= 150 / 200` before resuming teacher-guided student work

Only after that gate is passed should the project move to:

- dense successful teacher insertion data collection
- local insertion-frame student learning
- bounded teacher-guided insertion policy training

## Score Table

The following table is the most useful high-level score summary so far.

| Label | Meaning | Score | Notes |
| --- | --- | ---: | --- |
| public replay ref | public-sample-oriented reference | `202.68080794003197` | not submission-safe |
| `M0` | observability harness | `3` | debug-first scaffold |
| `M1` | dev target controller harness | `100.11421244878404` | development-only |
| `M2` | first SFP center localizer | `3` | no useful execution yet |
| `M3` | first task-conditioned SFP tuning | `-21` | regressed badly |
| `M4` | public-sample full-pipeline baseline | `93.084972039114518` | still development-oriented |
| `M5` | multi-camera late fusion | `109.96488164803876` | better, still dev-oriented |
| `M6` | SC force-guided refine | `110.0571545895495` | minimal gain |
| `M7` | residual refine with fresh observation gating | `111.43384153152546` | still not submission-safe |
| `S0` | first submission-safe baseline | `98.041744964269498` | legal-only pipeline exists |
| `S1` | triangulated legal SC acquisition | `122.08548930859087` | big legal improvement |
| `S2` | learned legal SC acquisition | `126.58206055565613` | current best submission-safe |
| `X1` | legal SC force-search probe | `33.404680405165706` | isolated SC probe |
| `P0` | larger balanced `center_uvz` gate | `71.320068644735088 / 200` | failed `SFP-only` mainline gate |
| `T0 v0` | randomized GT teacher feasibility | `86.653740214899685 / 200` | GT teacher still not insertion-led |
| `T0 v1` | GT teacher with terminal contact loop | `86.562241559624113 / 200` | no meaningful improvement |

## Current Decision

The current decision is:

- keep `S2` as the best submission-safe reference
- stop treating `center_uvz-only` as the mainline
- stop increasing student data until the teacher improves
- redesign the GT teacher before moving on

That is the most accurate summary of the work so far.
