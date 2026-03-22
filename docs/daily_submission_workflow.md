# Daily Submission Workflow

## Purpose

This document is the repo-specific operating procedure for making one
qualification submission per day.

The official generic registry instructions live in
[submission.md](/home/masa/ws_aic/src/aic/docs/submission.md).
This file is narrower and more practical:

- how to cut a submission candidate from ongoing experimentation
- how to freeze a reproducible source state
- how to package repo-external artifacts such as learned models
- how to verify locally before spending the daily submission slot
- how to record what was actually submitted

The goal is to make the daily workflow boring, repeatable, and auditable.

## Design Principles

The workflow should optimize for the following:

- a submission candidate must come from a clean, reviewable source state
- repo-external runtime dependencies must be copied into the repo or image
  explicitly
- local container verification is mandatory before push
- the submitted image URI must be traceable back to a git branch, commit, and
  local score
- research code and submission packaging should not share the same dirty
  working tree

## Recommended Branch Strategy

Use three logical states.

### 1. Main Research Branch

This is where ongoing experimentation happens.

- branch: `main`
- allowed to be noisy
- may contain unfinished ideas
- may depend on local-only paths during exploration

Do **not** submit directly from this state.

### 2. Submission Candidate Branch

For each daily submission, cut a dedicated branch from the commit you want to
package.

Suggested naming:

```bash
submit/YYYYMMDD-<stage>
```

Examples:

```bash
submit/20260322-submission-safe-v6
submit/20260323-submission-safe-v12
```

This branch should contain only what is required to reproduce the intended
submission.

### 3. Optional Dedicated Worktree

A separate worktree is strongly recommended, because it avoids mixing
submission packaging edits with research changes.

Example:

```bash
git -C /home/masa/ws_aic/src/aic worktree add \
  /home/masa/ws_aic_submit_20260322 \
  -b submit/20260322-submission-safe-v6 \
  HEAD
```

With this setup:

- `/home/masa/ws_aic/src/aic` stays available for research
- `/home/masa/ws_aic_submit_20260322` becomes the clean submission packaging
  area

If daily submissions become routine, this should be the default.

## Daily Submission Loop

The workflow below is the recommended default.

### Step 1. Pick the Candidate

Before touching Docker or ECR, decide exactly what is being submitted.

Required checklist:

- the run must be `Submission-safe: yes`
- the intended stage must be explicit
- the representative local score must be known
- the artifact path must be recorded
- the source state must be identifiable

For example, at the time of writing:

- best clean reproducible submission-safe run: `submission_safe_v7`
- score: `126.815412 / 300`
- evidence:
  [eval.log](/home/masa/aic_results/qual_submission_safe_v7_20260323_033038/eval.log)
- source branch: `submit/20260322-submission-safe-v6`
- source commit: `bc3e493d4e306cb4f4c95ef3c8b2216e0eb2d708`
- pushed image URI:
  `973918476471.dkr.ecr.us-east-1.amazonaws.com/aic-team/masapon:submission-safe-v7-bc3e493-20260323-040153`

If the candidate is not yet reproducible from source, stop here and fix that
first.

### Step 2. Freeze the Source State

Create the submission candidate branch from the source state you intend to
package.

Preferred flow:

```bash
cd /home/masa/ws_aic/src/aic
git switch -c submit/YYYYMMDD-<stage>
```

If the score came from an uncommitted run, then:

1. identify the minimal code and artifact state needed to reproduce it
2. commit only that state
3. verify the worktree is clean

Do not package from a dirty working tree.

### Step 3. Vendor Non-Repo Runtime Artifacts

This is the most common place where "works locally" diverges from
"works in the container".

Any runtime dependency outside the repo must be made explicit.

Examples:

- learned model weights under `/home/masa/ws_aic_runtime/learned_port_models/...`
- manifests used by learned inference code
- custom config files referenced via environment variables

For a submission candidate, do **not** rely on:

- `/home/masa/ws_aic_runtime/...`
- `/tmp/...`
- manually exported shell variables on one machine only

Instead, move or copy the necessary files under the repo, for example:

```text
docker/model_assets/sc_uvz_v1/
docker/model_assets/sfp_centeruvz_.../
```

Then update the Docker build so those assets are copied into the image.

### Step 4. Create a Dedicated Submission Packaging Layer

Avoid mutating the generic development Docker setup more than necessary.

A good pattern is:

- keep the generic
  [Dockerfile](/home/masa/ws_aic/src/aic/docker/aic_model/Dockerfile)
  as the base reference
- add a submission-specific Dockerfile or compose override for the current
  candidate

Recommended files:

- `docker/submission/<stage>/Dockerfile`
- `docker/submission/<stage>/docker-compose.yaml`

This keeps daily packaging auditable and avoids silently breaking the normal
local workflow.

### Step 5. Make the Runtime Contract Explicit

The image must fully define:

- which ROS policy is launched
- which `AIC_QUAL_STAGE` is used
- where learned model artifacts live inside the container

For `submission_safe_v7`, that means at minimum:

- policy: `aic_example_policies.ros.QualPhasePilot`
- stage: `submission_safe_v7`
- `AIC_QUAL_LEARNED_SC_MODEL_DIR` must be valid inside the container

If any of those still depend on ad hoc shell setup, the candidate is not ready.

### Step 6. Rebuild the Changed Packages

If policy code changed before packaging, rebuild first.

Current workspace rule:

```bash
cd /home/masa/ws_aic/src/aic
pixi reinstall ros-kilted-aic-example-policies
```

This is a local verification step, not the final container build step.

### Step 7. Reproduce the Candidate Locally

Before building the submission image, re-run the candidate locally from the
submission branch or worktree.

This step should answer:

- does the intended stage still run
- do the required learned assets resolve correctly
- does the score remain near the representative score

For qualification candidates, the practical bar should be:

- all 3 trials complete
- no policy crash
- score is close enough to the reference run that the candidate is credible

It is not necessary to reproduce the exact same floating-point total, but a
large regression should block submission.

### Step 8. Build the Submission Image

Use the official procedure in [submission.md](/home/masa/ws_aic/src/aic/docs/submission.md),
but from the frozen submission branch.

Generic command:

```bash
cd /home/masa/ws_aic/src/aic
docker compose -f docker/docker-compose.yaml build model
```

If using a dedicated submission compose file, use that instead.

### Step 9. Verify the Container Locally

This is mandatory.

Run the evaluation locally with the actual submission image.

Goals:

- the container starts cleanly
- the model process launches the intended policy
- the stage and learned model paths are correct
- the score is not catastrophically worse than the non-container run

If local container verification fails because of a host-specific GPU runtime
issue, separate that from repo-side validity. On this machine, for example, the
container start was blocked by:

```text
/run/nvidia-persistenced/socket: no such file or directory
```

In that case, still verify:

- the image builds successfully
- the image contains the expected learned assets
- the image environment points to the intended stage and model directory

### Step 10. Tag the Image With Immutable Metadata

Use a tag that encodes both date and commit.

Recommended format:

```text
YYYYMMDD-<stage>-<gitsha>
```

Example:

```text
submission-safe-v7-bc3e493-20260323-040153
```

This makes it easy to trace a portal submission back to the source state.

### Step 11. Push to ECR

Follow [submission.md](/home/masa/ws_aic/src/aic/docs/submission.md).

High-level flow:

```bash
aws configure --profile <team_slug>
export AWS_PROFILE=<team_slug>
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 973918476471.dkr.ecr.us-east-1.amazonaws.com
docker tag <local-image> 973918476471.dkr.ecr.us-east-1.amazonaws.com/aic-team/<team_slug>:<tag>
docker push 973918476471.dkr.ecr.us-east-1.amazonaws.com/aic-team/<team_slug>:<tag>
```

Do not reuse an existing tag. The registry is immutable.

### Step 12. Register in the Portal

Pushing to ECR is not enough.

You still need to register the image URI in the submission portal as described
in [submission.md](/home/masa/ws_aic/src/aic/docs/submission.md).

### Step 13. Record the Submission

After pushing and registering, write down exactly what was sent.

At minimum record:

- date
- submission branch
- git commit
- local representative score
- image URI
- portal submission time
- eventual official result

This can live in:

- `docs/qualification_experiment_log.md`
- or a dedicated future file such as `docs/submission_log.md`

Without this step, it becomes difficult to learn from the one-submission-per-day
constraint.

## Definition of Submission-Ready

A candidate is submission-ready only if all of the following are true.

- source is on a dedicated submission branch or worktree
- worktree is clean
- runtime assets are inside the repo or copied into the image explicitly
- local non-container score is known
- local container verification passes
- image tag is unique and traceable
- the exact image URI to be pasted into the portal is recorded

## Recommended Daily Checklist

Use this list every day.

```md
- [ ] Today’s candidate stage is decided
- [ ] Candidate score and artifact path are recorded
- [ ] Candidate branch/worktree is created
- [ ] Worktree is clean
- [ ] Runtime artifacts are vendored into the repo/image
- [ ] Policy/stage/model-dir contract is explicit in Docker config
- [ ] Local non-container run is verified
- [ ] Local container run is verified
- [ ] Image is tagged with date + stage + git sha
- [ ] Image is pushed to ECR
- [ ] Image URI is registered in the portal
- [ ] Submission metadata is logged
```

## Current Known Good Example

The current known-good reproducible `120+` submission-safe candidate is:

- stage: `submission_safe_v7`
- branch: `submit/20260322-submission-safe-v6`
- source commit: `bc3e493d4e306cb4f4c95ef3c8b2216e0eb2d708`
- score: `126.815412 / 300`
- evidence:
  [eval.log](/home/masa/aic_results/qual_submission_safe_v7_20260323_033038/eval.log)
- pushed image URI:
  `973918476471.dkr.ecr.us-east-1.amazonaws.com/aic-team/masapon:submission-safe-v7-bc3e493-20260323-040153`

The dedicated packaging worktree used for that candidate was:

- [/home/masa/ws_aic_submit_20260322_v6](/home/masa/ws_aic_submit_20260322_v6)

Use this candidate as the reference shape for:

- branch naming
- packaging layout
- score evidence capture
- ECR tag naming

## Suggested Next Automation

Once this manual flow works once, the next improvements should be:

- `scripts/prepare_submission_candidate.sh`
- `scripts/build_submission_image.sh`
- `scripts/push_submission_image.sh`
- a dedicated `docs/submission_log.md`

Automation should come **after** one clean end-to-end manual submission path is
working and documented.
