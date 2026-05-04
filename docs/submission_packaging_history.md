# Submission Packaging History

## Short Answer

- The last evidence-backed submission packaging version is
  `submission_safe_v7`.
- The reproducible packaged candidate was recorded on 2026-03-23 with:
  - stage: `submission_safe_v7`
  - source branch: `submit/20260322-submission-safe-v6`
  - source commit: `bc3e493d4e306cb4f4c95ef3c8b2216e0eb2d708`
  - pushed image tag:
    `submission-safe-v7-bc3e493-20260323-040153`
- The packaging was later copied onto `main` as
  `83cf433d0bd0319011cdc42d246a11caf6174485`.
- The latest local reruns found on disk also still used `submission_safe_v7`
  on 2026-04-11, so the last used packaging version is still `v7`.

## Version Meanings

These three things are different and should not be conflated:

- `stage`: the runtime policy path selected by `AIC_QUAL_STAGE`, for example
  `submission_safe_v7` or `submission_safe_v17`
- `packaging`: the Dockerfile / compose / image tag used to containerize a
  submission candidate
- `branch`: the frozen source state used for packaging, which may not match the
  stage number

Example: the packaged `submission_safe_v7` candidate came from branch
`submit/20260322-submission-safe-v6`.

## Current Repo State

- Latest submission-safe stage implemented in code:
  `submission_safe_v17`
  in
  [QualPhasePilot.py](/home/masa/ws_aic/src/aic/aic_example_policies/aic_example_policies/ros/QualPhasePilot.py:7983).
- Latest submission packaging present in repo:
  `docker/submission_safe_v7/Dockerfile`
  and
  `docker/docker-compose.submission_safe_v7.yaml`.
- No `submission_safe_v8+` Docker packaging files were found in `docker/`.

## Evidence Timeline

| Date | Evidence | Meaning |
| --- | --- | --- |
| 2026-03-22 | local results under `/home/masa/aic_results/qual_submission_safe_v12_*` through `..._v17_*` | experimentation continued past `v7`, but this alone does not mean those stages were packaged |
| 2026-03-23 | commit `f933552` | docs recorded `submission_safe_v7` as the reproducible submission-safe baseline |
| 2026-03-23 | commit `bc3e493` | added the actual `submission_safe_v7` Docker packaging and vendored model asset |
| 2026-03-23 | docs + workflow mention pushed image `submission-safe-v7-bc3e493-20260323-040153` | confirms `v7` was the packaged submission candidate |
| 2026-04-11 | commit `83cf433` on `main` | replayed the same `submission_safe_v7` packaging onto `main` |
| 2026-04-11 | local results `/home/masa/aic_results/qual_submission_safe_v7_local_image_20260411_0809` and `/home/masa/aic_results/qual_submission_safe_v7_20260411_160438` | latest observed local use also still targeted `submission_safe_v7` |

## Notes

- The branch name example `submit/20260323-submission-safe-v12` appears in
  [daily_submission_workflow.md](/home/masa/ws_aic/src/aic/docs/daily_submission_workflow.md:66)
  only as a naming example. In this repo, it is not backed by a matching
  packaging commit, Dockerfile, compose file, or pushed image record.
- If you need to resume submission work from the last known packaged state,
  start from commit `bc3e493` or from the same packaging as replayed on `main`
  by `83cf433`, then decide whether you want to keep stage `v7` or package a
  newer stage explicitly.
