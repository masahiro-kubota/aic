# Seed-Reproducible Qualification Configs

This repository now includes a public-spec randomization generator for local qualification validation.

It does not attempt to match the private cloud generator bit-for-bit. The goal is narrower:

- keep randomization inside public rail limits and documented grasp noise
- make trial generation deterministic from a seed
- feed the generated YAML into the existing `aic_engine_config_file` flow

## What It Generates

`scripts/generate_randomized_qualification_config.py` writes a complete `aic_engine` config YAML with three trials:

- `trial_1`: SFP insertion
- `trial_2`: SFP insertion
- `trial_3`: SC insertion

The generator keeps the sample config structure and randomizes:

- task board `x`, `y`, `yaw`
- target NIC rail and target SFP port for trial 1/2
- extra NIC occupancy for trial 1/2
- NIC translation and yaw for trial 1/2
- target SC port and optional second SC port presence for trial 3
- SC translation for trial 3
- pick-side mount translations
- grasp pose noise around the sample config nominal pose

## Profile

The default profile is:

```text
aic_engine/config/qualification_randomization_public_v1.yaml
```

This profile is the source of truth for the public approximation. In particular:

- trial 1/2 board pose ranges are inherited from the existing local SFP randomization helper
- trial 3 board pose ranges are a conservative public approximation centered around `sample_config.yaml`
- rail translation limits still come from `task_board_limits` in the template config

## Usage

Generate a config:

```bash
cd /home/masa/ws_aic/src/aic
python3 scripts/generate_randomized_qualification_config.py \
  /home/masa/ws_aic_runtime/generated_configs/qual_seed91.yaml \
  --seed 91
```

Run it with the existing qualification helper:

```bash
cd /home/masa/ws_aic/src/aic
AIC_ENGINE_CONFIG_FILE=/home/masa/ws_aic_runtime/generated_configs/qual_seed91.yaml \
  scripts/run_qualification_stage.sh submission_safe_v7
```

Use a different profile:

```bash
python3 scripts/generate_randomized_qualification_config.py \
  /home/masa/ws_aic_runtime/generated_configs/qual_seed123.yaml \
  --seed 123 \
  --profile aic_engine/config/qualification_randomization_public_v1.yaml
```

## Reproducibility

The output YAML is deterministic for a fixed seed and profile.

- same seed + same profile -> byte-identical YAML
- different seed -> different sampled placements and metadata

Top-level and per-trial metadata are included so the generated scene can be audited later.

## Notes

- This generator does not modify `aic_engine`.
- The existing `scripts/generate_randomized_sfp_config.py` remains available for SFP-only workflows.
- `trial_3` uses `sc_port_0` / `sc_port_1` as the task target names, while the task board scene is still configured via `sc_rail_0` / `sc_rail_1`, matching the current engine contract.

## Validated Run

The following run was validated locally on `2026-05-04`.

### Commands

Generate the seed-based config:

```bash
cd /home/masa/ws_aic/src/aic
python3 scripts/generate_randomized_qualification_config.py \
  /home/masa/ws_aic_runtime/generated_configs/qual_seed91.yaml \
  --seed 91
```

Run the existing qualification helper with the generated config:

```bash
cd /home/masa/ws_aic/src/aic
DISPLAY=:1 \
AIC_QUAL_GAZEBO_GUI=false \
AIC_QUAL_LAUNCH_RVIZ=false \
AIC_QUAL_LEARNED_SC_MODEL_DIR=/home/masa/ws_aic/src/aic/docker/model_assets/sc_uvz_v1 \
AIC_ENGINE_CONFIG_FILE=/home/masa/ws_aic_runtime/generated_configs/qual_seed91.yaml \
scripts/run_qualification_stage.sh submission_safe_v7
```

### Result

The generated config completed all three trials:

- `Successful: 3 Failed: 0`
- `Total Score: 59.610579326329656`

Artifacts from that run were written to:

- result directory: `/home/masa/aic_results/qual_submission_safe_v7_20260504_233246`
- generated config: `/home/masa/ws_aic_runtime/generated_configs/qual_seed91.yaml`
- debug directories:
  - `/home/masa/ws_aic_runtime/qualification_debug/20260504_233421_submission_safe_v7_task_1`
  - `/home/masa/ws_aic_runtime/qualification_debug/20260504_233654_submission_safe_v7_task_1`
  - `/home/masa/ws_aic_runtime/qualification_debug/20260504_233850_submission_safe_v7_task_1`

Relevant files:

- score summary: `/home/masa/aic_results/qual_submission_safe_v7_20260504_233246/scoring.yaml`
- eval log: `/home/masa/aic_results/qual_submission_safe_v7_20260504_233246/eval.log`
- model log: `/home/masa/aic_results/qual_submission_safe_v7_20260504_233246/model.log`
- run metadata: `/home/masa/aic_results/qual_submission_safe_v7_20260504_233246/metadata.txt`

The recorded score breakdown was:

```yaml
total: 59.610579326329656
trial_1:
  tier_1:
    score: 1
  tier_2:
    score: 23.367056289065719
  tier_3:
    score: 4.3539585176992208
trial_2:
  tier_1:
    score: 1
  tier_2:
    score: 0
  tier_3:
    score: 0
trial_3:
  tier_1:
    score: 1
  tier_2:
    score: 23.210332838746446
  tier_3:
    score: 5.6792316808182681
```

### Shutdown Note

The evaluation itself completed and `scoring.yaml` was written successfully.
During shutdown, `ros_gz` still emitted a `double free or corruption` error from `component_container`, and the host-side `aic_model` process needed to be stopped after scoring had already finished. This did not prevent score generation for the validated run above.
