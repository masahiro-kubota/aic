#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <stage>"
  exit 1
fi

STAGE="$1"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-/home/masa/ws_aic}"
REPO_ROOT="${REPO_ROOT:-$WORKSPACE_ROOT/src/aic}"
RESULT_ROOT="${RESULT_ROOT:-/home/masa/aic_results}"
DEBUG_ROOT="${DEBUG_ROOT:-/home/masa/ws_aic_runtime/qualification_debug}"
DISPLAY_VALUE="${DISPLAY:-:1}"
GROUND_TRUTH_VALUE="${AIC_QUAL_GROUND_TRUTH:-false}"
GAZEBO_GUI_VALUE="${AIC_QUAL_GAZEBO_GUI:-true}"
RVIZ_VALUE="${AIC_QUAL_LAUNCH_RVIZ:-false}"
ENGINE_CONFIG_VALUE="${AIC_ENGINE_CONFIG_FILE:-}"
ENGINE_CONFIG_IN_CONTAINER="$ENGINE_CONFIG_VALUE"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="$RESULT_ROOT/qual_${STAGE}_${RUN_ID}"

mkdir -p "$OUT_DIR"

pkill -f "pixi run --as-is ros2 run aic_model aic_model" >/dev/null 2>&1 || true
pkill -f "/.pixi/envs/default/bin/ros2 run aic_model aic_model" >/dev/null 2>&1 || true
pkill -f "aic_model .*QualPhasePilot" >/dev/null 2>&1 || true
docker restart aic_eval >/dev/null

if [[ -n "$ENGINE_CONFIG_VALUE" && -f "$ENGINE_CONFIG_VALUE" ]]; then
  ENGINE_CONFIG_IN_CONTAINER="/tmp/$(basename "$ENGINE_CONFIG_VALUE")"
  docker cp "$ENGINE_CONFIG_VALUE" "aic_eval:${ENGINE_CONFIG_IN_CONTAINER}" >/dev/null
fi

docker exec -u masa -i aic_eval bash -lc "
export DISPLAY='${DISPLAY_VALUE}'
/entrypoint.sh ground_truth:=${GROUND_TRUTH_VALUE} start_aic_engine:=true shutdown_on_aic_engine_exit:=true launch_rviz:=${RVIZ_VALUE} gazebo_gui:=${GAZEBO_GUI_VALUE} ${ENGINE_CONFIG_IN_CONTAINER:+aic_engine_config_file:=${ENGINE_CONFIG_IN_CONTAINER}}
" >"$OUT_DIR/eval.log" 2>&1 &
EVAL_PID=$!

sleep 20

(
  cd "$REPO_ROOT"
  export DISPLAY="$DISPLAY_VALUE"
  export RMW_IMPLEMENTATION=rmw_zenoh_cpp
  export AIC_QUAL_STAGE="$STAGE"
  export AIC_LEARNED_PORT_DATASET_ROOT="${AIC_LEARNED_PORT_DATASET_ROOT:-}"
  export AIC_LEARNED_PORT_DATASET_SPLIT="${AIC_LEARNED_PORT_DATASET_SPLIT:-train}"
  export AIC_QUAL_LEARNED_SC_MODEL_DIR="${AIC_QUAL_LEARNED_SC_MODEL_DIR:-}"
  export AIC_QUAL_LEARNED_SFP_MODEL_DIR="${AIC_QUAL_LEARNED_SFP_MODEL_DIR:-}"
  pixi run --as-is ros2 run aic_model aic_model --ros-args -p use_sim_time:=true -p policy:=aic_example_policies.ros.QualPhasePilot
) >"$OUT_DIR/model.log" 2>&1 || MODEL_STATUS=$?

MODEL_STATUS="${MODEL_STATUS:-0}"
wait "$EVAL_PID" || EVAL_STATUS=$?
EVAL_STATUS="${EVAL_STATUS:-0}"

cp /home/masa/aic_results/scoring.yaml "$OUT_DIR/scoring.yaml"
find "$DEBUG_ROOT" -maxdepth 1 -type d | sort | tail -n 6 >"$OUT_DIR/debug_dirs.txt"
cat >"$OUT_DIR/metadata.txt" <<EOF
stage=$STAGE
run_id=$RUN_ID
display=$DISPLAY_VALUE
ground_truth=$GROUND_TRUTH_VALUE
gazebo_gui=$GAZEBO_GUI_VALUE
launch_rviz=$RVIZ_VALUE
engine_config=${ENGINE_CONFIG_IN_CONTAINER:-default}
model_status=$MODEL_STATUS
eval_status=$EVAL_STATUS
EOF

echo "$OUT_DIR"
