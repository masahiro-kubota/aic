# AIC Engine

AIC Engine は AI for Industry Challenge の orchestrator です。
trial 実行を管理し、participant model を検証し、simulation 内に task board を生成し、task 完了を監視します。

## 概要

engine は state machine として動作し、次の state を遷移します。

1. **Uninitialized** → **Initialized** → **Running** → **Completed**（または **Error**）

各 trial で、engine は次の手順を実行します。

1. **Model Ready**: participant の lifecycle node が利用可能で、正しい state にあることを検証
2. **Endpoints Ready**: 必要な ROS node、topic、service がすべて利用可能であることを確認
3. **Simulator Ready**: 設定済み component 付きの task board を Gazebo に生成
4. **Scoring Ready**: scoring system を準備
5. **Task Started**: participant model に task goal を送信
6. **Task Completed**: task 完了を監視し検証

## 仕組み

### Lifecycle Node Validation

engine は participant model について次を検証します。
- ROS 2 lifecycle node が正しく実装されている
- 標準 lifecycle service（`get_state`, `change_state`）を公開している
- `unconfigured` state から開始する
- `unconfigured` の間は静止している（ロボットが動かない）
- `configured` state の間（activate 前）は action goal を拒否する

### Task Board Spawning

engine は YAML configuration file に基づいて task board を動的生成し、以下をサポートします。
- configurable pose（位置と向き）
- 5 本の rail（`nic_rail_0` から `nic_rail_4`）上の NIC card mount
- 2 本の rail（`sc_rail_0` と `sc_rail_1`）上の SC port
- 6 本の mount rail（`lc_mount_rail_0/1`, `sfp_mount_rail_0/1`, `sc_mount_rail_0/1`）上の LC、SFP、SC mount
- 各 component の平行移動と回転の調整
- ground truth pose の publish（任意）

### Trial Execution

各 trial は順番に実行されます。
1. YAML から trial configuration を読み込む
2. configuration の構造を検証する
3. trial state を順に進める
4. cleanup（生成した entity を削除）
5. 次の trial へ進むか、完了する

## 使用方法

### Engine を実行する

```bash
ros2 run aic_engine aic_engine --ros-args \
  -p config_file_path:=/path/to/config.yaml \
  -p model_node_name:=aic_model \
  -p ground_truth:=false \
  -p endpoint_ready_timeout_seconds:=10 \
  -p model_discovery_timeout_seconds:=30 \
  -p model_configure_timeout_seconds:=60 \
  -p use_sim_time:=true
```

### ROS Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `config_file_path` | string | "" | **Required**。trial configuration YAML file の path |
| `model_node_name` | string | "aic_model" | participant の lifecycle node 名 |
| `adapter_node_name` | string | "aic_adapter_node" | adapter node 名（将来用） |
| `gripper_frame_name` | string | "gripper/tcp" | gripper frame 名 |
| `ground_truth` | bool | false | task board から ground truth pose を publish するか |
| `skip_model_ready` | bool | false | model readiness check をスキップする（テスト専用） |
| `skip_ready_simulator` | bool | false | simulator readiness と entity の生成 / 削除をスキップする（テスト専用） |
| `endpoint_ready_timeout_seconds` | int | 10 | 必須 endpoint を待つ timeout |
| `model_discovery_timeout_seconds` | int | 30 | participant model を発見する timeout |
| `model_configure_timeout_seconds` | int | 60 | model configuration check の timeout |
| `model_activate_timeout_seconds` | int | 60 | model activation の timeout |
| `model_deactivate_timeout_seconds` | int | 60 | model deactivation の timeout |
| `model_cleanup_timeout_seconds` | int | 60 | model cleanup の timeout |
| `model_shutdown_timeout_seconds` | int | 60 | model shutdown の timeout |

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `AIC_RESULTS_DIR` | `$HOME/aic_results` | scoring data と bag file の出力先 directory。未設定または空の場合は `$HOME/aic_results` が使われる |


### テスト

sample configuration で実行する例:

```bash
ros2 run aic_engine aic_engine --ros-args \
  -p config_file_path:=$(ros2 pkg prefix aic_engine)/share/aic_engine/config/sample_config.yaml \
  -p skip_model_ready:=false \
  -p skip_ready_simulator:=false \
  -p use_sim_time:=true
```
