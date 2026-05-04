# aic_bringup

## 概要

`aic_bringup` は、AI for Industry Challenge のシミュレーション環境をセットアップするための launch file と設定を提供します。この package は evaluation component の一部であり、評価環境の起動、ロボットや task board の生成、trial の実行におけるエントリーポイントです。

**この package が行うこと:**
- UR5e ロボット付きの Gazebo シミュレーションを起動する
- さまざまなコネクタ mount を備えた構成可能な task board を生成する
- ロボット制御用の AIC controller を起動する
- 必要に応じて、自動 trial orchestration のために `aic_engine` を起動する
- sensor / actuator 通信用の ROS-Gazebo bridge を設定する

---

## クイックスタート

### 基本シミュレーション（Task Board なし）

```bash
source ~/ws_aic/install/setup.bash
export RMW_IMPLEMENTATION=rmw_zenoh_cpp
export ZENOH_CONFIG_OVERRIDE='transport/shared_memory/enabled=true;transport/shared_memory/transport_optimization/pool_size=536870912'

ros2 launch aic_bringup aic_gz_bringup.launch.py
```

これにより次が起動します。
- Gazebo シミュレーション
- 3 台の wrist camera を備えた UR5e ロボット
- AIC controller（impedance control mode）
- ROS-Gazebo bridge

### 完全な Qualification 環境

実際の qualification trial を実行する場合:

```bash
ros2 launch aic_bringup aic_gz_bringup.launch.py \
  ground_truth:=false \
  start_aic_engine:=true
```

これにより以下が追加されます。
- trial orchestration 用の AIC engine
- 評価 scoring system
- ground truth data の非表示（実評価と同様）

### Ground Truth ありの開発モード

policy を開発する際は、デバッグしやすくするため ground truth を有効にします。

```bash
ros2 launch aic_bringup aic_gz_bringup.launch.py \
  ground_truth:=true \
  spawn_task_board:=true
```

これにより次が提供されます。
- task board 要素の ground truth TF frame
- Gazebo での視覚的デバッグ
- より容易な policy 開発

---

## Launch Files

<a id="1-aic_gz_bringuplaunchpy"></a>
### 1. `aic_gz_bringup.launch.py`

完全な AIC シミュレーション環境用の**主要 launch file**です。

> [!NOTE]
> 評価時、task board とすべての component の roll / pitch は固定（すべて 0.0）、SC port の yaw も 0.0 に固定されますが、参加者は policy 学習時の domain randomization のために任意の向きを設定して構いません。

#### 使い方
```bash
ros2 launch aic_bringup aic_gz_bringup.launch.py [parameters]
```

#### 設定可能パラメータ

**Robot Spawn Position:**
- `robot_x` (default: `"-0.2"`) - ロボット生成位置 X（meters）
- `robot_y` (default: `"0.2"`) - ロボット生成位置 Y（meters）
- `robot_z` (default: `"1.14"`) - ロボット生成位置 Z（meters）
- `robot_roll` (default: `"0.0"`) - ロボット生成時の roll 向き（radians）
- `robot_pitch` (default: `"0.0"`) - ロボット生成時の pitch 向き（radians）
- `robot_yaw` (default: `"-3.141"`) - ロボット生成時の yaw 向き（radians）

**Controller Configuration:**
- `controllers_file` (default: `"ur_controllers.yaml"`) - controller 設定を含む YAML file
- `activate_joint_controller` (default: `"true"`) - 起動時に joint controller を activate するか
- `initial_joint_controller` (default: `"aic_controller"`) - 最初に activate する controller
- `description_file` (default: `"ur.urdf.xacro"`) - ロボット description file

**Task Board Configuration:**
- `spawn_task_board` (default: `"false"`) - task board を生成するか
- `task_board_description_file` (default: `"task_board.urdf.xacro"`) - task board URDF/XACRO file
- `task_board_x` (default: `"0.15"`) - task board 生成位置 X（meters）
- `task_board_y` (default: `"-0.2"`) - task board 生成位置 Y（meters）
- `task_board_z` (default: `"1.14"`) - task board 生成位置 Z（meters）
- `task_board_roll` (default: `"0.0"`) - task board 生成時の roll 向き（radians）
- `task_board_pitch` (default: `"0.0"`) - task board 生成時の pitch 向き（radians）
- `task_board_yaw` (default: `"0.0"`) - task board 生成時の yaw 向き（radians）

**Cable Configuration:**
- `spawn_cable` (default: `"false"`) - cable を生成するか
- `cable_description_file` (default: `"cable.sdf.xacro"`) - cable SDF/XACRO file
- `attach_cable_to_gripper` (default: `"false"`) - cable を gripper に取り付けるか
- `cable_type` (default: `"sfp_sc_cable"`) - 生成する cable の種類。選択肢: [`sfp_sc_cable`, `sfp_sc_cable_reversed`]
- `cable_x` (default: `"0.172"`) - cable 生成位置 X（meters）
- `cable_y` (default: `"0.024"`) - cable 生成位置 Y（meters）
- `cable_z` (default: `"1.518"`) - cable 生成位置 Z（meters）
    - Note: `cable_type` が `sfp_sc_cable_reversed` の場合は `cable_z` を `1.508` に設定してください
- `cable_roll` (default: `"0.4432"`) - cable 生成時の roll 向き（radians）
- `cable_pitch` (default: `"-0.48"`) - cable 生成時の pitch 向き（radians）
- `cable_yaw` (default: `"1.3303"`) - cable 生成時の yaw 向き（radians）

**Gazebo Configuration:**
- `world_file` (default: `"aic.sdf"`) - Gazebo world file
- `gazebo_gui` (default: `"true"`) - Gazebo GUI を起動するか
- `ros_gz_bridge_config_file` (default: `"ros_gz_bridge.yaml"`) - ROS-Gazebo bridge 設定

**Visualization:**
- `launch_rviz` (default: `"false"`) - 可視化用に RViz を起動するか
- `rviz_config_file` (default: `"view_robot.rviz"`) - RViz 設定 file

**Ground Truth:**
- `ground_truth` (default: `"false"`) - TF topic に ground truth pose data を含めるか

**AIC Engine:**
- `start_aic_engine` (default: `"false"`) - 評価用に `aic_engine` orchestrator node を起動する
- `shutdown_on_aic_engine_exit` (default: `"false"`) - `aic_engine` 終了時に launch 全体を終了し、その exit code を伝播する。`start_aic_engine` が `true` のときのみ有効。trial 完了後に container を終了したい自動評価で有用
- `aic_engine_config_file` (default: `"aic_engine/config/sample_config.yaml"`) - AIC engine 設定 YAML file の絶対 path
- `model_discovery_timeout_seconds` (default: `"30"`) - participant model を発見するまでの timeout

---

### 2. `spawn_task_board.launch.py`

既存の Gazebo シミュレーション内に task board を生成するための standalone launch file です。

> [!NOTE]
> 評価時、task board とすべての component の roll / pitch は固定（すべて 0.0）、SC port の yaw も 0.0 に固定されますが、参加者は policy 学習時の domain randomization のために任意の向きを設定して構いません。

#### 使い方
```bash
ros2 launch aic_bringup spawn_task_board.launch.py
```

#### 設定可能パラメータ

**Task Board Base Configuration:**
- `task_board_description_file` (default: `"task_board.urdf.xacro"`) - task board URDF/XACRO description file
- `task_board_x` (default: `"0.25"`) - task board 生成位置 X（meters）
- `task_board_y` (default: `"0.0"`) - task board 生成位置 Y（meters）
- `task_board_z` (default: `"1.14"`) - task board 生成位置 Z（meters）
- `task_board_roll` (default: `"0.0"`) - task board 生成時の roll 向き（radians）
- `task_board_pitch` (default: `"0.0"`) - task board 生成時の pitch 向き（radians）
- `task_board_yaw` (default: `"0.0"`) - task board 生成時の yaw 向き（radians）

**Mount Rails (LC/SFP/SC):**

task board には LC、SFP、SC connector mount 用の mount rail が 6 本あります。各 rail では、存在有無、rail に沿った平行移動量、向きを設定できます。

*LC Mount Rail 0（left side）:*
- `lc_mount_rail_0_present` (default: `"false"`) - rail 0 に LC mount が存在するか
- `lc_mount_rail_0_translation` (default: `"0.0"`) - rail に沿った平行移動量（meters, range: -0.09625 to 0.09625）
- `lc_mount_rail_0_roll` (default: `"0.0"`) - roll 向き（radians）
- `lc_mount_rail_0_pitch` (default: `"0.0"`) - pitch 向き（radians）
- `lc_mount_rail_0_yaw` (default: `"0.0"`) - yaw 向き（radians）

*SFP Mount Rail 0（left side）:*
- `sfp_mount_rail_0_present` (default: `"false"`) - rail 0 に SFP mount が存在するか
- `sfp_mount_rail_0_translation` (default: `"0.0"`) - rail に沿った平行移動量（meters, range: -0.09625 to 0.09625）
- `sfp_mount_rail_0_roll` (default: `"0.0"`) - roll 向き（radians）
- `sfp_mount_rail_0_pitch` (default: `"0.0"`) - pitch 向き（radians）
- `sfp_mount_rail_0_yaw` (default: `"0.0"`) - yaw 向き（radians）

*SC Mount Rail 0（left side）:*
- `sc_mount_rail_0_present` (default: `"false"`) - rail 0 に SC mount が存在するか
- `sc_mount_rail_0_translation` (default: `"0.0"`) - rail に沿った平行移動量（meters, range: -0.09625 to 0.09625）
- `sc_mount_rail_0_roll` (default: `"0.0"`) - roll 向き（radians）
- `sc_mount_rail_0_pitch` (default: `"0.0"`) - pitch 向き（radians）
- `sc_mount_rail_0_yaw` (default: `"0.0"`) - yaw 向き（radians）

*LC Mount Rail 1（right side）:*
- `lc_mount_rail_1_present` (default: `"false"`) - rail 1 に LC mount が存在するか
- `lc_mount_rail_1_translation` (default: `"0.0"`) - rail に沿った平行移動量（meters, range: -0.09625 to 0.09625）
- `lc_mount_rail_1_roll` (default: `"0.0"`) - roll 向き（radians）
- `lc_mount_rail_1_pitch` (default: `"0.0"`) - pitch 向き（radians）
- `lc_mount_rail_1_yaw` (default: `"0.0"`) - yaw 向き（radians）

*SFP Mount Rail 1（right side）:*
- `sfp_mount_rail_1_present` (default: `"false"`) - rail 1 に SFP mount が存在するか
- `sfp_mount_rail_1_translation` (default: `"0.0"`) - rail に沿った平行移動量（meters, range: -0.09625 to 0.09625）
- `sfp_mount_rail_1_roll` (default: `"0.0"`) - roll 向き（radians）
- `sfp_mount_rail_1_pitch` (default: `"0.0"`) - pitch 向き（radians）
- `sfp_mount_rail_1_yaw` (default: `"0.0"`) - yaw 向き（radians）

*SC Mount Rail 1（right side）:*
- `sc_mount_rail_1_present` (default: `"false"`) - rail 1 に SC mount が存在するか
- `sc_mount_rail_1_translation` (default: `"0.0"`) - rail に沿った平行移動量（meters, range: -0.09625 to 0.09625）
- `sc_mount_rail_1_roll` (default: `"0.0"`) - roll 向き（radians）
- `sc_mount_rail_1_pitch` (default: `"0.0"`) - pitch 向き（radians）
- `sc_mount_rail_1_yaw` (default: `"0.0"`) - yaw 向き（radians）

**SC Port Rails:**

SC port module を取り付けるための SC port rail が 2 本あります。

*SC Port 0:*
- `sc_port_0_present` (default: `"false"`) - rail 0 に SC port が存在するか
- `sc_port_0_translation` (default: `"0.0"`) - rail に沿った平行移動量（meters）
- `sc_port_0_roll` (default: `"0.0"`) - roll 向き（radians）
- `sc_port_0_pitch` (default: `"0.0"`) - pitch 向き（radians）
- `sc_port_0_yaw` (default: `"0.0"`) - yaw 向き（radians）

*SC Port 1:*
- `sc_port_1_present` (default: `"false"`) - rail 1 に SC port が存在するか
- `sc_port_1_translation` (default: `"0.0"`) - rail に沿った平行移動量（meters）
- `sc_port_1_roll` (default: `"0.0"`) - roll 向き（radians）
- `sc_port_1_pitch` (default: `"0.0"`) - pitch 向き（radians）
- `sc_port_1_yaw` (default: `"0.0"`) - yaw 向き（radians）

**NIC Card Mount Rails:**

Network Interface Card を取り付けるための NIC card mount rail が 5 本あります。

*NIC Card Mount 0-4:*（mount 0, 1, 2, 3, 4 それぞれで同じ parameter を持つ）
- `nic_card_mount_N_present` (default: `"false"`) - NIC card mount N が存在するか
- `nic_card_mount_N_translation` (default: `"0.0"`) - rail に沿った平行移動量（meters）
- `nic_card_mount_N_roll` (default: `"0.0"`) - roll 向き（radians）
- `nic_card_mount_N_pitch` (default: `"0.0"`) - pitch 向き（radians）
- `nic_card_mount_N_yaw` (default: `"0.0"`) - yaw 向き（radians）

---

### 3. `spawn_cable.launch.py`

既存の Gazebo シミュレーション内に cable を生成するための standalone launch file です。

#### 使い方
```bash
ros2 launch aic_bringup spawn_cable.launch.py
```

#### 設定可能パラメータ

- `cable_description_file` (default: `"cable.sdf.xacro"`) - cable URDF/XACRO description file
- `cable_x` (default: `"-0.35"`) - cable 生成位置 X（meters）
- `cable_y` (default: `"0.4"`) - cable 生成位置 Y（meters）
- `cable_z` (default: `"1.15"`) - cable 生成位置 Z（meters）
- `cable_roll` (default: `"0.0"`) - cable 生成時の roll 向き（radians）
- `cable_pitch` (default: `"0.0"`) - cable 生成時の pitch 向き（radians）
- `cable_yaw` (default: `"0.0"`) - cable 生成時の yaw 向き（radians）
- `attach_cable_to_gripper` (default: `"false"`) - cable を gripper に取り付けるか

---

## 使用例

### 基本シミュレーション起動
既定 parameter で完全なシミュレーションを起動します。
```bash
ros2 launch aic_bringup aic_gz_bringup.launch.py
```

### カスタム Robot Position で起動
```bash
ros2 launch aic_bringup aic_gz_bringup.launch.py robot_x:=1.0 robot_y:=0.5
```

### Task Board と Cable 付きで起動
```bash
ros2 launch aic_bringup aic_gz_bringup.launch.py spawn_task_board:=true spawn_cable:=true
```

### Rail 0 に LC Mount を置いて Task Board を生成
```bash
ros2 launch aic_bringup spawn_task_board.launch.py \
  lc_mount_rail_0_present:=true \
  lc_mount_rail_0_translation:=0.05
```

---

## Impedance Controller 付きで起動する

### AIC Controller 付きでシミュレーションを起動

```bash
ros2 launch aic_bringup aic_gz_bringup.launch.py
```

### joint（`JointMotionUpdate`）と Cartesian target（`MotionUpdate`）の両方を送る script を起動

この script は、target 送信の合間に `ChangeTargetMode` service も呼び出し、target mode を Joint と Cartesian の間で切り替えます。
```bash
ros2 run aic_bringup test_impedance.py
```

---

## Notes

- すべての position 値は meters 単位
- すべての orientation 値は radians 単位
- mount rail の平行移動範囲は衝突を防ぐため -0.09625 から 0.09625 meters に制限されている
- mount rail は type 固有であり、LC、SFP、SC mount はそれぞれ対応する rail にのみ取り付け可能
- port rail（`sc_port` と `nic_card_mount`）は mount rail とは別物
