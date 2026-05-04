# Scoring Test & Evaluation Guide

この文書では、AIC の scoring system を実際に試せる再現可能な例を示します。
各例では、目的、検証する scoring category、期待される結果、および各 terminal で実行する正確な command を示します。

## 前提条件

すべての terminal で、ROS 2 workspace と Zenoh middleware を設定しておく必要があります。

```bash
source ~/ws_aic/install/setup.bash
export RMW_IMPLEMENTATION=rmw_zenoh_cpp
export ZENOH_ROUTER_CHECK_ATTEMPTS=-1
export ZENOH_CONFIG_OVERRIDE='transport/shared_memory/enabled=true;transport/shared_memory/transport_optimization/pool_size=536870912'
```

まだ build していない場合は、workspace を build します。

```bash
cd ~/ws_aic
GZ_BUILD_FROM_SOURCE=1 colcon build \
  --cmake-args -DCMAKE_BUILD_TYPE=Release \
  --merge-install --symlink-install \
  --packages-ignore lerobot_robot_aic
```

## Scoring Tier 参照表

| Tier | Category | Range | Description |
|------|----------|-------|-------------|
| 1 | Model validity | 0-1 | Pass/fail: model がロードされ、期待どおりに動作することを確認する前提チェック |
| 2 | Trajectory smoothness | 0-5 | アーム motion の滑らかさ。jerk に反比例（高いほど滑らか）。成功挿入または plug が port に十分近い場合にのみ加点 |
| 2 | Task duration | 0-10 | 速い完了ほど高得点。成功挿入または plug が port に十分近い場合にのみ加点 |
| 2 | Trajectory efficiency | 0-5 | エンドエフェクタ経路が短いほど高得点（高いほど直線的）。成功挿入または plug が port に十分近い場合にのみ加点 |
| 2 | Insertion force | 0 to -12 | force > 20 N が 1 秒超継続した場合の penalty |
| 2 | Off-limit contacts | 0 to -24 | enclosure または task board との衝突 penalty |
| 3 | Cable insertion | -10 or 0 to 60 | 誤った port への挿入で -10、正しい port への挿入で 60、部分挿入または近接で 0-40 |

engine 使用時、結果は `$AIC_RESULTS_DIR/scoring.yaml` に書き出されます。
既定 directory は `~/aic_results` です。各 engine 実行は前回の `scoring.yaml` を **上書き** するため、結果を保持したい場合は実行ごとに固有の `AIC_RESULTS_DIR` を設定してください。

---

## Example 1: Tier 1 Failure -- model を起動しない

**Goal:** `aic_model` を起動せずに engine を開始します。engine は policy を見つけられず、タイムアウトするはずです。

**Expected outcome:**
- engine は各 trial で timeout または failure を報告する
- すべての trial で Tier 1 は **fail** する
- Tier 2 と Tier 3 も同様に失敗する

### Terminal 0 -- Zenoh Router

```bash
ros2 run rmw_zenoh_cpp rmw_zenohd
```

### Terminal 1 -- Simulation + Engine（model なし）

```bash
AIC_RESULTS_DIR=~/aic_results/no_model \
ros2 launch aic_bringup aic_gz_bringup.launch.py \
  start_aic_engine:=true
```

---

## Example 2: CheatCode 参照ソリューション

**Goal:** CheatCode policy を reference solution として engine 全体のパイプラインで実行します。Tier 1（pass）、Tier 2（smoothness、duration、efficiency、force）、Tier 3（cable insertion）を検証します。

**Expected outcome:**
- 3 trial すべてが完了する
- すべての trial で Tier 1 は **pass** する
- Tier 2 では smoothness score が高く（最大 5）、成功 trial で task duration bonus（最大 10）が付き、force penalty と off-limit contact は発生しない
- Tier 3 では、すべての trial で successful cable insertion（60 points）が報告される

### Terminal 0 -- Zenoh Router

```bash
ros2 run rmw_zenoh_cpp rmw_zenohd
```

### Terminal 1 -- AIC Model（CheatCode）

```bash
ros2 run aic_model aic_model --ros-args -p use_sim_time:=true -p policy:=aic_example_policies.ros.CheatCode
```

### Terminal 2 -- Simulation + Engine

```bash
AIC_RESULTS_DIR=~/aic_results/cheatcode \
ros2 launch aic_bringup aic_gz_bringup.launch.py \
  ground_truth:=true \
  start_aic_engine:=true
```

---

## Example 3: WaveArm Baseline

**Goal:** WaveArm policy を engine で実行します。アームは手を振るだけで、ケーブルは挿入しません。Tier 1（pass）と Tier 2（smoothness、efficiency）を検証します。

**Expected outcome:**
- 3 trial すべてが完了する
- すべての trial で Tier 1 は **pass** する
- Tier 2 では高い smoothness score（滑らかな waving motion）が出る一方、task duration bonus はなく、force penalty と off-limit contact も発生しない
- Tier 3 はすべての trial で 0 score（アームは手を振るだけで、port に近づかない）

### Terminal 0 -- Zenoh Router

```bash
ros2 run rmw_zenoh_cpp rmw_zenohd
```

### Terminal 1 -- AIC Model（WaveArm）

```bash
ros2 run aic_model aic_model --ros-args -p use_sim_time:=true -p policy:=aic_example_policies.ros.WaveArm
```

### Terminal 2 -- Simulation + Engine

```bash
AIC_RESULTS_DIR=~/aic_results/wavearm \
ros2 launch aic_bringup aic_gz_bringup.launch.py \
  start_aic_engine:=true
```

---

## Example 4: Off-Limit Contact

**Goal:** `WallToucher` policy を engine で実行します。この policy は joint-space control を使ってアームを横方向に伸ばし、forearm を enclosure の wall panel に接触させます。scoring output に off-limit contact penalty が現れるはずです。

**Expected outcome:**
- 3 trial すべてが完了する
- すべての trial で Tier 1 は **pass** する
- Tier 2 では、robot link（例: `forearm_link`）が enclosure wall に衝突した trial すべてで off-limit contacts penalty（-24）が出る
- Tier 3 はすべての trial で 0 score（挿入なし、plug も port 近傍外）

> **Note — Off-limit contacts:** "Off-limit" model とは、task 中にロボットが接触してはいけない surface のことです。`OffLimitContactsPlugin` は次の 3 つの model を監視します。
>
> | Model | What it includes |
> |-------|-----------------|
> | `enclosure` | 床、corner post、天井（構造フレーム） |
> | `enclosure walls` | workspace を囲む透明アクリル panel |
> | `task_board` | board 本体と、その上に取り付けられたすべて（NIC card mount、SC port など） |
>
> penalty 対象になるのは、接触の片側が **robot link** の場合のみです。cable は独立した Gazebo model なので、この penalty を引き起こすことは想定されていません。

### Terminal 0 -- Zenoh Router

```bash
ros2 run rmw_zenoh_cpp rmw_zenohd
```

### Terminal 1 -- AIC Model（WallToucher）

```bash
ros2 run aic_model aic_model --ros-args -p use_sim_time:=true -p policy:=aic_example_policies.ros.WallToucher
```

### Terminal 2 -- Simulation + Engine

```bash
AIC_RESULTS_DIR=~/aic_results/wall_toucher \
ros2 launch aic_bringup aic_gz_bringup.launch.py \
  ground_truth:=true \
  start_aic_engine:=true
```

---

## Example 5: Excessive Force

**Goal:** `WallPresser` policy を engine で実行します。この policy は joint-space control を使って forearm を enclosure wall に高剛性で押し付け、持続的な接触 force により Tier 2 の insertion force penalty を発生させます。

**Expected outcome:**
- 3 trial すべてが完了する
- すべての trial で Tier 1 は **pass** する
- Tier 2 では、すべての trial で insertion force penalty（-12）が出る。wall contact の副作用として off-limit contacts penalty（-24）も出る場合がある
- Tier 3 はすべての trial で 0 score（挿入なし、plug も port 近傍外）

### Terminal 0 -- Zenoh Router

```bash
ros2 run rmw_zenoh_cpp rmw_zenohd
```

### Terminal 1 -- AIC Model（WallPresser）

```bash
ros2 run aic_model aic_model --ros-args -p use_sim_time:=true -p policy:=aic_example_policies.ros.WallPresser
```

### Terminal 2 -- Simulation + Engine

```bash
AIC_RESULTS_DIR=~/aic_results/wall_presser \
ros2 launch aic_bringup aic_gz_bringup.launch.py \
  ground_truth:=true \
  start_aic_engine:=true
```

---

## Example 6: Smooth Motion -- 低 jerk

**Goal:** `GentleGiant` policy を engine で実行します。この policy は低剛性・高減衰で 2 つの joint configuration の間をゆっくり動き、最小限の jerk を生みます。

**Expected outcome:**
- 3 trial すべてが完了する
- すべての trial で Tier 1 は **pass** する
- Tier 2 では smoothness score は付かず（plug が port 近傍にいない）、task duration bonus もなく、force penalty と off-limit contact も発生しない
- Tier 3 はすべての trial で 0 score（挿入なし、plug も port 近傍外）

### Terminal 0 -- Zenoh Router

```bash
ros2 run rmw_zenoh_cpp rmw_zenohd
```

### Terminal 1 -- AIC Model（GentleGiant）

```bash
ros2 run aic_model aic_model --ros-args -p use_sim_time:=true -p policy:=aic_example_policies.ros.GentleGiant
```

### Terminal 2 -- Simulation + Engine

```bash
AIC_RESULTS_DIR=~/aic_results/gentle_giant \
ros2 launch aic_bringup aic_gz_bringup.launch.py \
  ground_truth:=true \
  start_aic_engine:=true
```

---

## Example 7: Aggressive Motion -- 高 jerk

**Goal:** `SpeedDemon` policy を engine で実行します。この policy は高剛性・低減衰で 2 つの joint configuration の間を急速に動き、激しい motion により insertion force penalty を発生させます。

**Expected outcome:**
- 3 trial すべてが完了する
- すべての trial で Tier 1 は **pass** する
- Tier 2 では smoothness score は付かず（plug が port 近傍にいない）、さらにすべての trial で insertion force penalty（-12）が出る。アームは減衰不足により激しく振動し、F/T sensor 上で持続的 force が発生する。見た目としては position 間を鋭く行き来するはず
- Tier 3 はすべての trial で 0 score（挿入なし、plug も port 近傍外）

### Terminal 0 -- Zenoh Router

```bash
ros2 run rmw_zenoh_cpp rmw_zenohd
```

### Terminal 1 -- AIC Model（SpeedDemon）

```bash
ros2 run aic_model aic_model --ros-args -p use_sim_time:=true -p policy:=aic_example_policies.ros.SpeedDemon
```

### Terminal 2 -- Simulation + Engine

```bash
AIC_RESULTS_DIR=~/aic_results/speed_demon \
ros2 launch aic_bringup aic_gz_bringup.launch.py \
  ground_truth:=true \
  start_aic_engine:=true
```

---
