# Example Policies

この package には、ケーブル挿入タスクに対する異なるアプローチを示す baseline policy 実装が含まれています。これらは reference implementation であり、自分の policy を開発する際の出発点にもなります。

> [!NOTE]
> **Prerequisites:** これらの policy を実行する前に、評価環境が起動していることを確認してください。セットアップ手順は [Getting Started](../getting_started.md) を参照してください。
>
> **Command Format:**
> - **container workflow**（推奨）を使う場合: `distrobox enter -r aic_eval -- /entrypoint.sh [parameters]` で起動
> - **source から build** した場合: `ros2 launch aic_bringup aic_gz_bringup.launch.py [parameters]` で起動
> - policy の実行は `pixi run ros2 run`（Pixi workspace）または `ros2 run`（native ROS 2）

---

## 利用可能な Policy

### 1. WaveArm - 最小例

![Wave Arm Policy](../../../media/wave_arm_policy.gif)

`insert_cable()` callback を実装し、アームへ motion command を送る方法を示す最小例です。この policy はタスクを解こうとはせず、ロボットアームを前後に手を振るように動かすだけです。

**Purpose:** 基本的な Policy API 構造を示します。

**評価環境を起動:**
```bash
/entrypoint.sh ground_truth:=false start_aic_engine:=true
```

**policy を実行:**
```bash
pixi run ros2 run aic_model aic_model --ros-args -p use_sim_time:=true -p policy:=aic_example_policies.ros.WaveArm
```

**Source:** [`WaveArm.py`](../../aic_example_policies/aic_example_policies/ros/WaveArm.py)

---

### 2. CheatCode - Ground Truth Policy

![Cheat Code Policy](../../../media/cheat_code_policy.gif)

launch 時に `ground_truth:=true` を設定したときに simulation から提供される TF transformation tree を使う「チート」解法です。この policy は plug と port の pose を使って、`aic_controller` に送る target pose を計算します。

**Purpose:** 学習とデバッグに有用です。ground truth data は公式評価では利用できません。

**ground truth 付きで simulation を起動:**
```bash
/entrypoint.sh ground_truth:=true start_aic_engine:=true
```

**policy を実行:**
```bash
pixi run ros2 run aic_model aic_model --ros-args -p use_sim_time:=true -p policy:=aic_example_policies.ros.CheatCode
```

**Source:** [`CheatCode.py`](../../aic_example_policies/aic_example_policies/ros/CheatCode.py)

---

### 3. RunACT - ACT Policy

![Run ACT Policy](../../../media/run_act_policy.gif)

[HuggingFace](https://huggingface.co/grkw/aic_act_policy) 上で公開されている [LeRobot ACT](https://huggingface.co/docs/lerobot/en/act)（Action Chunking with Transformers）policy の proof-of-concept 実装です。この policy は、NVIDIA RTX A5000 マシン上で `lerobot-train` の既定パラメータを使い、[`lerobot_robot_aic`](../aic_utils/lerobot_robot_aic/README.md#recording-training-data) で説明されている `lerobot-record` により収集した小規模 dataset で学習されています。

`lerobot` を自分の hardware 構成で動かすには `pixi.toml` の修正が必要な場合があります。[Troubleshooting](../troubleshooting.md#nvidia-rtx-50xx-cards-not-supported-on-pytorch-version-locked-in-pixi) を参照してください。

**Purpose:** 学習済み neural network policy をケーブル挿入タスクへ統合する方法を示します。

**評価環境を起動:**
```bash
/entrypoint.sh ground_truth:=false start_aic_engine:=true
```

**policy を実行:**
```bash
pixi run ros2 run aic_model aic_model --ros-args -p use_sim_time:=true -p policy:=aic_example_policies.ros.RunACT
```

**Source:** [`RunACT.py`](../../aic_example_policies/aic_example_policies/ros/RunACT.py)

---

## Scoring Examples

各 policy の想定 scoring 結果と再現可能なテスト command については、[Scoring Test & Evaluation Guide](../scoring_tests.md) を参照してください。
