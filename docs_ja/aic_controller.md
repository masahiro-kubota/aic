# aic_controller

[aic_controller](../aic_controller/) パッケージは ROS 2 用のコントローラです。制御 policy や planner から約 $\approx 10 \to 30\text{ Hz}$ で送られてくる目標コマンド（Joint または Cartesian）を受け取り、約 $\approx 500\text{ Hz}$ でロボットハードウェアへ橋渡しします。目標値には、安全性チェックと平滑化が適用された後にインピーダンス制御がかかります。

## アーキテクチャ

アーキテクチャの概要を次の図に示します。
<img width="1889" height="437" alt="image" src="../../media/aic_controller.png" />

### 制御パイプライン

1. **Command Clamping**: 入力された目標値を安全範囲内に収めます。
    - Joint 目標は、ロボットの URDF / description で定義された制限値にクランプされます。
    - Cartesian 目標は、ユーザー指定パラメータに従ってクランプされます。

2. **Command Interpolation**: クランプされた目標値を平滑化し、低頻度な policy コマンドを滑らかな高速 setpoint に変換します。

3. **Impedance Control**: 平滑化された setpoint を [CartesianImpedanceAction](../aic_controller/include/aic_controller/actions/cartesian_impedance_action.hpp) または [JointImpedanceAction](../aic_controller/include/aic_controller/actions/joint_impedance_action.hpp) が処理し、必要な関節トルクを計算します。

4. **Gravity Compensation**: [GravityCompensationAction](../aic_controller/include/aic_controller/actions/gravity_compensation_action.hpp) によって、ロボットリンクにかかる重力を相殺する追加トルクを計算します。

5. **Command Execution**: インピーダンス制御トルクと重力補償トルクを加算し、ロボット関節へ送ります。

> [!NOTE]
> `aic_controller` は、大きな追従誤差が一定時間内に減少しない場合、コントローラの目標値をリセットします（`tracking_error` として [aic_ros2_controllers.yaml](../aic_bringup/config/aic_ros2_controllers.yaml) で設定可能）。これはテレオペレーションでよくある問題への対策です。ロボットが衝突中にもユーザーがコマンドを送り続けると追従誤差が蓄積しますが、リセットがないと、衝突が解消した瞬間にその蓄積誤差を一気に実行してしまいます。

### Cartesian インピーダンス制御

Cartesian 目標は `CartesianImpedanceAction` が処理し、エンドエフェクタの現在位置と目標位置の差に基づいて関節トルクを計算します。

$$
\tau = \mathbf{J}^T \Big[ \mathbf{K}_p (\mathbf{x}_{des} - \mathbf{x}) + \mathbf{K}_d (\dot{\mathbf{x}}_{des} - \dot{\mathbf{x}}) + \mathbf{W}_f \Big] + \tau_{null}
$$

**各項の意味:**
- $\tau \in \mathbb{R}^n$: 計算された関節トルク
- $\mathbf{J} \in \mathbb{R}^{6 \times n}$: ロボットアームのヤコビ行列
- $\mathbf{K}_p, \mathbf{K}_d \in \mathbb{R}^{6 \times 6}$: 剛性行列と減衰行列
- $\mathbf{x}_{des}, \mathbf{x} \in \mathbb{R}^6$: 目標および現在のエンドエフェクタ pose
- $\mathbf{W}_f \in \mathbb{R}^6$: 外部から与える追加の force / torque
- $\tau_{null} \in \mathbb{R}^n$: 関節制限回避などの副次タスクに使う追加トルク

### Joint インピーダンス制御

Joint 目標は `JointImpedanceAction` が処理し、目標関節位置と現在の関節位置の差に基づいて関節トルクを計算します。

$$
\tau = \mathbf{K}_p (\mathbf{q}_{des} - \mathbf{q}) + \mathbf{K}_d (\dot{\mathbf{q}}_{des} - \dot{\mathbf{q}}) + \tau_f
$$

**各項の意味:**
- $\tau \in \mathbb{R}^n$: 計算された関節トルク
- $\mathbf{K}_p, \mathbf{K}_d \in \mathbb{R}^n$: 各関節の剛性と減衰
- $\mathbf{q}_{des}, \mathbf{q} \in \mathbb{R}^n$: 目標および現在の関節位置
- $\dot{\mathbf{q}}_{des}, \dot{\mathbf{q}} \in \mathbb{R}^n$: 目標および現在の関節速度
- $\tau_f \in \mathbb{R}^n$: 追加の関節トルク


### ROS 2 インターフェース

#### コマンドインターフェース

`aic_controller` は 2 つの ROS 2 Topic からコマンドを受け付けます。メッセージ定義の詳細は [コントローラターゲットパラメータ](#controller-target-parameters) を参照してください。

- **Cartesian Targets** ([`MotionUpdate`](../aic_interfaces/aic_control_interfaces/msg/MotionUpdate.msg)): `/aic_controller/pose_commands`
- **Joint Targets** ([`JointMotionUpdate`](../aic_interfaces/aic_control_interfaces/msg/JointMotionUpdate.msg)): `/aic_controller/joint_commands`

#### joint と Cartesian のターゲットモード切り替え

joint 制御と Cartesian 制御を切り替えるには、`/aic_controller/change_target_mode` に ROS 2 service request を送ります。コントローラは既定で **Cartesian** mode で起動します。

ROS 2 CLI を使ってコントローラのターゲットモードを切り替える service call の例:
```bash
# Cartesian target mode に切り替える service request を送信
ros2 service call /aic_controller/change_target_mode aic_control_interfaces/srv/ChangeTargetMode "{target_mode: {mode: 1}}"

# joint target mode に切り替える service request を送信
ros2 service call /aic_controller/change_target_mode aic_control_interfaces/srv/ChangeTargetMode "{target_mode: {mode: 2}}"
```

> **Note:** コントローラは同時に 1 つのモードでしか動作できません。たとえば `Cartesian` mode の場合は `/aic_controller/pose_commands` だけを受け取り、`/aic_controller/joint_commands` からのメッセージは無視します。対象のコマンドを受け付ける前に、`/aic_controller/change_target_mode` service を使ってモードを切り替える必要があります。詳しくは [コントローラ設定](./aic_interfaces.md#controller-configuration) を参照してください。

#### 状態フィードバック

コントローラは `/aic_controller/controller_state`（[`ControllerState`](../aic_interfaces/aic_control_interfaces/msg/ControllerState.msg)）にリアルタイムデータを publish します。このメッセージには以下が含まれます。
- 現在の TCP pose と velocity
- 目標 TCP pose
- 現在の TCP pose と目標 TCP pose の誤差
- 目標関節トルク

#### Force-Torque Sensor のゼロ点調整

コントローラは `/aic_controller/tare_force_torque_sensor` に、Force-Torque sensor をゼロ点調整する service を提供しています。この service は現在の force / torque の読み値をゼロへリセットし、センサーのキャリブレーションやバイアス除去に役立ちます。ゼロ点調整後のオフセットは [`ControllerState`](../aic_interfaces/aic_control_interfaces/msg/ControllerState.msg) メッセージ内の `fts_tare_offset` として publish されます。

> **Note:** 各学習エピソードの開始前（すなわち、テレオペレーションや環境内でのケーブル生成の前）には、正確な force-torque フィードバックのために Force/Torque Sensor（F/T Sensor）のゼロ点調整を行うことが重要です。

```bash
# FT sensor をゼロ点調整
ros2 service call /aic_controller/tare_force_torque_sensor std_srvs/srv/Trigger
```

> **Important:** この service は評価中には **利用できません**。Force-Torque sensor の読み値はスコアリングに使用されるため、参加者は競技実行中にセンサーをゼロ点調整できません。

<a id="controller-target-parameters"></a>
## コントローラターゲットパラメータ

以下の表は、policy がタスクに応じて変更することの多い主なコントローラパラメータを示します。

### MotionUpdate

| Parameter | Type | 説明 |
| :--- | :--- | :--- |
| `header` | `std_msgs/Header` | `frame_id` は `gripper/tcp`（TCP frame）または `base_link`（グローバル frame）のいずれかである必要があります。<br />`stamp` field には現在の timestamp を設定してください。 |
| `pose` | `geometry_msgs/Pose` | TCP の目標 Cartesian pose。<br />`trajectory_generation_mode` が `MODE_POSITION` のときに使用されます。`frame_id` が `base_link` の場合、この pose はロボット base からの相対値です。`frame_id` が `gripper/tcp` の場合は、現在の TCP 位置からのオフセットです。 |
| `velocity` | `geometry_msgs/Twist` | TCP の目標 velocity。<br />`trajectory_generation_mode` が `MODE_VELOCITY` のときに使用されます。velocity は `frame_id` で指定した frame に対する相対値です。 |
| `target_stiffness` | `float64[36]` | ロボットが目標 pose からずれることにどれだけ強く抵抗するかを決める 6x6 の剛性行列。<br />値が大きいほど硬い制御、小さいほどコンプライアントな制御になります。 |
| `target_damping` | `float64[36]` | 振動を抑える 6x6 の減衰行列。<br />通常は `target_stiffness` に対して調整し、ふらつきを防いで安定した動作を実現します。 |
| `feedforward_wrench_at_tip` | `geometry_msgs/Wrench` | TCP に加える任意の外力 / 外トルク。<br />一定の下向き force を与える接触タスクや、既知の tool と環境の相互作用を扱う場面で有用です。 |
| `wrench_feedback_gains_at_tip` | `float64[6]` | センサーで計測した force / torque に対するフィードバックゲイン。 |
| `trajectory_generation_mode` | `TrajectoryGenerationMode` | 目標値をどのように解釈するかを指定します。<br />`MODE_POSITION` は `pose` の値に従います。<br />`MODE_VELOCITY` は `velocity` の値に従います。 |

#### 例

ROS 2 CLI を使って [`MotionUpdate`](../aic_interfaces/aic_control_interfaces/msg/MotionUpdate.msg) メッセージで pose target を publish する例:
```bash
# Cartesian target mode に切り替える service request を送信
ros2 service call /aic_controller/change_target_mode aic_control_interfaces/srv/ChangeTargetMode "{target_mode: {mode: 1}}"

# Cartesian pose target を送信
ros2 topic pub --once /aic_controller/pose_commands aic_control_interfaces/msg/MotionUpdate "{
  header: {
    frame_id: 'base_link'
  },
  pose: {
    position: {x: -0.501, y: -0.175, z: 0.2},
    orientation: {x: 0.7071068, y: 0.7071068, z: 0.0, w: 0.0}
  },
  target_stiffness: [
    85.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 85.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 85.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 85.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 85.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 85.0
  ],
  target_damping: [
    75.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 75.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 75.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 75.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 75.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 75.0
  ],
  feedforward_wrench_at_tip: {
    force: {x: 0.0, y: 0.0, z: 0.0},
    torque: {x: 0.0, y: 0.0, z: 0.0}
  },
  wrench_feedback_gains_at_tip: [
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0
  ],
  trajectory_generation_mode: {mode: 2}
}"
```

同様に、velocity target の publish も簡単です。
```bash
# 以下のコマンドは、別の target で上書きされるまで、TCP を x 軸方向へ 0.025 m/s で移動し、z 軸まわりに 0.25 rad/s で回転させます
ros2 topic pub --once /aic_controller/pose_commands aic_control_interfaces/msg/MotionUpdate "{
  header: {
    frame_id: 'gripper/tcp'
  },
  velocity: {
    linear: {x: 0.025, y: 0.0, z: 0.0},
    angular: {x: 0.0, y: 0.0, z: 0.25}
  },
  target_stiffness: [
    85.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 85.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 85.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 85.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 85.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 85.0
  ],
  target_damping: [
    75.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 75.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 75.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 75.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 75.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 75.0
  ],
  trajectory_generation_mode: {mode: 1}
}"
```

`rclpy` を使った例は [test_impedance.py](../aic_bringup/scripts/test_impedance.py) 内の `generate_motion_update()` 関数を参照してください。

### JointMotionUpdate

| Parameter | Type | 説明 |
| :--- | :--- | :--- |
| `target_state` | `trajectory_msgs/JointTrajectoryPoint` | 各ロボット関節の目標値。<br />`trajectory_generation_mode` が `MODE_POSITION` の場合は `positions` field を使用します。<br />`trajectory_generation_mode` が `MODE_VELOCITY` の場合は `velocities` field を使用します。 |
| `target_stiffness` | `float64[]` | 各ロボット関節の剛性。<br />値が大きいほど硬い制御、小さいほどコンプライアントな制御になります。配列サイズは関節数と一致させてください。 |
| `target_damping` | `float64[]` | 各ロボット関節の減衰。<br />通常は `target_stiffness` に対して調整し、ふらつきを防いで安定した動作を実現します。配列サイズは関節数と一致させてください。 |
| `target_feedforward_torque` | `float64[]` | 各ロボット関節に与える任意の追加トルク。<br />一定 force を与える接触タスクや、既知の tool と環境の相互作用を扱う場面で有用です。 |
| `trajectory_generation_mode` | `TrajectoryGenerationMode` | 目標値をどのように解釈するかを指定します。<br />`MODE_POSITION` は `target_state.positions` の値に従います。<br />`MODE_VELOCITY` は `target_state.velocities` の値に従います。 |

#### 例

[test_impedance.py](../aic_bringup/scripts/test_impedance.py) 内の `generate_joint_motion_update()` 関数を参照してください。

ROS 2 CLI を使って [`JointMotionUpdate`](../aic_interfaces/aic_control_interfaces/msg/JointMotionUpdate.msg) メッセージで joint position target を publish する例:
```bash
# joint target mode に切り替える service request を送信
ros2 service call /aic_controller/change_target_mode aic_control_interfaces/srv/ChangeTargetMode "{target_mode: {mode: 2}}"

# joint position target を送信
ros2 topic pub --once /aic_controller/joint_commands aic_control_interfaces/msg/JointMotionUpdate "{
  target_state: {
    positions: [0.0, -1.57, -1.57, -1.57, 1.57, 0]
  },
  target_stiffness: [85.0, 85.0, 85.0, 85.0, 85.0, 85.0],
  target_damping: [75.0, 75.0, 75.0, 75.0, 75.0, 75.0], trajectory_generation_mode: {mode: 2}
}"
```

同様に、joint velocity target の publish も簡単です。
```bash
# 以下のコマンドは、別の target で上書きされるまで、すべての関節を 0.025 rad/s で回転させます
ros2 topic pub --once /aic_controller/joint_commands aic_control_interfaces/msg/JointMotionUpdate "{
  target_state: {
    velocities: [0.025, 0.025, 0.025, 0.025, 0.025, 0.025]
  },
  target_stiffness: [85.0, 85.0, 85.0, 85.0, 85.0, 85.0],
  target_damping: [75.0, 75.0, 75.0, 75.0, 75.0, 75.0], trajectory_generation_mode: {mode: 1}
}"
```

## コントローラ設定パラメータ

`aic_controller` は ROS 2 parameter を使って、目標値の制限、平滑化、インピーダンス制御に関する値を設定します。これらは [aic_controller_parameters.yaml](../aic_controller/src/aic_controller_parameters.yaml) に、説明とデータ型とともに定義されています。

> **Note:** この設定は評価中には固定され、すべての参加者が同一のコントローラ設定を使用します。
