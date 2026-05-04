# AIC インターフェース

この文書では、AI for Industry Challenge で参加者が利用できるすべてのインターフェースを定義します。標準 ROS 2 インターフェースと、このチャレンジ向けに新たに定義されたインターフェースの両方を含みます。

`aic_interfaces` フォルダには、ハードウェアと Insertion Policy をつなぐカスタム message / action 定義が含まれています。これらのインターフェースは、ロボットおよびタスク環境と連携するソリューションを開発するうえで重要です。

## インターフェース概要

このチャレンジでは、標準 ROS 2 インターフェースと [aic_interfaces](../aic_interfaces/) フォルダで定義されたカスタムインターフェースを組み合わせて使用します。

### 標準 ROS 2 インターフェース
- **[sensor_msgs/msg/Image](https://github.com/ros2/common_interfaces/blob/kilted/sensor_msgs/msg/Image.msg)** - カメラ画像データ
- **[sensor_msgs/msg/CameraInfo](https://github.com/ros2/common_interfaces/blob/kilted/sensor_msgs/msg/CameraInfo.msg)** - カメラのキャリブレーションデータ
- **[geometry_msgs/msg/WrenchStamped](https://github.com/ros2/common_interfaces/blob/kilted/geometry_msgs/msg/WrenchStamped.msg)** - Force/Torque sensor データ
- **[sensor_msgs/msg/JointState](https://github.com/ros2/common_interfaces/blob/kilted/sensor_msgs/msg/JointState.msg)** - 関節状態情報
- **[tf2_msgs/msg/TFMessage](https://github.com/ros2/geometry2/blob/kilted/tf2_msgs/msg/TFMessage.msg)** - 変換データ

### [aic_interfaces](../aic_interfaces/) で定義されたカスタムインターフェース
* **[aic_task_interfaces/action/InsertCable.action](../aic_interfaces/aic_task_interfaces/action/InsertCable.action)**
    * Insertion Policy にケーブル挿入タスクの実行を要求するための Action インターフェースです。
* **[aic_task_interfaces/msg/Task.msg](../aic_interfaces/aic_task_interfaces/msg/Task.msg)**
    * ケーブル挿入タスクの具体的なパラメータと状態を記述します。
* **[aic_control_interfaces/msg/MotionUpdate.msg](../aic_interfaces/aic_control_interfaces/msg/MotionUpdate.msg)**
    * Cartesian 空間制御向けの目標姿勢と関連許容値を記述します。
* **[aic_control_interfaces/msg/JointMotionUpdate.msg](../aic_interfaces/aic_control_interfaces/msg/JointMotionUpdate.msg)**
    * 関節空間制御向けの目標関節構成と関連許容値を記述します。
* **[aic_model_interfaces/msg/Observation.msg](../aic_interfaces/aic_model_interfaces/msg/Observation.msg)**
    * `aic_model` node が subscribe するワールド状態のスナップショットです。

---

## 入力

以下の topic は、モデルに対して感覚データと状態情報を提供します。

### センサートピック

| Topic | Message Type | 説明 |
| :--- | :--- | :--- |
| `/left_camera/image` | `sensor_msgs/msg/Image` | 左手首カメラからの rectified 画像データ。 |
| `/left_camera/camera_info` | `sensor_msgs/msg/CameraInfo` | 左手首カメラのキャリブレーションデータ。 |
| `/center_camera/image` | `sensor_msgs/msg/Image` | 中央手首カメラからの rectified 画像データ。 |
| `/center_camera/camera_info` | `sensor_msgs/msg/CameraInfo` | 中央手首カメラのキャリブレーションデータ。 |
| `/right_camera/image` | `sensor_msgs/msg/Image` | 右手首カメラからの rectified 画像データ。 |
| `/right_camera/camera_info` | `sensor_msgs/msg/CameraInfo` | 右手首カメラのキャリブレーションデータ。 |
| `/fts_broadcaster/wrench` | `geometry_msgs/msg/WrenchStamped` | Force/Torque sensor データ。 |
| `/joint_states` | `sensor_msgs/msg/JointState` | ロボット関節の現在状態。 |
| `/gripper_state` | `sensor_msgs/msg/JointState` | エンドエフェクタ / グリッパの現在状態。 |
| `/tf` | `tf2_msgs/msg/TFMessage` | 動的座標フレームの Transform データ。 |
| `/tf_static` | `tf2_msgs/msg/TFMessage` | 静的座標フレームの Transform データ。 |

### Action Server

| Action Name | Action Type | 説明 |
| :--- | :--- | :--- |
| `/insert_cable` | `aic_task_interfaces/action/InsertCable` | 自律挿入タスクの開始トリガ。 |

### コントローラトピック

以下の topic は、高頻度かつリアルタイムな状態テレメトリを提供し、監視やデバッグに利用できます。

| Topic | Message Type | 説明 |
| :--- | :--- | :--- |
| `/aic_controller/controller_state` | `aic_control_interfaces/msg/ControllerState` | 現在の TCP pose と velocity、参照 TCP pose、TCP の追従誤差、参照関節 effort に関するデータ。 |

---

## 出力

Insertion Policy は、以下の topic に publish することでロボットを制御します。

### コマンドトピック

| Topic | Message Type | 説明 |
| :--- | :--- | :--- |
| `/aic_controller/joint_commands` | `aic_control_interfaces/msg/JointMotionUpdate` | 関節空間制御向けの目標構成。 |
| `/aic_controller/pose_commands` | `aic_control_interfaces/msg/MotionUpdate` | Cartesian 空間制御向けの目標 pose。 |

> **Note:** コントローラは相互排他的なモードで動作します。たとえばコントローラが `Cartesian` ターゲットモードのときは `/aic_controller/pose_commands` topic のメッセージを処理し、`/aic_controller/joint_commands` からのメッセージは無視します。コントローラがその種類のコマンドを受け付ける前に、`/aic_controller/change_target_mode` service でアクティブなターゲットモードを設定する必要があります。

---

<a id="controller-configuration"></a>
## コントローラ設定

### サービス

| Service Name | Service Type | 説明 |
| :--- | :--- | :--- |
| `/aic_controller/change_target_mode` | `aic_control_interfaces/srv/ChangeTargetMode` | 期待する入力を定義するために、ターゲットモード（Cartesian または joint）を選択します。これに応じて、コントローラは `/aic_controller/pose_commands` または `/aic_controller/joint_commands` のいずれかを subscribe します。 |
| `/aic_controller/tare_force_torque_sensor` | `std_srvs/srv/Trigger` | Force/Torque sensor をゼロ点調整する service です。この service は評価中は無効になります。評価システムは、ケーブルが環境内に生成される前にこの service を自動で呼び出します。 |
