# 参加者向けユーティリティ

## テレオペレーション

### aic_teleoperation

- [aic_teleoperation](./aic_utils/aic_teleoperation/README.md): joint 空間および Cartesian 空間制御向けのキーボードベーステレオペレーション

### lerobot_robot_aic

- [lerobot_robot_aic](./aic_utils/lerobot_robot_aic/README.md#teleoperating-with-lerobot): LeRobot ベースのテレオペレーション（`lerobot-teleoperate`）。joint 空間および Cartesian 空間制御に対応し、キーボードまたは SpaceMouse デバイスを使用できます
- `lerobot-record` を使用したデータセット記録に対応しており、LeRobot policy の学習に利用できます

### 追加例

- 指定した pose や joint 構成へロボットを動かす例: [test_impedance.py](../aic_bringup/scripts/test_impedance.py), [home_robot.py](../aic_bringup/scripts/home_robot.py)

## LeRobot によるデータ収集と学習

- [lerobot_robot_aic](./aic_utils/lerobot_robot_aic/README.md#recording-training-data): AIC と [LeRobot](https://huggingface.co/lerobot) の統合。LeRobot を使ったテレオペレーションとデータセット記録を可能にします

## プロット / 可視化

- [PlotJuggler](https://github.com/facontidavide/PlotJuggler): ROS topic の時系列データを可視化するためのツール

## RViz

- [RViz](https://docs.ros.org/en/kilted/Tutorials/Intermediate/RViz/RViz-User-Guide/RViz-User-Guide.html) は ROS 2 用の可視化ツールです。提供されている RViz 設定ファイル（`aic.rviz`）は帯域幅の都合から中央カメラのストリームのみを表示しますが、必要に応じて残り 2 台のカメラ表示を追加すると役立つ場合があります。

## ROS 2 CLI ツール

ROS 2 には、システムの introspection とデバッグのための包括的なコマンドラインツール群があります。

- **[ROS 2 Beginner CLI Tools](https://docs.ros.org/en/rolling/Tutorials/Beginner-CLI-Tools.html)**: 次の重要なチュートリアルを含みます
  - `ros2 node` - 実行中の node を一覧表示し、調査する
  - `ros2 topic` - topic を確認し、message を echo し、publish レートを監視する
  - `ros2 service` - service を呼び出し、service type を確認する
  - `ros2 param` - node parameter を取得 / 設定する
  - `ros2 action` - action と対話する
  - `ros2 bag` - データを記録 / 再生する
  - `ros2 launch` - 複数 node を起動する
  - `ros2 interface` - message / service / action type を調べる

**簡単な例:**
```bash
# アクティブな node をすべて一覧表示
ros2 node list

# topic を echo
ros2 topic echo /aic_controller/state

# node parameter を取得
ros2 param list /aic_controller

# データを bag file に記録
ros2 bag record -o my_recording /aic_controller/state /camera/image
```
