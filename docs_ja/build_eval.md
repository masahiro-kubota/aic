# Evaluation Component を source から build する

このガイドは、提供されている Docker container を使わず、Ubuntu 24.04 上で Evaluation Component をローカル build したい上級ユーザー向けです。

> [!NOTE]
> **ほとんどのユーザーには、[Getting Started](./getting_started.md) で説明されているビルド済み `aic_eval` Docker container の利用を推奨します。** この container は、公式評価環境に一致する、一貫したテスト済み環境を提供します。

## なぜ source から build するのか

ローカル build が有用な場合:
- Evaluation Component を変更またはデバッグしたい
- container を使わず native に開発したい
- ホストシステム上の他ツールと統合したい

> [!IMPORTANT]
> Evaluation Component への変更は公式評価には **反映されません**。評価されるのは、container として提出した Participant Model だけです。

---

## 前提条件

| Dependency | Release / Distro |
| ---------- | ------- |
| Operating System | [Ubuntu 24.04 (Noble Numbat)](https://releases.ubuntu.com/noble/) |
| ROS 2 | [ROS 2 Kilted Kaiju](https://docs.ros.org/en/kilted/Installation/Ubuntu-Install-Debs.html) |

---

## セットアップ手順

現在のシステムに ROS 2 Kilted と Gazebo の binary package がインストールされている場合は、source install を始める前にそれらを削除する必要があります。特定の repository を source から build する際に、事前インストール済み binary があると環境競合が起きるためです。

関連する binary を削除するには、次のコマンドを実行します。

```bash
sudo apt purge ros-kilted-ros2-control* ros-kilted-control* ros-kilted-kinematics* ros-kilted-joint-state-publisher ros-kilted-realtime-tools ros-kilted-gz*
```

### 1. Gazebo repository を追加

```bash
sudo curl https://packages.osrfoundation.org/gazebo.gpg --output /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] http://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/gazebo-stable.list > /dev/null
sudo apt-get update
```

### 2. workspace を clone して build

```bash
# workspace を作成
sudo apt update && sudo apt upgrade -y
mkdir -p ~/ws_aic/src
cd ~/ws_aic/src

# repository を clone
git clone https://github.com/intrinsic-dev/aic

# 依存関係を import
vcs import . < aic/aic.repos --recursive

# Gazebo の依存関係を install
sudo apt -y install $(sort -u $(find . -iname 'packages-'`lsb_release -cs`'.apt' -o -iname 'packages.apt' | grep -v '/\.git/') | sed '/gz\|sdf/d' | tr '\n' ' ')

# ROS 2 の依存関係を install
cd ~/ws_aic
sudo rosdep init  # rosdep を初めて実行する場合のみ
rosdep install --from-paths src --ignore-src --rosdistro kilted -yr --skip-keys "gz-cmake3 DART libogre-dev libogre-next-2.3-dev rosetta"

# rmw_zenoh_cpp middleware と追加依存関係を install
sudo apt install -y ros-kilted-rmw-zenoh-cpp python3-pynput

# workspace を build
source /opt/ros/kilted/setup.bash
GZ_BUILD_FROM_SOURCE=1 colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release --merge-install --symlink-install --packages-ignore lerobot_robot_aic
```

### 3. 環境を設定

以下の environment variable をシェル設定ファイル（例: `~/.bashrc`）に追加します。

```bash
export RMW_IMPLEMENTATION=rmw_zenoh_cpp
export ZENOH_CONFIG_OVERRIDE='transport/shared_memory/enabled=true;transport/shared_memory/transport_optimization/pool_size=536870912'
```

その後、シェル設定を再読み込みします。
```bash
source ~/.bashrc
```

> [!NOTE]
> この challenge では ROS 2 middleware として [rmw_zenoh](https://github.com/ros2/rmw_zenoh) を使用します。すべてのターミナルで `RMW_IMPLEMENTATION` environment variable を `rmw_zenoh_cpp` に設定する必要があります。

---

## システムの実行

3 つのターミナルが必要です。各ターミナルで workspace を source してください。

```bash
source ~/ws_aic/install/setup.bash
```

> [!TIP]
> 手順 3 で environment variable を `~/.bashrc` に追加していない場合は、各ターミナルで export する必要があります。
> ```bash
> export RMW_IMPLEMENTATION=rmw_zenoh_cpp
> export ZENOH_CONFIG_OVERRIDE='transport/shared_memory/enabled=true;transport/shared_memory/transport_optimization/pool_size=536870912'
> ```

そのうえで、各ターミナルで次のコマンドを実行します。

### Terminal 1 - Zenoh router を起動

```bash
ros2 run rmw_zenoh_cpp rmw_zenohd
```

### Terminal 2 - 評価環境を起動

```bash
ros2 launch aic_bringup aic_gz_bringup.launch.py ground_truth:=false start_aic_engine:=true
```

これにより Gazebo がロボットアームとエンドオブアーム tooling とともに起動します。`TaskBoard` と `Cable` は、モデルの準備ができた時点で `aic_engine` によって生成されます。

### Terminal 3 - あなたの policy を実行

```bash
ros2 run aic_model aic_model --ros-args -p use_sim_time:=true -p policy:=aic_example_policies.ros.WaveArm
```

`aic_example_policies.ros.WaveArm` を自分の policy 実装に置き換えてください。

---

## 次のステップ

ローカルで評価環境を起動できたら、次に以下を確認してください。

- [Scene Description](./scene_description.md) を確認し、環境のカスタマイズや探索方法を理解する
- [Policy Integration Guide](./policy.md) を読み、自分の policy node の作り方を理解する
- 参考実装として [`aic_example_policies`](./aic_example_policies/README.md) を確認する
- [AIC Interfaces](./aic_interfaces.md) を見て、利用可能なセンサーやアクチュエータを理解する
- [AIC Controller](./aic_controller.md) を読んで、モーションコマンドを理解する
- [Scoring Test Examples](./scoring_tests.md) を実行し、各ベースライン policy の想定結果を確認する

---

## トラブルシューティング

問題が発生した場合:

1. **Build Errors**: すべての依存関係が正しくインストールされているか確認する
2. **Runtime Issues**: environment variable がすべてのターミナルで設定されているか確認する
3. **ROS 2 Communication**: Zenoh router が起動しており、middleware が設定されているか確認する

さらに必要であれば [Troubleshooting](./troubleshooting.md) を参照するか、[GitHub](https://github.com/intrinsic-dev/aic/issues) で issue を報告してください。
