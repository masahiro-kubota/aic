# AIC MuJoCo Integration

この package は、AI for Industry Challenge（AIC）環境を MuJoCo に読み込むためのドキュメント、script、utility を提供します。

## Overview

[MuJoCo](https://mujoco.org/) は、ロボティクス、バイオメカニクス、グラフィックス、アニメーションの研究開発向けに設計された physics engine です。**Google DeepMind** との協力により、この integration では参加者は次を行えます。

- `sdformat_mjcf` を使って Gazebo の SDF world を MuJoCo の MJCF 形式へ変換する
- export した Gazebo world（`/tmp/aic.sdf`）から AIC task board と robot を読み込む
- camera image、joint state、FT sensor data にアクセスし、同じ ROS topic 経由で simulation robot を制御する
- Gazebo と MuJoCo の間で policy を変更せずにデータ収集や実行を行う

このガイドは、独立した 2 つの part に分かれています。

| | What | ROS 2 Control needed? |
|---|---|---|
| [**Part 1**](#part-1-building-the-mujoco-scene) | Gazebo から MJCF scene を生成し、MuJoCo で表示する | No |
| [**Part 2**](#part-2-mujoco-with-ros-2-control) | `ros2_control` 付きで scene を動かす（Gazebo と同じ controller interface） | Yes |

<a id="import-mujoco-dependencies"></a>
## MuJoCo Dependencies を import

ROS 2 workspace から、必要な MuJoCo repository をすべて import します。

```bash
cd ~/ws_aic/src
vcs import < aic/aic_utils/aic_mujoco/mujoco.repos
```

これにより以下が追加されます。
- `gz-mujoco`（`sdformat_mjcf` tool を含む）— Gazebo SDF file を MuJoCo MJCF 形式へ変換する
- `mujoco_vendor`（v0.0.6）— plugin（elasticity、actuator、sensor、SDF）と `simulate` binary を含む、MuJoCo 3.x 向け ROS 2 wrapper
- `mujoco_ros2_control` — MuJoCo と ros2_control の統合

---

<a id="part-1-building-the-mujoco-scene"></a>
## Part 1: MuJoCo Scene を構築する

このセクションでは、`ros2_control` を必要とせずに AIC scene を MuJoCo で生成・表示する方法を扱います。必要なのは `sdformat_mjcf` converter と MuJoCo viewer だけです。

### Prerequisites

#### 1. `sdformat_mjcf` Python Bindings を install

`sdf2mjcf` CLI tool は、`rosdep` では解決されない SDFormat と Gazebo Math の Python binding を必要とします。OSRF Gazebo apt repository から install してください。

```bash
# OSRF Gazebo stable apt repository を追加（未追加の場合）
sudo wget https://packages.osrfoundation.org/gazebo.gpg -O /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] http://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" \
  | sudo tee /etc/apt/sources.list.d/gazebo-stable.list > /dev/null
sudo apt update

# 必要な Python binding を install
sudo apt install -y python3-sdformat16 python3-gz-math9
```

binding が import 可能か確認します。

```bash
python3 -c "import sdformat; print('sdformat OK')"
python3 -c "from gz.math import Vector3d; print('gz.math OK')"
```

#### 2. Converter を build

`sdformat_mjcf` package を build します。

```bash
cd ~/ws_aic
source /opt/ros/kilted/setup.bash
colcon build --packages-select sdformat_mjcf
source install/setup.bash
```

### Scene Generation Workflow

#### 1. Gazebo から export

- 望む domain randomization parameter で `aic_gz_bringup` を起動します。たとえば:
```bash
ros2 launch aic_bringup aic_gz_bringup.launch.py spawn_task_board:=true spawn_cable:=true   cable_type:=sfp_sc_cable   attach_cable_to_gripper:=true   ground_truth:=true
```
- Gazebo は world を `/tmp/aic.sdf` に export します。

詳細は [Scene Description](../../scene_description.md) を参照してください。

#### 2. Export された SDF を修正

export された `/tmp/aic.sdf` には、変換前に修正が必要な既知の URI 破損問題が 2 つあります。

##### Issue 1: mesh URI 内の `<urdf-string>`

model が URDF string（`ros_gz_sim create -string`）から spawn された場合、SDFormat parser は file source として `<urdf-string>` という placeholder path を使います。world export 時、この情報が `file://<urdf-string>/model://...` のような形で mesh URI に漏れ出し、`<urdf-string>` が XML tag と解釈されるため XML parsing が壊れます。

```bash
# 破損した model:// URI を修正
sed -i 's|file://<urdf-string>/model://|model://|g' /tmp/aic.sdf
```

##### Issue 2: 壊れた relative mesh URI

一部の model（SC Plug、LC Plug、SFP Module）は relative mesh URI（例: `<uri>sc_plug_visual.glb</uri>`）を使っています。world export 時に model 相対の文脈が失われ、`file:///sc_plug_visual.glb` のような root-path URI になってしまい、存在しない file を指すようになります。

```bash
# 壊れた mesh URI を、aic_assets 内の実ファイルを指す形に修正
sed -i 's|file:///lc_plug_visual.glb|model://LC Plug/lc_plug_visual.glb|g' /tmp/aic.sdf
sed -i 's|file:///sc_plug_visual.glb|model://SC Plug/sc_plug_visual.glb|g' /tmp/aic.sdf
sed -i 's|file:///sfp_module_visual.glb|model://SFP Module/sfp_module_visual.glb|g' /tmp/aic.sdf
```

> **Note:** これらの問題は、string として parse された URDF と relative URI を world save 時に処理する SDFormat library 側の挙動に起因します。Gazebo から world を再 export するたびに発生します。

#### 3. SDF を MJCF に変換

- `sdf2mjcf` CLI tool を使って、修正済み `/tmp/aic.sdf` を MJCF 形式へ変換します。
  ```bash
  source ~/ws_aic/install/setup.bash
  mkdir -p ~/aic_mujoco_world
  sdf2mjcf /tmp/aic.sdf ~/aic_mujoco_world/aic_world.xml
  ```
- これにより、MJCF XML file と mesh asset が `~/aic_mujoco_world` に生成されます。

#### 4. MJCF File を整理


- MuJoCo が見つけられるよう、生成された mesh asset（`.obj` と `.png` file）は **常に** `~/aic_mujoco_world` から `mjcf` folder に copy または symlink する必要があります。
  ```bash
  cp ~/aic_mujoco_world/* ~/ws_aic/src/aic/aic_utils/aic_mujoco/mjcf
  ```

#### 5. 最終 MJCF File を生成

`sdformat_mjcf` converter は 1 つの巨大な monolithic MJCF file を生成します。`add_cable_plugin.py` script はこれを robot / world / scene file に分割・整形し、converter だけでは自動処理できない修正を加えます。

- **3 file に分割:** monolithic な `aic_world.xml` を `aic_robot.xml`（robot body、actuator、sensor）、`aic_world.xml`（environment、task board、cable）、`scene.xml`（両方を include する top-level file）に分割する
- **motor actuator を追加:** 6 本の UR5e joint と Robotiq gripper finger joint すべてに position-controlled actuator を挿入する
- **gripper mimic joint を追加:** 右 finger を左 finger に equality constraint で連動させる（重複する右 finger motor は削除）
- **FT sensor を追加:** `AtiForceTorqueSensor` site に force / torque sensor を取り付ける
- **`gripper_tcp` site を追加:** policy 用として、gripper 先端に tool-center-point site を挿入する
- **robot quaternion を修正:** robot link（例: `shoulder_link`, `upper_arm_link`, `wrist_*_link`）上の、ほぼ単位 quaternion やノイズを含む quaternion を綺麗な値に正規化する
- **camera を設定:** center / left / right camera に orientation（`quat`）、field of view（`fovy`）、resolution を追加する
- **cable plugin を設定:** `mujoco.elasticity.cable` を有効化し、twist / bend stiffness を設定し、joint damping を追加し、すべての cable body に plugin を適用する
- **cable link_1 を親付け変更:** `link_1` を `cable_end_0` から `cable_connection_0` に、計算した relative pose 付きで移動する（正しい cable attachment に必要）
- **cable physics を調整:** cable body の inertia を `0.01` から `1e-6` に下げ、`cable_connection_1`（SC plug 側）の inertia を `4e-4` に設定し、`joint_connection_end_0` に damping を追加し、`cable_end_0` を 5cm 持ち上げる
- **weld constraint を追加:** LC plug を `ati/tool_link` に対し、調整済み relative pose で weld する
- **contact exclusion を追加:** `tabletop`↔`shoulder_link`、gripper finger、`sc_port`↔`sc_plug`、`cable_end_0`↔`link_1` の self-collision を防ぐ
- **asset を分割配置:** keyword matching に基づいて mesh、material、texture を適切な file（robot 側または world 側）へ振り分ける

これを実行する際は、新しい terminal で ROS 2 workspace を source せずに実行してください（必要なら virtual env を使ってください）。

  ```bash
  cd ~/ws_aic/src/aic/aic_utils/aic_mujoco/
  python3 scripts/add_cable_plugin.py --input mjcf/aic_world.xml --output mjcf/aic_world.xml --robot_output mjcf/aic_robot.xml --scene_output mjcf/scene.xml
  cd ~/ws_aic && colcon build --packages-select aic_mujoco
  ```
  - `--input`: 初期 MJCF world file（通常は `aic_world.xml`）の path
  - `--output`: 最終 world file（`aic_world.xml`）の path
  - `--robot_output`: robot-only file（`aic_robot.xml`）の path
  - `--scene_output`: scene file（`scene.xml`）の path



#### 6. MuJoCo で表示

この時点で、ROS 2 control のセットアップなしに、生成した scene を MuJoCo で表示できます。

##### pixi environment を使う

Python viewer は既定で **paused mode** で開始します。Space を押すと simulation の開始 / 一時停止を切り替えられます。

```bash
# pixi shell に入る
pixi shell

# Option 1: 空の viewer を起動（その後 scene.xml を window に drag and drop）
python -m mujoco.viewer

# Option 2: 用意された convenience script を使う（paused で開始）
cd ~/ws_aic
python src/aic/aic_utils/aic_mujoco/scripts/view_scene.py ~/aic_mujoco_world/scene.xml

# Option 3: 1 行 Python command を使う（paused mode）
python -c "import mujoco, mujoco.viewer; m = mujoco.MjModel.from_xml_path('~/aic_mujoco_world/scene.xml'); d = mujoco.MjData(m); v = mujoco.viewer.launch_passive(m, d); v.sync(); exec('while v.is_running(): v.sync()')"
```

> **Tip:** viewer 内で Space を押すと開始 / 一時停止、Backspace で reset できます。

##### `simulate` binary を使う

> **Note:** `simulate` binary は `mujoco_vendor` により提供され、[Part 2](#part-2-mujoco-with-ros-2-control) で build されます。Part 2 をすでに完了している場合は、次の方法も使えます。

```bash
simulate ~/ws_aic/src/aic/aic_utils/aic_mujoco/mjcf/scene.xml
```

---

<a id="part-2-mujoco-with-ros-2-control"></a>
## Part 2: MuJoCo with ROS 2 Control

![](../../../../media/wave_arm_policy_mujoco.gif)

`ros2_control` と統合された MuJoCo を使うと、Gazebo と同じ `aic_controller` interface で UR5e ロボットを制御できます。これにより、policy code を simulator 非依存のまま保てます。

### Installation Steps

> **Note:** すでに上の [Import MuJoCo Dependencies](#import-mujoco-dependencies) で `mujoco.repos` から依存関係を import 済みなら、repository はすでに clone されています。そのまま以下の install / build 手順へ進んでください。

#### 1. Dependencies を install

MuJoCo package の依存関係を install します。

```bash
cd ~/ws_aic
rosdep install --from-paths src --ignore-src --rosdistro kilted -yr --skip-keys "gz-cmake3 DART libogre-dev libogre-next-2.3-dev"
```

#### 2. Workspace を build

```bash
cd ~/ws_aic
source /opt/ros/kilted/setup.bash

# すべての package（aic_mujoco を含む）を build
GZ_BUILD_FROM_SOURCE=1 colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release --merge-install --symlink-install --packages-ignore lerobot_robot_aic
```

#### 3. Install を確認

```bash
# workspace を source（未実行なら）
source ~/ws_aic/install/setup.bash

# environment hook により MUJOCO_DIR が自動設定されているか確認
echo $MUJOCO_DIR
# 次のような出力になるはず:
# /home/user/ws_aic/install/opt/mujoco_vendor

# MUJOCO_PLUGIN_PATH が設定されているか確認（MuJoCo が plugin を見つけるために必要）
echo $MUJOCO_PLUGIN_PATH
# 次のような出力になるはず:
# /home/user/ws_aic/install/opt/mujoco_vendor/lib

# MuJoCo install directory を確認
ls $MUJOCO_DIR
# 次が見えるはず: bin, include, lib, share, simulate directories

# plugin library が install されているか確認
ls $MUJOCO_DIR/lib/*.so
# 次が見えるはず: libelasticity.so, libactuator.so, libsensor.so, libsdf_plugin.so, libmujoco.so*

# MuJoCo simulate binary が動くか確認
which simulate
# 次のような出力になるはず:
# /home/user/ws_aic/install/opt/mujoco_vendor/bin/simulate
```

> **⚠️ Important:** 以前の MuJoCo install がある場合、`mujoco_vendor` と競合する可能性があります。build 前に shell 設定（`~/.bashrc`, `~/.zshrc` など）から既存の `MUJOCO_PATH`、`MUJOCO_PLUGIN_PATH`、`MUJOCO_DIR` environment variable を確認し、必要なら削除してください。環境を整理したら shell を再起動し、workspace を build し直します。
> ```bash
> # 競合する environment variable を確認
> env | grep MUJOCO
>
> # MUJOCO_PATH または MUJOCO_PLUGIN_PATH が別の場所を指していたら、
> # ~/.bashrc（または ~/.zshrc）から削除して shell を再起動
>
> # その後 mujoco_vendor を再 build
> cd ~/ws_aic
> colcon build --packages-select mujoco_vendor --cmake-clean-cache
> source install/setup.bash
>
> # 正しい MUJOCO_PLUGIN_PATH が設定されているか確認
> echo $MUJOCO_PLUGIN_PATH
> # 指しているべき値: /home/user/ws_aic/install/opt/mujoco_vendor/lib
> ```

### ros2_control 付きで MuJoCo を起動する

`aic_mujoco_bringup.launch.py` launch file は、Gazebo simulation と同じ controller を読み込んだ状態で、ros2_control 付き MuJoCo simulation を起動します。

#### 基本起動例

```bash
# terminal 1: まだ動いていなければ Zenoh router を起動
source ~/ws_aic/install/setup.bash
export RMW_IMPLEMENTATION=rmw_zenoh_cpp
export ZENOH_CONFIG_OVERRIDE='transport/shared_memory/enabled=true'
ros2 run rmw_zenoh_cpp rmw_zenohd
```

```bash
# terminal 2: ros2_control 付き MuJoCo simulation を起動
source ~/ws_aic/install/setup.bash
export RMW_IMPLEMENTATION=rmw_zenoh_cpp
export ZENOH_CONFIG_OVERRIDE='transport/shared_memory/enabled=true'
ros2 launch aic_mujoco aic_mujoco_bringup.launch.py
```

これでロボットは `aic_teleoperation` package を使って teleoperate できるようになります。詳しくは [teleoperation](../aic_teleoperation/README.md) を参照してください。Cartesian teleop の場合:

```bash
source ~/ws_aic/install/setup.bash
ros2 run aic_teleoperation cartesian_keyboard_teleop
```

`aic_example_policies` 内の任意の policy を使って MuJoCo 上のロボットを制御できます。詳しくは [example policies](../../aic_example_policies/README.md) を参照してください。

## Resources

- [MuJoCo Documentation](https://mujoco.readthedocs.io/)
- [mujoco_ros2_control GitHub](https://github.com/ros-controls/mujoco_ros2_control)
- [AIC Getting Started Guide](../../getting_started.md)
- [AIC Scene Description](../../scene_description.md)
