# Policy の統合

コンピューティングの多くの概念と同様に、AI における _model_ や _policy_ という用語は文脈によって意味が異なります。以下の図は、AI for Industry Challenge のソフトウェアブロックにおいて、これらの用語がどのように使われているかを示しています。

![Block diagram](../../media/aic_policy_diagram.png)

_policy_ とは、センサーデータを受け取り、ロボットへの出力コマンドを生成するソフトウェアです。_policy_ の作成は AI for Industry Challenge の中心であり、センサーとアクチュエータの間で「ループを閉じる」重要なブロックです。

より具体的には、_policy_ は最大 20 Hz で以下のデータを受け取れます。
 * :camera: :camera: :camera: ロボット手首に取り付けられた 3 台のカメラからの画像
 * :mechanical_arm: ロボットアームとグリッパの関節角度
 * :balance_scale: ロボット手首における 3 次元の force と 3 次元の torque の計測値
 * :triangular_ruler: グリッパフィンガー tool center point（TCP）の目標 pose と実 pose
 * :comet: グリッパフィンガー tool center point（TCP）の velocity

利便性のため、Challenge 環境の `aic_adapter` は、センサースイートの時刻同期済みの値を 1 つの複合的な `Observation` データ構造にまとめ、それを 20 Hz で `aic_model` ブロックへ渡します。ユーザー定義の `policy` は実行時に `aic_model` へ動的に読み込まれ、いつでも最新の `Observation` を取得できます。

policy は `aic_controller` に対して position または velocity の target を発行する責務を持ちます。`aic_controller` はアームの低レベル制御を担い、接触力を管理します。target は任意のレートで `aic_controller` に送れます。

policy を実装する際には複数の API スタイルが考えられます。この Challenge は ROS ベースで実装されているため、もっとも単純な API は ROS 2 Python client library である `rclpy` が生成するデータ構造を使う方法です。次のセクションではその形を示します。

<a id="ros-policy-api"></a>
## Policy API

`geometry_msgs.msg.Pose`、`sensor_msgs.msg.Image` などの ROS データ構造を使って policy を統合するには、次を行います。
 * [`aic_model.Policy`](https://github.com/intrinsic-dev/aic/blob/main/aic_model/aic_model/policy.py) を継承した Python class を定義する
 * `aic_engine` が新しいタスクを要求したときに呼ばれる [`insert_cable()`](https://github.com/intrinsic-dev/aic/blob/main/aic_model/aic_model/policy.py#L49) method を実装する
 * 実行時に、この Python class 名を `aic_model` の parameter として渡す

`insert_cable()` 関数は、引数としていくつかの `Callable` method を受け取ります。
 * `get_observation()` は、最新の [`Observation`](https://github.com/intrinsic-dev/aic/blob/main/aic_interfaces/aic_model_interfaces/msg/Observation.msg) を ROS message として返します。この message は複数の ROS submessage で構成されます。
   * [`sensor_msgs/Image left_image`](https://github.com/ros2/common_interfaces/blob/kilted/sensor_msgs/msg/Image.msg)（および `center_image`, `right_image`）
   * [`sensor_msgs/CameraInfo left_camera_info`](https://github.com/ros2/common_interfaces/blob/kilted/sensor_msgs/msg/CameraInfo.msg)（および `center_camera_info`, `right_camera_info`）
   * [`sensor_msgs/JointState joint_states`](https://github.com/ros2/common_interfaces/blob/kilted/sensor_msgs/msg/JointState.msg)
   * [`geometry_msgs/WrenchStamped wrist_wrench`](https://github.com/ros2/common_interfaces/blob/kilted/geometry_msgs/msg/WrenchStamped.msg)
   * [`aic_control_interfaces/ControllerState controller_state`](https://github.com/intrinsic-dev/aic/blob/main/aic_interfaces/aic_control_interfaces/msg/ControllerState.msg)
 * `move_robot()` は、`MotionUpdate` または `JointMotionUpdate` message をロボットアームコントローラへ送ります。
 * `send_feedback()` は、`string` を `InsertCable` action の [feedback](https://docs.ros.org/en/kilted/Tutorials/Intermediate/Creating-an-Action.html#defining-an-action) message として publish します。これはデバッグに役立ちます。

_policy_ は、ロボットへ動作コマンドを発行する API 関数を呼び出せます。実装上は、これらの API 関数は `aic_model` ROS node を使って `aic_controller` へデータを publish します。`aic_controller` は [`ros2_control`](https://control.ros.org/rolling/index.html) framework 上に実装されています。

<a id="baseline-policies"></a>
## ベースライン Policy

このケーブル挿入タスクへの異なるアプローチを示すため、[`aic_example_policies`](./aic_example_policies/README.md) にいくつかのベースライン policy 実装を用意しています。

- **WaveArm** - 基本的な Policy API 構造を示す最小限の例
- **CheatCode** - 学習やデバッグのために ground truth データを使う「チート」policy
- **RunACT** - ACT（Action Chunking with Transformers）policy 実装

詳細な説明、使用方法、ソースコードについては [Example Policies README](./aic_example_policies/README.md) を参照してください。

各ベースライン policy の想定スコア結果については [Scoring Test & Evaluation Guide](./scoring_tests.md) を参照してください。

<a id="tutorial-creating-a-new-policy-node"></a>
## チュートリアル: 新しい policy node の作成

policy node は本質的には、observation を subscribe し、実行すべき action を publish する ROS 2 node です。

このチュートリアルでは、policy node の実装に [aic_model](../aic_model/) を使用します。

> [!Important]
> bash の例にあるプロンプト表記に注意してください。`(aic) $` で始まる場合は、pixi 環境の中で実行する必要があります。
>
> 例:
> ```bash
> $ pixi shell # これは pixi 環境の外です
> (aic) $ ros2 pkg list # これは環境の中です
> ```

### 新しい ROS 2 package を作成する

```bash
# "pixi shell" を実行して pixi 環境に入る
(aic) $ ros2 pkg create my_policy_node --build-type ament_python
```

### AIC 依存関係を追加する

以下を `package.xml` に追加してください。
```xml
	<depend>aic_control_interfaces</depend>
	<depend>aic_model</depend>
	<depend>aic_model_interfaces</depend>
	<depend>aic_task_interfaces</depend>
	<depend>geometry_msgs</depend>
	<depend>rclpy</depend>
	<depend>sensor_msgs</depend>
	<depend>std_srvs</depend>
	<depend>trajectory_msgs</depend>
```

### pixi package を作成する

`my_policy_node` package のディレクトリに `pixi.toml` を作成し、次の内容を記述します。

```toml
[package.build.backend]
name = "pixi-build-ros"
version = "==0.3.3.20260113.c8b6a54"
channels = [
	"https://prefix.dev/pixi-build-backends",
	"robostack-kilted",
	"conda-forge",
]

[package.host-dependencies]
ros-kilted-aic-control-interfaces = { path = "../aic_interfaces/aic_control_interfaces" }
ros-kilted-aic-model = { path = "../aic_model" }
ros-kilted-aic-model-interfaces = { path = "../aic_interfaces/aic_model_interfaces" }
ros-kilted-aic-task-interfaces = { path = "../aic_interfaces/aic_task_interfaces" }

[package.build-dependencies]
ros-kilted-aic-control-interfaces = { path = "../aic_interfaces/aic_control_interfaces" }
ros-kilted-aic-model = { path = "../aic_model" }
ros-kilted-aic-model-interfaces = { path = "../aic_interfaces/aic_model_interfaces" }
ros-kilted-aic-task-interfaces = { path = "../aic_interfaces/aic_task_interfaces" }
```

> [!Tip]
> 通常、pixi は `package.xml` から依存関係を自動検出します。ただし、ここでは aic interfaces を source から build するため、それらの場所を pixi に明示する必要があります。

### workspace に pixi package を追加する

root の `pixi.toml` の `[dependencies]` に、新しい package を追加します。

次のようになります。

```toml
[dependencies]
# ...
ros-kilted-my-policy-node = { path = "my_policy_node" }
```

### `PolicyRos` を実装する

簡潔にするため、ここでは `aic_example_policies` のコードを再利用します。実装詳細は上記の [ROS Policy API](#ros-policy-api) セクションを参照してください。

```bash
(aic) $ cp aic_example_policies/aic_example_policies/ros/WaveArm.py my_policy_node/my_policy_node/WaveArm.py
```

### policy node をテストする

Terminal 1:
```bash
# 'export DBX_CONTAINER_MANAGER=docker' を実行しておくこと
$ distrobox enter -r aic_eval -- /entrypoint.sh
```

Terminal 2:
```bash
$ pixi reinstall ros-kilted-my-policy-node
$ pixi run ros2 run aic_model aic_model --ros-args -p use_sim_time:=true -p policy:=my_policy_node.WaveArm
```

> [!Note]
> 上のコマンドは `aic_model` node を起動し、その node が特定の policy 実装（`my_policy_node.WaveArm`）を動的に読み込んで実行します。

### 依存関係の管理

これは、pixi workspace で依存関係を管理するための簡単なガイドです。一般的な ROS workspace と異なり、system 依存は使用せず、すべての依存関係は conda または pypi から取得する必要があります。

#### 依存関係の追加

##### ROS 依存関係

pixi workspace は robostack-kilted channel を使うよう設定されています。そのため、多くの ROS package は pixi でインストールできます。

```bash
$ pixi add ros-kilted-ros-core
```

##### PyPI 依存関係

native ROS workspace と違い、pixi workspace では ROS 依存関係と pypi 依存関係を混在させられます。

```bash
$ pixi add --pypi torch
```

##### ローカル依存関係

同じ workspace 内のローカル依存関係を package が必要とする場合、それらは package 側の `pixi.toml` と root の `pixi.toml` の両方に宣言する必要があります。

たとえば `my_policy_node` が `my_local_dep` を必要とする場合:

`my_policy_node/pixi.toml`:
```toml
[package.host-dependencies]
# ...
ros-kilted-my-local-dep = { path = "../my_local_dep" }

[package.build-dependencies]
# ...
ros-kilted-my-local-dep = { path = "../my_local_dep" }
```

`pixi.toml`:
```toml
[dependencies]
# ...
ros-kilted-my-policy-node = { path = "my_policy_node" }
ros-kilted-my-local-dep = { path = "my_local_dep" }
```

> [!Tip]
> pixi は ROS package に自動的に `ros-<distro>-` を接頭辞として付与し、underscore を hyphen に変換します。

### Build-Run-Debug サイクル (Python)

> [!IMPORTANT]
> Pixi 環境内の package 変更は自動では追跡されません。変更を反映するには、`pixi reinstall <package_name>` を実行する必要があります。

```bash
$ pixi reinstall <package>
```

> [!Tip]
> `pixi shell` で pixi 環境に入り、`pip install -e` を使って「editable」install を強制することもできます。ただし、これは pixi の管理を迂回するため、意図しない副作用を招く可能性があります。

### 提出の準備

policy に満足できたら、提出用の Docker image を準備する必要があります。詳細は [Submission](./submission.md) を参照してください。

### まとめ

おめでとうございます。これで policy node の作成、テスト、パッケージ化が完了しました。

このチュートリアルでは、以下を学びました。
- policy node 用の新しい ROS 2 package を作成し、セットアップする方法
- `pixi` workspace 内で Python および ROS の依存関係を管理する方法
- policy 開発における build、run、debug のサイクル
- 提出用の policy のために Docker image を準備する方法

これで、AI for Industry Challenge 向けに独自の policy を開発、テスト、提出するための基礎スキルが揃いました。より高度な概念や着想を得るために、提供されている example policy や他のドキュメントもぜひ確認してください。
