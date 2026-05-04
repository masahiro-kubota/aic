# シーン説明

![](../../media/aic_scene.png)

> [!NOTE]
> このガイドは、[Getting Started](./getting_started.md) を完了し、評価環境が起動していることを前提としています。

シミュレーション環境は [`aic_description`](./../aic_description) パッケージで定義されており、ロボット、タスクボード、およびケーブル挿入タスクに必要な各種オブジェクトで構成されています。シーン内の 3D モデルはすべて [`aic_assets`](./../aic_assets) パッケージに格納されています。

## シーン構成要素

### ロボット

このチャレンジでは **Universal Robots UR5e** ロボットアームを使用し、以下のハードウェアを搭載しています。
* **Gripper:** **Robotiq Hand-E**
* **Force-Torque Sensor:** **ATI AXIA80-M20**
* **Camera:** **Basler acA2440-20gc** と **Edmunds lens 58-000**（解像度: 1152x1024、フレームレート: 20 FPS）

* **Configuration:** ロボットの物理特性とセットアップは [`ur_gz.urdf.xacro`](../aic_description/urdf/ur_gz.urdf.xacro) で定義されています。
* **Control:** ロボットは `aic_controller` を介して操作されます。詳細なインターフェースと使用方法については、[AIC Controller ドキュメント](./aic_controller.md) を参照してください。

### タスクボード

チャレンジの中核となる要素は、[`task_board.urdf.xacro`](../aic_description/urdf/task_board.urdf.xacro) で定義されたタスクボードです。このモジュール式プラットフォームには、課題に必要な各種マウント、コネクタ、モジュールが搭載されます。

**主な構成要素:**
* **Connectors:** SC や SFP などの標準光ファイバコネクタ
* **NIC Cards:** ネットワークインターフェースカード
* **Mounts:** コネクタやモジュールを固定する専用治具

詳しい仕様は [Task Board Description](./task_board_description.md) を参照してください。

### 環境

照明、物理特性、ワールド全体のセットアップを含むグローバルなシミュレーション設定は [`aic.sdf`](../aic_description/world/aic.sdf) で定義されています。

---

## 環境の探索

基本環境を起動できたら、さまざまな構成を試してチャレンジへの理解を深め、多様な学習シナリオを作成できます。

> [!TIP]
> Gazebo での [シーンの移動方法](https://gazebosim.org/docs/latest/gui/#the-scene) も参照してください。

### 環境のカスタマイズ

launch コマンドにパラメータを渡すことで、シミュレーション環境をカスタマイズできます。**eval container** を使う場合でも **source build** を使う場合でも、指定するパラメータは同じです。

**eval container 内（distrobox 経由）:**
```bash
/entrypoint.sh [parameters]
```

**source build から:**
```bash
ros2 launch aic_bringup aic_gz_bringup.launch.py [parameters]
```

> [!TIP]
> 現在のターミナルが eval container の中かどうかを確認するには:
> ```bash
> echo $CONTAINER_ID  # 出力例: aic_eval
> ```

### 例: カスタムタスクボード構成

以下は、さまざまなコンポーネントを含むタスクボードを生成する完全な例です。

```bash
spawn_task_board:=true \
    task_board_x:=0.3 task_board_y:=-0.1 task_board_z:=1.2 \
    task_board_roll:=0.0 task_board_pitch:=0.0 task_board_yaw:=0.785 \
    sfp_mount_rail_0_present:=true sfp_mount_rail_0_translation:=-0.08 \
    sc_mount_rail_0_present:=true sc_mount_rail_0_translation:=-0.09 \
    nic_card_mount_0_present:=true nic_card_mount_0_translation:=0.005 \
    sc_port_0_present:=true sc_port_0_translation:=-0.04 \
    spawn_cable:=true cable_type:=sfp_sc_cable attach_cable_to_gripper:=true \
    ground_truth:=true start_aic_engine:=false
```

**探索時によく使う主なパラメータ:**
- `ground_truth:=true` - 開発中のデバッグを容易にするため、ground truth の TF フレームを有効にします
- `start_aic_engine:=false` - 試行の自動オーケストレーションを無効にし、自由に探索できるようにします
- `spawn_task_board:=true` - タスクボードを即座に生成します
- `spawn_cable:=true` - シーン内にケーブルを生成します
- `attach_cable_to_gripper:=true` - ケーブルをグリッパに取り付けます
- `cable_type:=sfp_sc_cable` - ケーブルの種類（選択肢: `sfp_sc_cable`, `sfp_sc_cable_reversed`）

設定可能な全パラメータ一覧は [aic_bringup README](./aic_bringup/README.md) を参照してください。

### 学習シナリオの作成

**パラメータを変化させることで多様な学習環境を生成**できます。

1. **異なる構成で起動**してランダム化されたシナリオを作成する
2. **エンティティ生成後、完全なワールド状態が自動的に** `/tmp/aic.sdf` に保存される
3. **複数シナリオを保存するためにファイルをコピーする:**
   ```bash
   cp /tmp/aic.sdf ~/training_scenarios/scenario_001.sdf
   ```
4. **IsaacLab や MuJoCo など他のシミュレータへ取り込む**

**ワークフロー例:**
```bash
# シナリオ 1: slot 2 に NIC card
/entrypoint.sh spawn_task_board:=true nic_card_mount_2_present:=true \
    spawn_cable:=true cable_type:=sfp_sc_cable ground_truth:=true start_aic_engine:=false
cp /tmp/aic.sdf ~/training_scenarios/nic_slot_2.sdf

# シナリオ 2: 右レール上の SC connector と異なる姿勢
/entrypoint.sh spawn_task_board:=true task_board_yaw:=1.57 \
    sc_mount_rail_1_present:=true spawn_cable:=true ground_truth:=true start_aic_engine:=false
cp /tmp/aic.sdf ~/training_scenarios/sc_right_rotated.sdf
```

### テレオペレーション

**関節空間または Cartesian 空間でロボットをテレオペレーション**して、以下を行えます。
- ワークスペースの探索
- ケーブル挿入の手動テスト
- ロボットの可動範囲と制約の理解
- ケーブル装着あり / なしの両方での練習

テレオペレーションを始める前に、このチャレンジで使われるコントローラを理解するため [AIC Controller ガイド](./aic_controller.md) を読むことを推奨します。

詳しい手順は [Robot Teleoperation Guide](./aic_utils/aic_teleoperation/README.md) を参照してください。

テレオペレーションで学習データを収集する場合は、各学習エピソードの開始時に Force/Torque Sensor のゼロ点調整を行ってください。詳しくは [学習前のゼロ点調整](#taring-before-training) を参照してください。

> [!TIP]
> 物体の近くでロボットが動かない場合、見た目では触れていなくても衝突状態になっていることがあります。物体の collision mesh を表示するには、その物体を右クリックして `View >` を開き、`Collisions` を選択してください。

---

## AI 学習向けのワールド状態エクスポート

このシミュレーションには、すべてのエンティティ（ロボット、タスクボード、ケーブル）が生成された後に、完全なワールド状態を自動エクスポートする world plugin が含まれています。この機能は、AI policy の学習や異なるシミュレータ間でのワークフローに特に有用です。

**主な利点:**
- **再現可能なシナリオ:** launch パラメータで作成したランダム構成を保存し、一貫した学習環境として再利用できます
- **クロスプラットフォーム互換性:** エクスポートした SDF ファイルを IsaacLab や MuJoCo など他のシミュレータに取り込めます
- **学習データ生成:** launch パラメータを変化させながら各構成をエクスポートし、多様な学習シナリオを作成できます

**エクスポート詳細:**
- **保存先の既定値:** `/tmp/aic.sdf`
- **Plugin 設定:** [`aic.sdf`](../aic_description/world/aic.sdf) 内で以下のパラメータを定義しています
  - `<save_world_path>`: ワールドファイルの保存先パス（既定値: `/tmp/aic.sdf`）
  - `<save_world_delay_s>`: エクスポート前に待機するシミュレーション秒数（既定値: `0.0`）

> [!NOTE]
> **MuJoCo Integration:** AIC 環境は MuJoCo へのエクスポートと、MuJoCo 上での policy 学習をネイティブにサポートしています。エクスポートしたシナリオは MJCF 形式へ変換でき、Gazebo で使用するものと同じ ROS 2 制御インターフェースで実行できます。シミュレーション設定、Gazebo ワールドの変換、MuJoCo での `ros2_control` 利用方法については [MuJoCo Integration Guide](./aic_utils/aic_mujoco/README.md) を参照してください。

> [!NOTE]
> **Isaac Lab Integration:** AIC 環境は NVIDIA の Isaac Lab に読み込んで、データ収集や学習に利用することもできます。詳細は [Isaac Lab Integration Guide](./aic_utils/aic_isaac/README.md) を参照してください。

---

<a id="taring-before-training"></a>
## 学習前のゼロ点調整

各学習エピソードの開始時（すなわち、テレオペレーション開始前かつ環境内にケーブルを生成する前）には、以下のサービス呼び出しで Force/Torque Sensor（F/T Sensor）のゼロ点調整を行ってください。
```bash
ros2 service call /aic_controller/tare_force_torque_sensor std_srvs/srv/Trigger
```

---

## 次のステップ

シーンを理解したら、次は以下を確認してください。

- **policy を開発する:** [Policy Integration Guide](./policy.md) を参照
- **インターフェースを理解する:** [AIC Interfaces](./aic_interfaces.md) を確認
- **スコアリングを学ぶ:** [Scoring](./scoring.md) を読む
- **サンプル policy を確認する:** [`aic_example_policies`](./aic_example_policies/README.md) を参照
