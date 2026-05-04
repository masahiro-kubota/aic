# Qualification Phase: 技術概要

**Qualification Phase** は、参加者が自分の policy によってロボットを制御し、target に収束し、異なる plug type 間で一般化できることを示す最初の段階です。この phase は完全に simulation 上で実施され、参加者 policy は提供される [Gazebo simulation environment](./scene_description.md) 内で評価されます。
また、**NVIDIA** および **Google Deepmind** との協力により、toolkit にはそれぞれ IsaacLab と MuJoCo の mirror simulation environment も含まれており、参加者はそれらを使って堅牢な policy を学習できます。

## シミュレーションについて

どの simulator も現実を完全には再現しません。
接触を多く含むプロセス（挿入など）では "Reality Gap" が強調されがちですが、私たちの目的は物理の完全一致ではなく、機能としての妥当性確認です。
物理差異に対して、私たちは次のように対応しています。
- **Signal over Precision**: Gazebo を使い、過度に特殊な挿入物理へ最適化するのではなく、意図したタスクを policy が正しく遂行しているかを確認します。
- **Tuned Environment**: ケーブル物理と挿入ダイナミクスをできるだけ近似するよう調整した Gazebo 環境を提供します。
- **Domain Randomization**: 実際、複数 simulator にまたがって学習することを推奨しています。こうした物理差異は domain randomization の良い機会となり、model を "sim-to-sim-to-real" 転移により適したものにします。

## 1. Phase のセットアップと制約

* **Task Scope:** 各 trial では 1 回のケーブル挿入だけを評価します。ケーブルの片側 plug のみが挿入対象で、反対側は自由で未接続のままです。評価中に扱われる plug-port の組み合わせは `SFP_MODULE` から `SFP_PORT`、および `SC_PLUG` から `SC_PORT` のみです。同じ柔軟ケーブルが全 trial で使われますが、task board の一般構成（例: NIC card や SC port の数と配置）は変化します。また、把持中の plug をどの component のどの port に挿入すべきかは task 定義で明示されます。この task 定義は [`aic_task_interfaces/msg/Task.msg`](../aic_interfaces/aic_task_interfaces/msg/Task.msg) message で記述され、ROS 2 action request を通じて participant model に渡されます。提供される Python template を使う場合、この `Task` object は `Policy.insert_cable` method の parameter として直接受け取れます。これらの component の 3D asset はすべて [`aic_assets/models`](../aic_assets/models/) directory にあります。未知の plug type や port type は出題されません。
* **Environment:** Flowstate を使わない Gazebo 上で評価されます。
* **Robot State:** ロボットは片側 plug をすでに把持した状態から開始します。
* **Grasp Pose:** 理想としては、plug と `gripper/tcp` frame の間に [sample_config.yaml](../aic_engine/config/sample_config.yaml) にある `gripper_offset` 値（例: SFP Module では `x: 0.0, y: 0.015385, z: 0.04245, roll: 0.4432, pitch: -0.4838, yaw: 1.3303`、SC plug では `x: 0.0, y: 0.015385, z: 0.04045, roll: 0.4432, pitch: -0.4838, yaw: 1.3303`）に近い一貫した把持を目指しますが、実際には把持相対 pose に小さなずれ（約 2mm、約 0.04 rad）が生じます。こうした微小差に対して頑健な policy を開発することを推奨します。
* **Proximity:** ロボットは挿入 target の数センチ以内から開始します。
* **Randomization:** [task board](./task_board_description.md) の pose、向き、および rail 上の各 component pose は各 trial ごとにランダム化されます。
* **Orchestration:** `aic_engine` node が trial lifecycle 全体を管理します。これには task board の生成、policy 挙動の検証、task 実行の監視、score data の収集が含まれます。engine の動作と設定の詳細は [AIC Engine README](./aic_engine/README.md) を参照してください。

## 2. Trial 説明

qualification phase は、参加者の policy の異なる側面を試すために設計された **3 つの specific trial** で構成されます。

参加者が提出した同一 policy が、3 つすべての trial に使われます。

各 trial では、ロボットは事前指定された（ランダムではない）pose で出現し、ケーブル plug は gripper に固定されています。

> [!NOTE]
> 最終評価時の正確な trial 数と順序は変更される可能性があります。ただし、常に以下に示す SFP 挿入と SC 挿入のいずれかの組み合わせになります。

### Trial 1 と 2: policy の妥当性と収束性

![TRIAL 1](../../media/aic_board_trial_1_sfp.png)

* **Objective:** policy の収束性と、ランダム化された NIC pose を扱う能力を検証します。この 2 つの trial の違いは、1) task board の pose、2) `NIC_CARD` がどの `NIC_RAIL` に生成されるか、3) その `NIC_RAIL` 上での `NIC_CARD` の平行移動および向きオフセット、のランダム性だけです。

* **Start State:**
	* ロボットは [sfp_sc_cable](../aic_assets/models/sfp_sc_cable/) の `SFP_MODULE` plug 側を把持しています。
	* task board はランダムな pose（位置と yaw angle）で生成されます。board 上には複数の component や NIC card が存在し得ますが、対象となる特定 port は常にロボット camera の view 内にあります。
	* 1 枚以上の `NIC_CARD` がランダムに選ばれた `NIC_RAIL`（`nic_rail_0` から `nic_rail_4` の 5 本）に取り付けられ、それぞれ rail に沿ったランダムな平行移動とランダムな yaw offset を持ちます。
	* ケーブルの反対側（SC plug）は自由で未接続のままです。

* **Manipulation Task:** 把持している `SFP_MODULE` plug を、生成された NIC card 上の `SFP_PORT_0` または `SFP_PORT_1` に挿入します（どちらかは `aic_engine` からの task config で指定されます）。

### Trial 3: 一般化性能（SC）

![TRIAL 3](../../media/aic_board_trial_3_sc.png)

* **Objective:** 異なる plug type と port type をまたいで policy が一般化できるかを検証します。

* **Start State:**
	* ロボットは同じ [sfp_sc_cable](../aic_assets/models/sfp_sc_cable/) の `SC_PLUG` 側を把持しています。
	* task board はランダムな pose（位置と yaw angle）で生成されます。複数の component や SC port が存在し得ますが、対象となる specific port は常にロボット camera の view 内にあります。
	* 一方または両方の SC port が task board に取り付けられます。`SC_PORT_0` は `SC_RAIL_0` に、`SC_PORT_1` は `SC_RAIL_1` に取り付けられ、それぞれ rail に沿ったランダムな平行移動を持ちます。target port になるのはそのうち 1 つだけです。
	* ケーブルの反対側（SFP module）は自由で未接続のままです。

* **Manipulation Task:** 把持している `SC_PLUG` を、`aic_engine` が指定する SC port（`SC_PORT_0` または `SC_PORT_1`）のいずれかに挿入し、task board の SC rail に対して適切に位置合わせを行います。


## 3. 評価指標とスコアリング

詳細は [Scoring](./scoring.md) を参照してください。

---

## 次のステップ

policy 実装と提出ワークフローの詳細については、Competition Phases 文書の [Key Steps for Participation](./phases.md#key-steps-for-participation) を参照してください。
