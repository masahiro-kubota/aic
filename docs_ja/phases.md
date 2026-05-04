# Competition Phases

<a id="qualification-phase-train-your-model"></a>
## Qualification Phase: model を学習する

qualification では、参加者は open source software や simulator を含む好みのツールを使いながら、Intrinsic challenge toolkit と組み合わせてケーブル挿入タスク用 model を学習します。提出された model はすべて Gazebo で評価されます。

![](../../media/qualification_overview.png)

### Technical Overview

この phase における主要な技術要件、セットアップ制約、trial 説明、評価指標を確認してください。完全な仕様は [Qualification Phase: Technical Overview](./qualification_phase.md) を参照してください。

<a id="key-steps-for-participation"></a>
### 実装ワークフロー

qualification を通過するには、参加者は [Challenge Rules](./challenge_rules.md) に定義された挙動要件に従う ROS 2 node を作成する必要があります。

#### 推奨アプローチ: `aic_model` framework を使う

利便性のため、ROS 2 lifecycle 管理と boilerplate を処理する `aic_model` framework を提供しています。参加者は Python policy class を実装するだけです。

1. **Policy Class を作成する:** [`Policy`](https://github.com/intrinsic-dev/aic/blob/main/aic_model/aic_model/policy.py) を継承した Python class を定義します。
2. **`insert_cable()` を実装する:** `aic_engine` が新しいタスクを要求したときに呼ばれる method です。observation data と、ロボット制御用の callable method を受け取ります。
3. **model をロードする:** class の instantiate 時に、学習済み policy（例: PyTorch checkpoint、ONNX model、control algorithm）を初期化します。
4. **observation を処理する:** 提供される `get_observation()` callback を使って、最大 20 Hz で sensor data を取得します。
5. **command を出力する:** `move_robot()` などの提供 method を使ってロボットへ指令します。
6. **完了時に return する:** task 完了時に `insert_cable()` method は return する必要があります。

> **Tutorial:** 手順ごとのガイドは [Creating a New Policy Node](./policy.md#tutorial-creating-a-new-policy-node) を参照してください。
>
> **Example:** 参考実装: [`WaveArm.py`](../aic_example_policies/aic_example_policies/ros/WaveArm.py)

#### 代替案: 自前の node を実装する

以下を満たす限り、独自の ROS 2 node を一から実装しても構いません。
- 名前が `aic_model` で、ROS 2 Lifecycle interface を実装している
- `/insert_cable` action server に応答する
- [Challenge Rules](./challenge_rules.md) の要件をすべて満たす

---

### 参加ガイドライン

* **Policy Development:** 参加者は次のような任意のアプローチで policy を開発できます。
    * 実環境でのテレオペレーションデータ
    * 任意の simulator（MuJoCo、Isaac Sim、O3DE など）での学習
    * 古典制御アルゴリズム
* **Interface Requirements:** policy（上記 service にラップされたもの）は、標準形式で world 情報を受け取り、action を出力する必要があります。
* **Evaluation:** 提供される Evaluator Simulator（Gazebo）が participant model の性能を採点します。
    * 開発中は、Evaluator Simulator をローカル実行して性能を確認できます。
    * 提出時には、クラウドインスタンスが同じ Evaluator Simulator を実行し、公式 score を記録します。

詳しくは次を参照してください。
* [Scene Description](./scene_description.md)
* [AIC Interfaces](./aic_interfaces.md)

---

## Phase 1: Flowstate で開発する

Phase 1 に進んだチームは **Intrinsic Flowstate**（開発環境）と **Intrinsic Vision Model** へアクセスできます。これらのツールを使って、学習済み model を組み込んだ完全なロボットケーブルハンドリングソリューションを構築します。

*Coming Soon*

## Phase 2: 実ロボットで動かす

Phase 2 の参加者は、Intrinsic 本社にある physical robotic workcell へソリューションを展開します。この phase では、実環境でソリューションを検証し、賞金受賞者を決定します。

*Coming Soon*
