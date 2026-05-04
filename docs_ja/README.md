# AI for Industry Challenge Toolkit

[![build](https://github.com/intrinsic-dev/aic/actions/workflows/build.yml/badge.svg)](https://github.com/intrinsic-dev/aic/actions/workflows/build.yml)
[![style](https://github.com/intrinsic-dev/aic/actions/workflows/style.yml/badge.svg)](https://github.com/intrinsic-dev/aic/actions/workflows/style.yml)

![](../../media/aic_banner.png)

**AI for Industry Challenge** は、robotics と manufacturing における特に難しく、かつ影響の大きい課題の解決を目指す開発者と roboticist のための open competition です。

この repository には、参加者が自分の solution 開発を始めるための公式 toolkit が含まれています。登録方法、公式ルール、FAQ については、[AI for Industry Challenge event page](https://www.intrinsic.ai/events/ai-for-industry-challenge) を参照してください。

---

<a id="toolkit-guide"></a>
## Toolkit Guide

AIC toolkit documentation へようこそ。この guide では、challenge 参加に必要な一連の流れを、要件の理解から solution の提出まで順番に案内します。

以下の section に沿って、各 phase を進めてください。

1. **📖 Challenge を理解する**
   - 目的を把握するために [Challenge Overview](./overview.md) を読む。
   - 何を作るのかを把握するために [Qualification Phase](./phases.md#qualification-phase-train-your-model) を確認する。
   - どのように採点されるかを把握するために [Scoring Guide](./scoring.md) を確認する。

2. **🔧 環境をセットアップする**
   - 開発環境のセットアップと検証のために [Getting Started](./getting_started.md) に従う。
   - evaluation container を起動し、Pixi を使って local workspace をセットアップする。

3. **💻 Policy を開発する**
   - 環境のカスタマイズ方法と探索方法を理解するために [Scene Description](./scene_description.md) を確認する。
   - sensor と actuator と通信するために利用できる interface を理解するために [AIC Interfaces](./aic_interfaces.md) を確認する。
   - robot の制御方法を理解するために [AIC Controller](./aic_controller.md) を確認する。
   - 要件順守を確認するために [Challenge Rules](./challenge_rules.md) を確認する。
   - solution 実装の開始点として [Policy Integration Guide](./policy.md) を確認する。
   - 役立つ tool 一覧として [Participant Utilities](./participant_utilities.md) を参照する。

4. **🧪 Solution をテストする**
   - 提供されている simulation environment を使って policy をテストする。
   - さまざまな scenario を試すために [`aic_engine/config/`](../aic_engine/config/) 内の `sample_config` で `aic_engine` を実行する。異なる config で `aic_engine` を実行する詳細は [aic_engine README file](./aic_engine/README.md) を参照する。
   - `aic_engine` と一緒に動かす独自の test scenario を作るには、[`aic_engine/config/`](../aic_engine/config/) の設定例に従う。
   - 問題が起きた場合は [Troubleshooting](./troubleshooting.md) を参照する。

5. **📦 エントリを提出する**
   - [Submission Guidelines](./submission.md) に従って solution を package 化する。
   - 提出前に [these instructions](./submission.md#verify-locally) に従って container を local でテストする。
   - [these instructions](./submission.md#2-upload-your-image-to-our-registry) に従って official portal から提出する。

---

<a id="toolkit-architecture"></a>
## Toolkit Architecture

![AIC Competition Components](../../media/aic_competition_components.png)

AI for Industry Challenge toolkit は **2 つの主要 component** に分かれています。

### 1. Evaluation Component（提供物 - 運営側が実行）

この component は、評価に必要な完全な infrastructure を提供します。
- **`aic_engine`** - trial の orchestration と score 計算を担当。
- **`aic_bringup`** - simulation environment（Gazebo、robot、sensor）を起動。
- **`aic_controller`** - force management を備えた low-level robot control。
- **`aic_adapter`** - sensor fusion と data synchronization。

**参加者が受け取るもの:** camera image、joint state、force/torque measurement、TF frame を提供する標準 ROS sensor topic。

### 2. Participant Model Component（実装対象 - 提出物）

こちらが参加者自身で開発して提出する component です。
- [Challenge Rules](./challenge_rules.md) で定義された挙動要件に従う **ROS 2 node**。
- sensor data を処理し、robot に cable insertion を行わせる **独自 logic**。

**参加者が提供するもの:** `/insert_cable` action に応答し、標準 ROS topic/service を通じて robot motion command を出力する `aic_model` という名前の ROS 2 Lifecycle node を含む container。

**便利な entry point:** ROS 2 boilerplate と lifecycle management を扱う `aic_model` framework を提供しています。参加者は runtime で動的に読み込まれる Python policy class を実装するだけで済みます。詳細は [Policy Integration Guide](./policy.md) を参照してください。

### Development and Submission Workflow

> [!IMPORTANT]
> **ROS 2 Distribution:** すべての提出物の公式評価は **ROS 2 Kilted Kaiju** を使って実施されます。別の ROS 2 distribution（例: Humble や Jazzy）で policy を開発またはテストすることもできますが、その場合の互換性確保とサポートはすべて参加者自身の責任です。**異なる distribution 間の通信は保証されず、公式にはサポートされません。**

**Development Options:**
- container 内で開発する（推奨 - 評価環境と一致）。
- または native Ubuntu 24.04 環境で開発する（すべての dependency が必要）。

**Submission Requirements:**
- 提供されている `aic_model` Dockerfile を使って solution を package 化する。
- container を提出する。この container は標準 ROS input に応答し、robot に cable insertion を実行させる必要がある。
- 参加者の container は ROS topic を通じて evaluation component と接続される。

---
## Repository Structure

```
aic/
├── aic_adapter/          # model と controller を接続する adapter
├── aic_assets/           # 3D model と simulation asset
├── aic_bringup/          # challenge environment を起動する launch file
├── aic_controller/       # robot controller 実装
├── aic_description/      # robot と environment の URDF/SDF description
├── aic_engine/           # trial orchestration と validation engine
├── aic_example_policies/ # 参考となる policy 実装
├── aic_gazebo/           # Gazebo 固有の plugin と設定
├── aic_interfaces/       # ROS 2 message、service、action 定義
├── aic_model/            # 参加者向け policy 実装 template
├── aic_scoring/          # scoring system 実装
├── aic_utils/            # utility package と tool
├── docker/               # Docker container 定義
└── docs/                 # 包括的な documentation
```

---

## Key Packages for Participants

### `aic_model` - 便利な Policy Framework（推奨）
この package は、Python policy 実装を動的に読み込んで実行する、すぐに使える ROS 2 Lifecycle node を提供します。ROS 2 boilerplate、lifecycle management、challenge rule への適合を処理するため、参加者は policy logic の実装に集中できます。
- **Location**: `aic_model/`
- **Documentation**: [Policy Integration Guide](./policy.md)
- **Tutorial**: [Creating a New Policy Node](./policy.md#tutorial-creating-a-new-policy-node)

> **Note:** この framework の利用を推奨しますが、[Challenge Rules](./challenge_rules.md) に従う限り、独自の ROS 2 node を一から実装しても構いません。

### `aic_interfaces` - 通信 protocol
challenge で使用されるすべての ROS 2 message、service、action を定義します。
- **Location**: `aic_interfaces/`
- **Documentation**: [AIC Interfaces](./aic_interfaces.md)

### `aic_example_policies` - 参考実装
異なる approach や technique を示す example policy 群です。
- **Location**: `aic_example_policies/`
- **README**: [aic_example_policies/README.md](./aic_example_policies/README.md)

### `aic_bringup` - 環境を起動する
simulation、robot、scoring system を起動する launch file 群です。
- **Location**: `aic_bringup/`
- **README**: [aic_bringup/README.md](./aic_bringup/README.md)

### `aic_engine` - Trial Orchestrator
trial 実行を管理し、participant model を検証し、scoring data を収集します。
- **Location**: `aic_engine/`
- **README**: [aic_engine/README.md](./aic_engine/README.md)

---

## Additional Documentation

### Challenge Information

* **[Challenge Overview](./overview.md):** competition の目的と構成の高レベルな概要。
* **[Competition Phases](./phases.md):** Qualification、Phase 1、Phase 2 の詳細。
* **[Qualification Phase](./qualification_phase.md):** qualification phase の trial と scoring に関する技術的な詳細。
* **[Challenge Rules](./challenge_rules.md):** participant model に求められる挙動要件。
* **[Scoring](./scoring.md):** performance 評価に使われる指標と方法。
* **[Scoring Test Examples](./scoring_tests.md):** 各 scoring tier を正確な command で再現する example。

### Technical Documentation

* **[Getting Started](./getting_started.md):** local 開発環境のセットアップ方法。
* **[Policy Integration](./policy.md):** `aic_model` framework 上で policy を実装するための guide。
* **[AIC Interfaces](./aic_interfaces.md):** policy から利用できる ROS 2 topic、service、action。
* **[AIC Controller](./aic_controller.md):** robot controller と motion command の理解。
* **[Scene Description](./scene_description.md):** simulation environment の技術仕様。
* **[Task Board Description](./task_board_description.md):** task board の physical layout と仕様。
* **[Troubleshooting](./troubleshooting.md):** よくある問題と debugging 方針。

### Reference Materials

* **[Glossary](./glossary.md):** AI for Industry Challenge 全体で使われる用語と定義。

### Submission

* **[Submission Guidelines](./submission.md):** final model の package 化と提出方法。

---


## Support and Resources

- **Discussions**: challenge に関する議論や質問は [Open Robotics Discourse](https://discourse.openrobotics.org/c/competitions/ai-for-industry-challenge/) を利用してください。community 全体で相互に助け合うことが推奨されています。
- **Issues**: bug や technical issue は [GitHub Issues](https://github.com/intrinsic-dev/aic/issues) で報告してください。challenge 全般に関する質問には Issue tracker を使わないでください。
  - **Note:** 新しい ticket を作る前に、[known issues](https://github.com/intrinsic-dev/aic/issues?q=is%3Aissue%20state%3Aopen%20label%3A%22known%20issue%22) と [bugs](https://github.com/intrinsic-dev/aic/issues?q=is%3Aissue%20state%3Aopen%20label%3Abug) の一覧を確認してください。
- **Event Page**: 公式の更新情報は [AI for Industry Challenge](https://www.intrinsic.ai/events/ai-for-industry-challenge) を参照してください。

---

## License

この project は Apache License 2.0 の下で提供されています。詳細は各 package 内の file を参照してください。
[`aic_isaac`](./aic_utils/aic_isaac/README.md) には BSD-3 で提供される file が含まれています。詳細は [aic_isaac/LICENSE](../aic_utils/aic_isaac/LICENSE) を参照してください。
