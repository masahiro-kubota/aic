# AI for Industry Challenge: ルールと技術仕様

## 1. 競技の精神
**AI for Industry Challenge (AIC)** は、現実世界のロボットマニピュレーション課題を解くうえで、創造性、技術的卓越性、そして協調を促進することを目的としています。私たちは、産業オートメーションにおける AI の可能性を示す革新的な解法を参加者に期待しています。

参加者には、**公式に提供されたインターフェースのみ**を使用して解法を構築することで、競技の精神を尊重することが求められます。対象となるのは [AIC Interfaces ドキュメント](./aic_interfaces.md) で定義された ROS 2 topic、service、message type です。目標は、汎化性が高く、堅牢で、実機システムおよびそのシミュレーションの制約に適合した AI モデルを構築することです。

---

## 2. 不正行為の禁止

### a. 提供されたインターフェースのみを使用すること
参加者は、不公平な優位を得るために提供されたインターフェースを迂回または回避してはなりません。禁止される行為には以下が含まれます。
* **Direct State Manipulation:** ロボット、タスクボード、またはシミュレーション環境の状態を手動で変更すること（例: コンポーネントのテレポートやケーブル挿入の強制）。
* **Backend & System Interference:** 公式 API を迂回して ROS や Gazebo の backend と通信すること。これには、以下を含む未承認の操作が含まれます。
	* **Configuration & Lifecycle:** 評価 node に対する ROS 2 parameter の変更や lifecycle state の変更。
	* **Simulation Control:** `/scoring`, `/gazebo`, `/gz_server` namespace にあるすべての topic / service。
	* **Entity Management:** モデルやエンティティを spawn、despawn、delete するための service。
	* **Environment States:** `/clock`, `/model`, `/world_stats` に関するデータ、physics 制御（`/pause_physics`）、または simulation reset。
* **Exploitative Hardcoding:** 特定のシミュレーション設定に関する知識を悪用するために、センサーデータや環境構成をハードコードすること。
* **Process Manipulation:** AIC Engine、スコアリングシステム、または logging infrastructure に干渉すること。

### b. 不正アクセス
参加者は、クラウドベースの評価基盤をリバースエンジニアリング、悪用、または改ざんしてはなりません。これには以下が含まれます。
* 未承認のツールを用いて評価環境へアクセスまたは変更すること。
* 悪意あるコード、バックドア、または競技アーキテクチャを無効化する仕組みを含むコンテナやモデルを提出すること。
* 提出ポータルやレジストリを迂回してデータを注入または改ざんすること。

### c. 外部データおよび情報漏えい
参加者は、公式インターフェース経由で取得できない評価環境の情報を使用してはなりません。
* **State Leaking:** **評価中に**、シミュレーションや backend システムの内部状態情報を再利用すること。学習時には、`/tf` topic から取得できる ground truth データを含め、すべての内部状態情報を使用して構いません。
* **Data Misuse:** toolkit の一部ではない評価基盤から生成されたデータを使ってモデルを学習すること。

---

## 3. 施行と報告

### 評価の完全性
評価プラットフォームには自動アクセス制御と完全性チェックが含まれています。加えて、上位チームの提出物については、以下の観点から手動レビューを行い、ルール準拠を確認します。
1. **Container Audits:** 提出された container image が、定義済みインターフェースのみを介してシステムとやり取りしていることを確認します。
2. **Behavioral Verification:** ライブ評価中のモデル挙動がルールに従っていることを確認します。
3. **Metric Analysis:** 想定されるベースラインと照合し、異常を検出するために性能指標をクロスチェックします。

> **Note:** これらのルールに違反していると判断されたチームには、失格や賞の取り消しを含むペナルティが科されます。

### 違反の報告
他の参加者がこれらのルールに違反していると疑われる場合は、公式チャネルを通じて主催者に報告してください。報告内容は機密として扱われます。

---

<a id="aic_model"></a>
## 4. 技術仕様: `aic_model`

- 提出する container は、名前が `aic_model` の ROS 2 Lifecycle node を持つプロセスを起動しなければなりません。
- `aic_model` node は以下の挙動を満たす必要があります。
  - 初期状態は lifecycle の `unconfigured` state であること。
    - `unconfigured` state では、特にロボットを動かすための topic を含め、node から topic を publish してはいけません。
  - `configured` state への遷移は 60 秒以内に成功しなければなりません。あらゆるモデルの読み込みはこの時間内に行う必要があります。
    - `configured` state では、特にロボットを動かすための topic を含め、node から topic を publish してはいけません。
    - `configured` state では、`/insert_cable` Action Server に送られた goal request を拒否しなければなりません。
  - `active` state への遷移は 60 秒以内に成功しなければなりません。
    - `active` state では、`/insert_cable` に送られた goal request を受け入れ、goal は cancel 可能でなければなりません。
    - goal request は、要求された [Task](../aic_interfaces/aic_task_interfaces/msg/Task.msg) の `time_limit` field 内に完了しなければなりません。評価実行時、この field は `aic_engine` に渡される [config](https://github.com/intrinsic-dev/aic/blob/main/aic_engine/config/sample_config.yaml) に指定された `time_limit` に基づいて設定されます。
  - `deactivate` transition request は、60 秒以内に node を `configured` state へ正常に戻さなければなりません。
  - `cleanup` transition request は、60 秒以内に node を `unconfigured` state へ正常に戻さなければなりません。
  - `shutdown` transition request は、60 秒以内に成功しなければなりません。
    - `shutdown` state では、特にロボットを動かすための topic を含め、node から topic を publish してはいけません。
    - `shutdown` state では、ロボットコマンド用の publisher が graph 上に存在してはいけません。
