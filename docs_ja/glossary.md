# AIC Toolkit 用語集

## コアアーキテクチャ用語

**Adapter (aic_adapter)**
- ロボットハードウェアと参加者の policy 実装の間で、sensor fusion とデータ同期を扱うブリッジコンポーネントです。複数ソースのセンサーデータを統合し、一貫した observation にまとめます。

**Evaluation Component**
- challenge 主催者が提供する基盤側コンポーネントです。orchestration engine、launch system、robot controller、sensor pipeline を含みます。参加者はこのコンポーネントを変更しません。

**Participant Model Component (aic_model)**
- challenge 参加者が開発する policy 実装です。ここで独自ロジックがセンサーデータを処理し、ロボットへコマンドを送ってケーブル挿入タスクを実行します。

**Policy**
- 参加者が実装するアルゴリズムまたは AI model で、sensor observation を処理してロボットの motion command を生成します。container 化されたソリューションとして提出されます。

## タスクと環境

**Cable Insertion Task**
- challenge の主目的です。force sensing 機能を持つロボットアームを使って、光ファイバケーブルを自律的に配線し、network interface card に挿入します。qualification trial では、ケーブルの片側 plug だけが挿入対象で、もう片側は自由で未接続のままです。

**Task Board**
- ケーブル挿入 challenge が行われるモジュール式・再構成可能なプラットフォームです。4 つの機能ゾーンに分かれ、assembly target（Zone 1 & 2）と pick location（Zone 3 & 4）を持ち、部品の位置と向きはランダム化されます。

**Zone (Task Board Zones)**
- タスクボード上の 4 つの機能エリアで、電子機器組立ワークフロー全体を模擬します。
  - **Zone 1:** SFP port を備えた Network Interface Card（NIC）。データリンクが接続されるサーバ compute tray を表現します
  - **Zone 2:** SC optical port。サーバラックの optical patch panel または backplane を模擬します
  - **Zone 3 & 4:** Pick location。SFP module、LC plug、SC plug を adjustable mount 上に並べる整理された供給エリアです

**Gazebo**
- 評価および学習に使用される主たるシミュレーション環境です。ロボットとケーブル挿入タスクの現実的な物理シミュレーションを提供します。ワールド状態は `/tmp/aic.sdf` にエクスポートでき、他の simulator でも利用できます。

## コネクタ種類と構成部品

**SFP (Small Form-factor Pluggable)**
- Network Interface Card で使用される transceiver module の種類です。ケーブルの SFP module 側が、Zone 1 の NIC card 上の SFP port に挿入されます。

**LC (Lucent Connector)**
- 小型 form factor の光ファイバコネクタです。LC plug は Zone 3 と 4 の pick location に配置され、ケーブル組立タスクに使われます。

**SC (Subscriber Connector)**
- LC より大きい form factor を持つ光ファイバコネクタです。SC plug は Zone 2 の SC port に挿入され、optical patch panel 接続を模擬します。

**NIC (Network Interface Card)**
- Zone 1 の主要な挿入対象となる SFP port を持つ network hardware 部品です。最大 5 枚の dual-port NIC を adjustable rail に取り付けられます。

**Port**
- タスクボード上の挿入対象です。種類には SFP port（NIC card 上）と SC port（optical panel 上）があります。各 port への挿入には高精度な位置合わせが必要です。

**Plug**
- 操作され、挿入されるケーブル端のコネクタです。種類には SFP module、LC plug、SC plug があります。挿入中、ロボットは片側の plug を把持し、もう片側は自由なままです。

**Rails**
- タスクボード上の可動式 mount track で、部品をランダム位置にスライドさせられます。
  - **NIC Rails**（合計 5 本）: network interface card を保持し、平行移動範囲 [0, 0.062] meters、回転範囲 [-10, +10] degrees
  - **SC Rails**（合計 2 本）: SC port を保持し、平行移動範囲 [0, 0.115] meters
  - **Fixture Rails**（Zone 3 & 4）: component mount を保持し、平行移動範囲 [0, 0.188] meters、回転範囲 [-60, +60] degrees

**Fixture/Mount**
- タスクボード上で部品を固定する専用ホルダです。Zone 3 と 4 では、fixture が LC plug、SC plug、SFP module を整理された pick location に保持し、位置と向きを調整できます。

## ロボットと制御

**Robot Controller (aic_controller)**
- ロボットの motion、force 管理、actuator command を扱う低レベル制御システムです。joint 空間と Cartesian 空間の両方の制御に対応します。

**End-Effector/Gripper**
- ケーブルや部品を把持・操作するロボットの取り付け先で、**Tool Center Point (TCP)** とも呼ばれます。対応する frame は `gripper/tcp` です。gripper state は `/gripper_state` topic で監視されます。

**Joint-Space Control**
- 目標 joint 構成によってロボット motion を指令する制御方式です。command は `/aic_controller/joint_commands` に publish されます。

**Cartesian-Space Control**
- 3D 空間における目標 pose（位置と向き）または linear / angular velocity によってロボット motion を指令する制御方式です。command は `/aic_controller/pose_commands` に publish されます。

**Motion Update**
- Cartesian 空間制御向けの目標 pose と制御パラメータを指定する control command message（`aic_control_interfaces/msg/MotionUpdate`）です。

**Joint Motion Update**
- joint 空間制御向けの目標 joint 構成と制御パラメータを指定する control command message（`aic_control_interfaces/msg/JointMotionUpdate`）です。

**Impedance Control**
- 操作中の force 管理を可能にする compliance ベースのロボット制御で、繊細なケーブル挿入を損傷なく行うために重要です。

## センシングと知覚

**Sensor Fusion**
- 複数の sensor 入力（camera、force/torque sensor、joint state）を組み合わせ、一貫した環境理解へまとめる処理です。

**Observation**
- ロボットの感覚環境のスナップショット（`aic_model_interfaces/msg/Observation`）で、camera 画像、joint state、force 計測値、transform frame を含みます。

**Force/Torque Sensor (F/T Sensor)**
- 操作中に加わる force と torque を計測します。データは `/fts_broadcaster/wrench` topic に publish され、感度の高い force feedback 制御を可能にします。F/T sensor は通常、各学習 episode 前にゼロ点調整されます。詳しくは [学習前のゼロ点調整](./scene_description.md#taring-before-training) を参照してください。

**Wrist Cameras**
- ロボットの手首に取り付けられた 3 台の RGB camera:
  - Left camera (`/left_camera/image`)
  - Center camera (`/center_camera/image`)
  - Right camera (`/right_camera/image`)
- 各 camera は対応する `/camera_info` topic で calibration data を提供します。

**Joint State**
- 全ロボット関節の現在構成と速度で、`/joint_states` topic に publish されます。

## 通信とインターフェース

**ROS 2 (Robot Operating System 2)**
- toolkit の各コンポーネント間で topic、service、action による通信を実現する middleware framework です。

**Topic**
- ROS 2 node 間で pub/sub message をやり取りする名前付き channel です。sensor streaming や command broadcast に使われます。

**Action**
- goal ベースの request、feedback、result を扱える ROS 2 通信パターンです。タスク起動（例: `/insert_cable`）に使われます。

**Message (ROS Message)**
- ROS 通信用の構造化データ形式です。例: `InsertCable.action`、`Task.msg`、`MotionUpdate.msg`。

**aic_interfaces**
- challenge で使用するすべてのカスタム ROS 2 message、service、action を定義し、一貫した protocol 定義を提供する package です。

**InsertCable Action**
- ケーブル挿入 policy の実行を開始させる ROS 2 action interface（`aic_task_interfaces/action/InsertCable`）です。

**Task Message**
- ケーブル挿入タスクの具体的なパラメータと状態を記述する ROS 2 message（`aic_task_interfaces/msg/Task`）です。

## 開発と提出

**Container/Docker**
- 参加者の policy ソリューションとすべての依存関係を package 化する container 技術です。異なる環境でも再現可能な評価を可能にします。

**Dockerfile**
- 提出用 policy container image の build 方法を定義する設定ファイルです。

**aic_engine**
- 完全な trial lifecycle を管理する orchestration system です。ランダム化 task board の生成、policy の挙動および lifecycle 準拠確認、task 実行監視、InsertCable action の起動、複数の ROS 2 topic からのスコアリングデータ収集を行います。

**Qualification Phase**
- 初期の simulation-only competition phase です。参加者は model を学習し、3 つの specific trial に対して評価されるソリューションを提出します。実行はすべて Gazebo 上で行われ、堅牢な policy 学習のために IsaacLab（NVIDIA）と MuJoCo（Google DeepMind）の mirror environment も利用できます。

**Trial**
- ケーブル挿入タスクの単一実行で、ケーブルの片側 plug だけがランダム化 target port への挿入対象になります。各 trial では、特定の board 構成と部品配置の下で policy の性能が評価されます。

## 特化コンポーネント

**aic_bringup**
- simulation、robot、sensor、scoring を含む challenge 環境全体を起動するための設定を持つ launch file package です。

**aic_example_policies**
- ケーブル挿入タスクを解くためのさまざまなアプローチやテクニックを示す reference implementation です。

**aic_description**
- simulation 用に、robot、task board、environment の URDF / SDF 記述を格納しています。

**aic_assets**
- Gazebo simulation 環境で使われる 3D model と visual asset の repository です。

**aic_scoring**
- performance metric の計算と、challenge criteria に基づく trial 成功判定を実装する system です。

**aic_utils**
- toolkit 全体で使われる utility package と helper tool 群です。

## シミュレーションと学習

**IsaacLab**
- NVIDIA による代替 simulation 環境で、AIC challenge の mirror を提供します。異なる physics engine 間での domain randomization を可能にし、sim-to-real transfer の改善に役立ちます。

**MuJoCo**
- Google DeepMind による代替 simulation 環境で、AIC challenge のもう 1 つの mirror を提供します。Gazebo や IsaacLab と並行して使い、複数 simulator をまたぐ学習戦略に利用できます。

**Domain Randomization**
- policy を複数 simulator（Gazebo、IsaacLab、MuJoCo）の差異にさらす学習手法です。simulator 間の物理差異自体を自然な randomization として利用し、sim-to-sim-to-real transfer に備えます。

## 評価指標

**Jerk**
- joint acceleration の変化率（位置の 3 階微分）として計算される滑らかさ指標です。`/joint_states` topic のデータから算出されます。jerk が低いほど軌道は滑らかで制御されており、高い score を得ます。

**Convergence**
- 時間経過に伴う plug tip と target port の Euclidean distance を測る性能指標です。`/ground_truth_poses` topic を利用します。距離が小さく、より速く収束するほど高い score になります。

**Task Completion Time**
- task 開始から挿入成功完了までの所要時間です。aic_engine の `TaskState` message によって監視されます。短いほど高い score になります。

**Smoothness**
- jerk 計算に基づく全体の軌道品質指標です。挿入試行中にロボットがどれだけ滑らかに動いたかを評価します。

**Success Rate**
- 正しい位置合わせと完全挿入が許容範囲内で達成されたかを示す binary 指標です。contact sensor、force/torque feedback、Gazebo plugin によって検証されます。Tier 3 の主要評価基準です。

**Collision Penalty**
- 環境または task board との意図しない接触に対する減点です。custom Gazebo plugin が、意図した cable 関連接触を除くすべての接触を監視して検出します。深刻度は接触頻度と force に比例します。

**Force Safety**
- 挿入中に加わる force を監視し、部品安全限界内に収まっているかを確認します。閾値を超える過大な force には、大きさと継続時間に応じて減点が適用されます。

**Command Safety**
- `aic_controller` に送られる motion command が過大値でないかを検証します。`MotionUpdate` と `JointMotionUpdate` message を監視し、危険な command 値には減点が適用されます。

## 変換と座標系

**TF (Transform Frames)**
- robot、sensor、environment 間の座標 frame 関係を追跡する ROS 2 system です。sensor-to-robot-to-world の適切な座標変換を可能にします。

**Pose**
- 物体の位置（x, y, z）と向き（quaternion）をまとめた完全な表現で、通常は 3D Cartesian 空間で用いられます。

**Quaternion**
- 3D の向きを 4 つの成分（x, y, z, w）で表す数学表現で、Euler angle に内在する gimbal lock 問題を回避できます。
