# AI for Industry Challenge Qualification 作業仕様書

この文書は、Qualification で **高得点かつ submission-safe** な解法を考える AI が、そのまま前提条件として読めるように整理した仕様書である。目的は、何を最適化すべきか、何に依存してはいけないか、実行時に何が見えて何が禁止されるかを、単体で判断できるようにすることにある。

## 1. 何を達成する競技か

**確定ルール:**

- 競技の中心課題は、**柔軟ケーブルのコネクタを正しいポートへ挿入すること**である。
- 参加者は、ROS ベースの入出力を使ってセンサ情報を受け取り、ロボットへ動作指令を出す policy を開発する。
- Qualification は **Gazebo 上のシミュレーション評価**である。
- 後続フェーズとして Flowstate を使う段階と、実機ワークセルで評価する段階がある。

**実務前提:**

- 狙うべきものは public sample の暗記ではなく、**ランダマイズされた board / port 配置に対して頑健に target-local に整列し、挿入まで到達する policy** である。
- Qualification では実機展開そのものは不要だが、後続フェーズを考えるなら過度な simulator 固有ハックは避けた方がよい。

## 2. Qualification で実際に解くタスク

### 2.1 Trial 構成

**確定ルール:**

現行の Qualification は 3 trial 構成で、**同一 policy が全 trial を処理**する。

| Trial | 把持して開始する plug | 挿入先 | 主な評価点 |
| :--- | :--- | :--- | :--- |
| Trial 1 | `SFP_MODULE` | `NIC_CARD` 上の `SFP_PORT_0` または `SFP_PORT_1` | SFP 挿入 |
| Trial 2 | `SFP_MODULE` | `NIC_CARD` 上の `SFP_PORT_0` または `SFP_PORT_1` | SFP 挿入、ランダム条件違い |
| Trial 3 | `SC_PLUG` | task board 上の `SC_PORT_0` または `SC_PORT_1` | SC 挿入 |

**実務前提:**

- SFP 専用 policy と SC 専用 policy を分けるのではなく、**task 条件に応じて 1 つの policy 内で分岐**する設計が自然である。

### 2.2 初期状態とランダマイズ

**確定ルール:**

- ロボットは開始時点で **片側の plug を把持済み** である。
- 反対側の plug は free で、接続されていない。
- 1 trial では **単一 cable insertion** だけが評価される。
- Qualification で評価対象になる組み合わせは、現時点では次の 2 系統である。
  - `SFP_MODULE -> SFP_PORT`
  - `SC_PLUG -> SC_PORT`
- Qualification では **未知の plug / port type は出ない**。
- target port は camera の view 内にある。
- robot は target の近傍から開始する。
- trial ごとに少なくとも次が変化しうる。
  - task board の pose と yaw
  - board 上の component 配置
  - `NIC_CARD` の rail 選択
  - `NIC_CARD` の rail 上 translation と yaw offset
  - `SC_PORT` の rail 上 translation
  - grasp pose の微小ずれ

**実務前提:**

- 主戦場は「遠距離探索」ではなく、**近接視野での target-local 認識と最終整列**である。
- 固定 world 座標の replay より、**視覚と現在姿勢に基づく局所制御**の方がルール適合性も一般化も高い。

### 2.3 `Task` で与えられる task 指示

**確定ルール:**

各 task は `Task` として `/insert_cable` action request で渡される。主要フィールドは次のとおり。

| フィールド | 意味 |
| :--- | :--- |
| `id` | task の一意 ID |
| `cable_type` / `cable_name` | cable の種類 / 名前 |
| `plug_type` / `plug_name` | 挿入すべき plug の種類 / 名前 |
| `port_type` / `port_name` | 挿入先 port の種類 / 名前 |
| `target_module_name` | target port を持つ component / module 名 |
| `time_limit` | request 受信から task 完了までに許される秒数 |

- goal は **`time_limit` 内に完了**しなければならない。

**実務前提:**

- `Task` は「何をどこに挿すか」を決める一次情報であり、runtime 中の分岐条件として強く使ってよい。
- 解法は `plug_type` と `port_name` を見て、SFP / SC の処理経路や最終アライメント戦略を切り替えるのが合理的である。

## 3. 絶対制約

### 3.1 Runtime で使ってよい情報源

**確定ルール:**

- runtime policy は **公式に公開されたインターフェースだけ**を使う必要がある。
- training / debugging 中は内部状態や ground truth を使ってよい。
- ただし evaluation 中に、公式 interface では見えない内部状態へ依存することは不可である。

**実務前提:**

- submission-safe の基準は、**提出物が evaluation 中に何へ依存するか**で決まる。
- offline 学習で ground truth を使ってもよいが、提出 policy の runtime ロジックは official input だけで閉じるべきである。

### 3.2 明確に禁止される行為

**確定ルール:**

- robot, cable, task board, simulation state の **直接改変**は禁止。
  - teleport
  - 強制 insertion
  - state の書き換え
- 公式 API を迂回した **backend / system 干渉**は禁止。
  - evaluation node の parameter 変更
  - evaluation node の lifecycle state 変更
  - Gazebo / scoring backend への不正アクセス
- 特定の public sample や既知配置への **過剰ハードコード**は禁止。
- `aic_engine`、scoring system、logging infrastructure への干渉は禁止。
- cloud evaluation 基盤の reverse engineering、tampering、portal / registry bypass は禁止。
- backdoor や submission architecture 回避機構を含む container / model の提出は禁止。

**実務前提:**

- 「ACL が通るかどうか」ではなく、**公開 participant interface だけを使っているか**で合法性を判断するべきである。
- public sample 固有の world pose や component 配置を決め打ちした戦略は、たとえローカルで動いても本番では危険である。

### 3.3 依存してはいけない namespace / 系統

**確定ルール:**

少なくとも次の系統は、提出 policy が依存してはいけない。

- `/scoring/*`
- `/gazebo/*`
- `/gz_server/*`
- entity の spawn / despawn / delete 系 service
- `/clock`
- `/model`
- `/world_stats`
- `/pause_physics`
- simulation reset 系

- submission environment では ACL により一部 backend へのアクセス自体が block される。

**実務前提:**

- runtime は `Task`、`Observation`、公式 topics / services / action だけで完結させるべきである。
- backend の state leak がないと成立しない設計は採用しない方がよい。

## 4. 実行時インターフェース

### 4.1 提出ノードの契約

**確定ルール:**

提出物は **ROS 2 Lifecycle node `aic_model`** として動作する必要がある。

| 項目 | 契約 |
| :--- | :--- |
| ノード名 | `aic_model` |
| 必須機能 | ROS 2 Lifecycle interface |
| task 受付 | `/insert_cable` action server |
| robot 制御 | 公式 command interface を通す |

### 4.2 Lifecycle の要求

**確定ルール:**

| State / 遷移 | 要求 |
| :--- | :--- |
| `unconfigured` | topic publish をしない。特に robot command を出さない |
| `configure` | 60 秒以内に成功する |
| `configured` | topic publish をしない。特に robot command を出さない。`/insert_cable` の goal は reject する |
| `activate` | 60 秒以内に成功する |
| `active` | `/insert_cable` の goal を accept する。goal は cancellable である |
| `deactivate` | 60 秒以内に `configured` へ戻る |
| `cleanup` | 60 秒以内に `unconfigured` へ戻る |
| `shutdown` | 60 秒以内に成功する。publish 不可。robot command publisher が graph 上に残ってはいけない |

- `active` で受けた goal は `Task.time_limit` 内に完了する必要がある。

**実務前提:**

- 重い model load、GPU warm-up、大量 checkpoint 展開は `configure` / `activate` の 60 秒制約を壊さない形で設計する必要がある。
- `configured` で publish してしまう構成は、それだけで失格リスクになる。

### 4.3 `/insert_cable` action

**確定ルール:**

`/insert_cable` の形は次のとおり。

| 部分 | 内容 |
| :--- | :--- |
| Request | `Task task` |
| Response | `bool success`, `string message` |
| Feedback | `string message` |

### 4.4 `Observation` と主要入力

**確定ルール:**

`aic_model` フレームワークを使う場合、policy は `Observation` を取得して runtime 情報を読む。`Observation` の主要構成は次のとおり。

| フィールド | 内容 |
| :--- | :--- |
| `left_image` / `center_image` / `right_image` | 3 つの wrist camera 画像 |
| `left_camera_info` / `center_camera_info` / `right_camera_info` | 各 camera の calibration |
| `wrist_wrench` | wrist の force / torque |
| `joint_states` | robot の joint state |
| `controller_state` | TCP pose / velocity などの controller state |

- 観測更新は最大 20 Hz。

### 4.5 公式 I/O 一覧

**確定ルール:**

主な公式入力:

- `/left_camera/image`
- `/left_camera/camera_info`
- `/center_camera/image`
- `/center_camera/camera_info`
- `/right_camera/image`
- `/right_camera/camera_info`
- `/fts_broadcaster/wrench`
- `/joint_states`
- `/gripper_state`
- `/tf`
- `/tf_static`
- `/aic_controller/controller_state`
- `/insert_cable`

主な公式出力:

- `/aic_controller/pose_commands`
  - `MotionUpdate`
- `/aic_controller/joint_commands`
  - `JointMotionUpdate`

関連 service:

- `/aic_controller/change_target_mode`
- `/aic_controller/tare_force_torque_sensor`

補足:

- evaluation 中は `tare_force_torque_sensor` を前提にした設計はできない。
- force/torque の tare は evaluation system 側で cable spawn 前に処理される。

**実務前提:**

- runtime の主要信号は `Task`、3 camera、camera intrinsics、joint state、wrench、controller state である。
- board や cable の正解 pose を backend から読む設計ではなく、**camera と現在状態から target を推定する設計**が必要になる。

## 5. 採点

### 5.1 スコアの構造

**確定ルール:**

1 trial の score は次で計算される。

```text
Total Score = Tier 1 + Tier 2 + Tier 3
```

- 現行の 3 trial 構成をそのまま前提にすると、理論上の合計上限は `300`。
- 1 trial あたりの理論上限は `100`。

### 5.2 Tier 1: Model Validity

**確定ルール:**

| 条件 | 点数 |
| :--- | :--- |
| validation passed | `1` |
| validation failed | `0` |

- node が正常に load / activate できること
- `InsertCable` request に応答できること
- valid な robot command を送れること
- lifecycle 契約と behavioral requirement を守ること

### 5.3 Tier 2: Performance & Convergence

**確定ルール:**

Tier 2 の加点 / penalty は次のとおり。

| 項目 | 範囲 | 条件 |
| :--- | :--- | :--- |
| Trajectory smoothness | `0` から `6` | jerk が小さいほど高得点 |
| Task duration | `0` から `12` | 速いほど高得点 |
| Trajectory efficiency | `0` から `6` | path length が短いほど高得点 |
| Insertion force penalty | `0` から `-12` | `20 N` 超を `1 sec` 超維持で penalty |
| Off-limit contact penalty | `0` から `-24` | restricted area contact で penalty |

追加の数値条件:

- smoothness は jerk `0` で `6`、jerk `>= 50 m/s^3` で `0`
- duration は `<= 5 sec` で `12`、`>= 60 sec` で `0`
- efficiency は初期 plug-port 直線距離が最良基準で、そこから `+1 m` 以上余計に動くと `0`

- Tier 2 の正の加点は、**task 成功または port 近傍到達により Tier 3 が正になる場合**にのみ有効になる。

### 5.4 Tier 3: Task Success

**確定ルール:**

| 結果 | 点数 |
| :--- | :--- |
| correct port への full insertion | `75` |
| wrong port への insertion | `-12` |

full insertion でない場合:

| 状態 | 範囲 | 条件 |
| :--- | :--- | :--- |
| Partial insertion | `38` から `50` | port 内に入っており、深いほど高得点 |
| Proximity | `0` から `25` | port 外でも入口近傍なら加点 |

### 5.5 スコア設計上の意味

**実務前提:**

- Tier 1 を通すだけでは `1` 点しか入らない。
- Tier 3 を `0` より上に乗せないと、Tier 2 の正加点も伸びない。
- proximity 帯だけでは高得点の上限が低い。
- 本気で勝ちに行くなら、**partial insertion ではなく full insertion を主目標**に置くべきである。
- wrong port insertion は `-12` なので、誤ポートへ押し込む戦略は有害である。

## 6. 提出運用

### 6.1 提出の流れ

**確定ルール:**

Qualification submission は **OCI-compliant image** で提出する。

標準的な流れ:

1. image を build する
2. ローカルで評価を走らせる
3. ECR に push する
4. portal で image URI を登録する

### 6.2 ECR と tag 制約

**確定ルール:**

- team ごとに AWS credential と ECR repository URI が割り当てられる。
- **ECR tag は immutable**。
- 既存 tag の上書きはできない。
- 新しい提出ごとに新しい tag が必要。

### 6.3 Portal 登録

**確定ルール:**

- **ECR へ push しただけでは評価は始まらない。**
- push 後に portal で Qualification phase を選び、OCI image URI を登録する必要がある。

### 6.4 Submission status

**確定ルール:**

| Status | 意味 |
| :--- | :--- |
| `Submitted` | image URI を受理した段階 |
| `Queued` | 評価待ち |
| `Running` | 評価実行中 |
| `Finished` | 評価完了 |
| `Failed` | crash / dependency 不足 / timeout などで失敗 |

### 6.5 提出頻度

**確定ルール:**

- 提出総数に絶対上限はない。
- ただし **1 日あたり 1 submission** に制限される。

**実務前提:**

- 失敗 submission でも日次の試行機会を失う可能性があるため、**ローカル検証を先に通す価値が高い**。
- portal 登録忘れや tag 再利用のような運用ミスは、技術力と無関係に機会損失になる。

## 7. 解法設計時の実務前提

**実務前提:**

- runtime を閉じるべき入力は `Task`、`Observation`、公式 topics / services / action である。
- `SFP` と `SC` を別提出に分けるのではなく、**単一 policy 内で task-conditioned に処理する**。
- board pose や rail 上の配置は変わるので、固定 world-frame waypoint の replay は危険である。
- target は近く、camera の view 内にあるため、**target-local な visual servo / local pose estimation / contact-aware insertion** が有力である。
- 最終数 mm は pure open-loop push より、低速・軸方向・wrench を使った time-bounded correction の方が scoring と整合する。
- `configure` / `activate` の 60 秒制約があるため、初期化の重さも policy 設計の一部である。
- evaluation 中に backend 情報や ground truth がなくても成立するロジックでなければ、本番では壊れる。

## 8. 未確定事項と優先順位

### 8.1 未確定事項

**未確定:**

- Qualification 最終評価では、trial 数や trial 順序が変わり得る。
- 後続の Phase 1 / Phase 2 の詳細仕様は未公開である。

### 8.2 AI が解法検討で見るべき優先順位

1. Tier 1 を確実に通す。
2. Tier 3 を `0` より上に乗せる。
3. proximity 止まりではなく full insertion を目標にする。
4. backend 情報、ground truth runtime 依存、public sample 固定条件への依存を排除する。
