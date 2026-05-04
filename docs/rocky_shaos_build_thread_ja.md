# Rocky_Shao Build Thread 日本語訳

- 元スレッド: [Rocky's Open-Source Build Thread (AI for Industry Challenge)](https://discourse.openrobotics.org/t/rockys-open-source-build-thread-ai-for-industry-challenge/53155)
- 英語版: [rocky_shaos_build_thread.md](./rocky_shaos_build_thread.md)
- 対象投稿: `Rocky_Shao` のみ
- 投稿数: `15`
- 時刻表記: Discourse の ISO 8601 を `UTC` のまま記載
- 取得基準: 英語版 dump をもとに、日本語へ翻訳

## 投稿 01
- 投稿番号: `1`
- 投稿日時: `2026-03-12T01:31:09.557Z`
- 元投稿: [Discourse Post #1](https://discourse.openrobotics.org/t/rockys-open-source-build-thread-ai-for-industry-challenge/53155/1)

こんにちは！ 僕は Rocky Shao、The Ohio State University の1年生です。

いわゆる「実世界の」ロボティクス経験はほとんどありません。このコンペには、自分を追い込みつつ、短期間で一気に学ぶために参加しました。もちろん締切があるのも大きいですが……。

それでも、自分の進捗を共有するのは、ロボティクス、AI、そしてクールで楽しい技術が好きな人たちとつながるいい方法だと思っています。AI では答えきれない疑問について、詳しい人たちから助けをもらえるかもしれませんし。

現在の進捗はこちらです: [https://github.com/Rocky0Shao/IntrinsicAIChallenge](https://github.com/Rocky0Shao/IntrinsicAIChallenge)

## 投稿 02
- 投稿番号: `2`
- 投稿日時: `2026-03-12T01:38:47.549Z`
- 元投稿: [Discourse Post #2](https://discourse.openrobotics.org/t/rockys-open-source-build-thread-ai-for-industry-challenge/53155/2)

**今の計画:**

1. [`lerobotTeleop`](https://github.com/intrinsic-dev/aic/blob/main/aic_utils/lerobot_robot_aic/lerobot_robot_aic/aic_teleop.py) の上に「データ収集」用 teleop クラスを追加して、CheatCode サンプル policy と同じロジックで「完璧な」データセットを作る。

2. CheatCode policy のロジックを修正して、ハードコードした Z オフセットではなく、力覚フィードバックを検知したら停止するようにする。

3. 記録したデータを Hugging Face に保存し、LeRobot の Google Colab ノートブックテンプレートと、学生向けクレジット 300 ドルを使って LeRobot ACT モデルを学習する。

## 投稿 03
- 投稿番号: `3`
- 投稿日時: `2026-03-12T02:09:42.437Z`
- 元投稿: [Discourse Post #3](https://discourse.openrobotics.org/t/rockys-open-source-build-thread-ai-for-industry-challenge/53155/3)

# 3月11日

### 今日の進捗
Ground-Truth TF フレームを使ってケーブル挿入を行う「CheatCode」LeRobot Teleop モードを追加しました。これで、キーボード teleop をしなくてもきれいな学習データを記録できます。

Trial 1 と Trial 2 では動きますが、Trial 3 では動きません。違う plug type 間で名前の不一致があるのではないかと疑っています。

https://github.com/Rocky0Shao/IntrinsicAIChallenge/blob/19a0ac32e01445bcadf8a08b789dcf8ac1a17217/aic_utils/lerobot_robot_aic/lerobot_robot_aic/aic_teleop.py#L340-L591

上のコードが動いている動画はこちらです（これは配布されている CheatCode policy そのものではなく、上のコードです）:
https://youtube.com/shorts/aCZ-L5IKyE4

### Pixi の小ネタ
それから、この version 番号を手動で大きくしないといけないことに気づきました:
https://github.com/Rocky0Shao/IntrinsicAIChallenge/blob/19a0ac32e01445bcadf8a08b789dcf8ac1a17217/aic_utils/lerobot_robot_aic/package.xml#L5-L5
その上で
```bash
pixi install
```
を実行して、`pixi run ...` が正しく振る舞うようにする必要があります。

### 自分のハードウェア構成
* Asus Zypherus G16 2024 (32GB RAM, Intel Ultra 9 CPU)
* Ubuntu 24 を 300GB パーティションでデュアルブートしています。  
今のところ大きな互換性問題には当たっていません（WSL や Arch を使っていたら、かなり大変だっただろうなと思います）。

### 明日の予定
次は、LeRobot で複数エピソードのデータを正しく記録する方法を学ぶつもりです。

おまけの短い動画もどうぞ:
https://youtube.com/shorts/pViucR-X3vQ

## 投稿 04
- 投稿番号: `4`
- 投稿日時: `2026-03-12T17:51:02.853Z`
- 元投稿: [Discourse Post #4](https://discourse.openrobotics.org/t/rockys-open-source-build-thread-ai-for-industry-challenge/53155/4)

# 3月12日: Trial 3 の TF Lookup 問題を解決

**問題:** 前に触れた LeRobot CheatCode Teleop が、Trial 3 の SC-Port（短い plug）で動いていませんでした。ポート名自体は正しそうなのに、ロボットがまったく動かなかったのです。
> Rocky_Shao より:
> Trial 1 と Trial 2 では動きますが、Trial 3 では動きません。違う plug type 間で名前の不一致があるのではないかと疑っています。

**根本原因:** 原因は名前の不一致でした。自分のコードは `cable_1` の下で TF フレームを探していましたが、manual spawner はエンティティを `cable_0` として作っていました。コード側を `cable_0` に向けたことで、TF lookup 問題が解消し、動くようになりました。

### 詳細: なぜこうなるのか

シーンの生成方法によって、エンティティ名が変わります。

* **自動スポーン (`aic_engine`)**: `start_aic_engine:=true` を使う場合、engine は `sample_config.yaml` の YAML キーをエンティティ名として動的にケーブルを生成します。つまり Trial 3 の `cable_1` キーは、正しく `cable_1` という名前のエンティティを生成します。

https://github.com/intrinsic-dev/aic/blob/5596c67374ee2b56a815152548f35f46f29f9d78/aic_engine/src/aic_engine.cpp#L1216-L1218
* **手動スポーン (`spawn_cable.launch.py`)**: `/entrypoint.sh` から `spawn_cable:=true` で手動スポーンする場合、この launch file はエンティティ名を `cable_0` に固定でハードコードしています。

https://github.com/intrinsic-dev/aic/blob/5596c67374ee2b56a815152548f35f46f29f9d78/aic_bringup/launch/spawn_cable.launch.py#L64-L67

manual scene spawning は `sample_config.yaml` を完全に無視するため（[参照](https://github.com/intrinsic-dev/aic/blob/5596c67374ee2b56a815152548f35f46f29f9d78/aic_engine/config/sample_config.yaml#L320-L320)）、Trial 3 のように `cable_1` を期待する場合でも、常に `cable_0` を作ってしまいます。

### まとめ

> **Ground-truth TF を使って custom scene を手動スポーンするなら、必ず `cable_0` を使うこと。**

## 投稿 05
- 投稿番号: `6`
- 投稿日時: `2026-03-12T21:29:58.298Z`
- 元投稿: [Discourse Post #6](https://discourse.openrobotics.org/t/rockys-open-source-build-thread-ai-for-industry-challenge/53155/6)

温かい言葉をありがとうございます。そして、ロボティクスコミュニティにこんなにわくわくする、よく整理されたチャレンジを提供してくれてありがとうございます！

## 投稿 06
- 投稿番号: `7`
- 投稿日時: `2026-03-13T02:12:37.863Z`
- 元投稿: [Discourse Post #7](https://discourse.openrobotics.org/t/rockys-open-source-build-thread-ai-for-industry-challenge/53155/7)

# 3月12日: Trial 3 の TF Lookup 問題を解決

**問題:** 前に触れた LeRobot CheatCode Teleop が、Trial 3 の SC-Port（短い plug）で動いていませんでした。ポート名自体は正しそうなのに、ロボットがまったく動かなかったのです。
> Rocky_Shao より:
> Trial 1 と Trial 2 では動きますが、Trial 3 では動きません。違う plug type 間で名前の不一致があるのではないかと疑っています。

**根本原因:** 原因は名前の不一致でした。自分のコードは `cable_1` の下で TF フレームを探していましたが、manual spawner はエンティティを `cable_0` として作っていました。コード側を `cable_0` に向けたことで、TF lookup 問題が解消し、動くようになりました。

### 詳細: なぜこうなるのか

シーンの生成方法によって、エンティティ名が変わります。

* **自動スポーン (`aic_engine`)**: `start_aic_engine:=true` を使う場合、engine は `sample_config.yaml` の YAML キーをエンティティ名として動的にケーブルを生成します。つまり Trial 3 の `cable_1` キーは、正しく `cable_1` という名前のエンティティを生成します。

https://github.com/intrinsic-dev/aic/blob/5596c67374ee2b56a815152548f35f46f29f9d78/aic_engine/src/aic_engine.cpp#L1216-L1218
* **手動スポーン (`spawn_cable.launch.py`)**: `/entrypoint.sh` から `spawn_cable:=true` で手動スポーンする場合、この launch file はエンティティ名を `cable_0` に固定でハードコードしています。

https://github.com/intrinsic-dev/aic/blob/5596c67374ee2b56a815152548f35f46f29f9d78/aic_bringup/launch/spawn_cable.launch.py#L64-L67

manual scene spawning は `sample_config.yaml` を完全に無視するため（[参照](https://github.com/intrinsic-dev/aic/blob/5596c67374ee2b56a815152548f35f46f29f9d78/aic_engine/config/sample_config.yaml#L320-L320)）、Trial 3 のように `cable_1` を期待する場合でも、常に `cable_0` を作ってしまいます。

### まとめ

> **Ground-truth TF を使って custom scene を手動スポーンするなら、必ず `cable_0` を使うこと。**

## 投稿 07
- 投稿番号: `8`
- 投稿日時: `2026-03-13T02:31:09.587Z`
- 元投稿: [Discourse Post #8](https://discourse.openrobotics.org/t/rockys-open-source-build-thread-ai-for-industry-challenge/53155/8)

# 3月12日 - パート2

宿題を早めに終わらせました。ロボティクスより楽しいことなんてあるでしょうか？

## 古い "CheatCode" LeRobot Teleop の問題

自分の古い **CheatCode LeRobot teleop** は、plug を**何も見ずに真下へまっすぐ**押し込もうとします。  
つまり plug が port と揃っていなくても、**再アライメントせずにそのまま押し続けてしまう**のです。

その結果、こんなよくない挙動が起きます:

https://www.youtube.com/shorts/zupveMsGg2o

## 目標

自分の custom **LeRobot "CheatCode" Teleop** を **力覚フィードバック**対応に改造して、次のようにしたいです。

- 力が **20N** を超えたら、押し続ける代わりに **持ち上げる**

現状、これは**まだ動いていません**  
（コード自体は GitHub に追加済みです）。

予想では、自分の **改造版 LeRobot CheatCode Teleop** が `/observations` ROS topic を**正しく subscribe できていない**気がしています。

---

## Force Data の確認

現在の force 読み取り値を可視化するために、次のコマンドを使いました:
```bash
ros2 topic echo /observations | grep -A 14 wrist_wrench
```
出力はこうなりました:
```yaml
wrist_wrench:
  header:
    stamp:
      sec: 86
      nanosec: 142000000
    frame_id: ati/tool_link
  wrench:
    force:
      x: -1.213755535327079
      y: -13.563044908263814
      z: 27.869689319898924
    torque:
      x: 1.8214097057249585
      y: -0.6879985208072665
      z: -0.17330135150952428
```
これは問題のよい例です:

* **Force z ≈ 27.8N (>20N)**
* それなのにロボットは**持ち上がらず、そのまま押し続ける**

---

## 疑問点
明日調べることがいくつかあります。

### 1. LeRobot の `record` は実際に何を記録しているのか？

**力覚フィードバック**はデータセットに含まれるのでしょうか。それとも記録しているのは次だけでしょうか:

* 画像
* joint state
* action

### 2. `ros2 bag record` が動かない

**pixi shell** の中でも `ros2 bag record` が動きません。

bag recording 機能をインストールしてみましたが、それでも**失敗します**。

---

## Gazebo と RViz の spawn バグ

小さなバグにも気づきました。

* **Taskboard / Cable は RViz には出る**
* でも **Gazebo には spawn されない**
![image](https://us1.discourse-cdn.com/flex022/uploads/ros/optimized/3X/4/a/4af8b945507d4f5f32892fdcc36bcfd905d9fe6e_2_690x421.jpeg)
`/entrypoint.sh` を再起動すると、この問題は解消するようです。

まだやることは多いです。  
でも、そのぶん学べてもいます。

## 投稿 08
- 投稿番号: `9`
- 投稿日時: `2026-03-14T02:30:33.742Z`
- 元投稿: [Discourse Post #9](https://discourse.openrobotics.org/t/rockys-open-source-build-thread-ai-for-industry-challenge/53155/9)

# 3月13日

昨日書いたとおり、自分の `CheatCode LeRobot Teleop` は `/obesrvations` ROS topic を subscribe できていませんでした。原因は単純で、コマンドの先頭に `pixi run` を付け忘れていただけでした。

次を実行したことで、自分の `cheatcode_teleop` が今は正しく subscribe できていると確認できました:
```bash
pixi run ros2 topic info /observations --verbose
```

### Force Sensor の妙な挙動
force sensor に少し変な挙動があります。ロボットを spawn した直後から、`wrist_wrench` の z 軸がいきなり 20N 前後を示します:
```yaml
wrist_wrench:
  header:
    stamp:
      sec: 10
      nanosec: 146000000
    frame_id: ati/tool_link
  wrench:
    force:
      x: 0.19647534913152367
      y: 0.5709232984170273
      z: 20.372605348811124
    torque:
      x: 0.15718355450545468
      y: -0.19727286686559287
      z: 0.006627747694545185
```
この service call で force sensor を tare してみました:
```bash
pixi run ros2 service call /aic_controller/tare_force_torque_sensor std_srvs/srv/Trigger
```
しかし z 軸は頑固に 20N のままです。

さらに面白いことに、ロボットが硬い面に押しつけられて詰まると、z 軸の値はむしろ*減少*します:
![image](https://us1.discourse-cdn.com/flex022/uploads/ros/optimized/3X/9/6/96c7a976fc6b5613bf89b75dfc4689c39f6a462b_2_459x500.jpeg)
```yaml
wrist_wrench:
  header:
    stamp:
      sec: 72
      nanosec: 384000000
    frame_id: ati/tool_link
  wrench:
    force:
      x: -1.0565478791114393
      y: -2.8273204030726795
      z: 7.123609734869961
    torque:
      x: -0.9946708484649165
      y: -1.1107599148782015
      z: -0.07033790872857923
```

### 今の仮説

1. **見ている topic が違う:** 公開されている TCP force feedback の topic を、自分が間違えて見ているかもしれない。
2. **Gripper の重さ vs 法線力:** 一定の 20N 下向き力は、単に gripper 自身の重さを読んでいるだけかもしれない。ロボットが面を押すと、その反力が重さを相殺するので、sensor が読む下向き力がむしろ小さくなる、と考えると説明がつく。

つまり、自作 teleop は今や正しく subscribe できているものの、力覚フィードバックをまだ正しく使えてはいません。

Ohio State では今日から Spring Break が始まるので、次の9日間はこの課題をいじる時間がたっぷりあります。（もちろん、その前に中間試験の復習を終えられればですが。）20N の tare 問題について何かアイデアがあればぜひ教えてください！

## 投稿 09
- 投稿番号: `11`
- 投稿日時: `2026-03-14T20:07:33.651Z`
- 元投稿: [Discourse Post #11](https://discourse.openrobotics.org/t/rockys-open-source-build-thread-ai-for-industry-challenge/53155/11)

@jlamperez さんへ

すごく丁寧な書き込み、本当にありがとうございます！ まさに 20N 問題について自分が探していた内容そのものでした。

しかも、cheatcode を使って学習データを作るという発想まで同じで、やっぱり考えることは似るんだなと思いました。

教えてもらった `get_observation` の snippet は、raw contact force を手で取り出すためにぜひ実装してみるつもりです。

ワークフローや動画例まで共有してくれて、改めてありがとうございます。本当に大きな助けになりました。

よろしくお願いします。  
Rocky

## 投稿 10
- 投稿番号: `12`
- 投稿日時: `2026-03-15T03:08:14.118Z`
- 元投稿: [Discourse Post #12](https://discourse.openrobotics.org/t/rockys-open-source-build-thread-ai-for-industry-challenge/53155/12)

# 3月14日 作業ログ更新

## 今日の大きな前進: Force Feedback が動き始めた

@jlamperez さんの素晴らしい writeup のおかげで、ついに自分の cheatcode teleop フローで force feedback を動かせるようになりました。

![image](https://us1.discourse-cdn.com/flex022/uploads/ros/optimized/3X/1/e/1ebe1673281e02eaa5c093f42202fc567bda8d2a_2_690x359.jpeg)

右側のターミナル出力を見ると分かるとおり:

```yaml

TFs found! Starting APPROACH phase.

[CheatCode] Force: 0.2N | Phase: APPROACH | z_off: 0.2000

[CheatCode] Force: 3.1N | Phase: APPROACH | z_off: 0.2000

[CheatCode] Force: 1.5N | Phase: APPROACH | z_off: 0.2000

[CheatCode] Force: 1.6N | Phase: APPROACH | z_off: 0.2000

[CheatCode] Force: 1.3N | Phase: APPROACH | z_off: 0.2000

Hover reached (err=0.0063m). Entering ALIGN phase.

[CheatCode] Force: 1.2N | Phase: ALIGN | z_off: 0.0500

[CheatCode] Force: 1.1N | Phase: ALIGN | z_off: 0.0500

Aligned! (xy=0.0024m, ang=0.000rad, dwell=1.7s). Starting INSERT.

[CheatCode] Force: 0.6N | Phase: INSERT | z_off: 0.0500

[CheatCode] Force: 0.6N | Phase: INSERT | z_off: 0.0100

[CheatCode] Force: 0.9N | Phase: INSERT | z_off: -0.0100

[CheatCode] Force: 1.1N | Phase: INSERT | z_off: -0.0100

[CheatCode] Force: 5.8N | Phase: INSERT | z_off: -0.0150

[CheatCode] Force: 8.6N | Phase: INSERT | z_off: -0.0150

[CheatCode] Force: 10.5N | Phase: INSERT | z_off: -0.0150

[CheatCode] Force: 12.9N | Phase: INSERT | z_off: -0.0150

[CheatCode] Force: 12.6N | Phase: INSERT | z_off: -0.0150

[CheatCode] Force: 12.6N | Phase: INSERT | z_off: -0.0150

[CheatCode] Force: 12.4N | Phase: INSERT | z_off: -0.0150

[CheatCode] Force: 10.9N | Phase: INSERT | z_off: -0.0150

[CheatCode] Force: 12.7N | Phase: INSERT | z_off: -0.0150

[CheatCode] Force: 12.7N | Phase: INSERT | z_off: -0.0150

[CheatCode] Force: 12.9N | Phase: INSERT | z_off: -0.0150

[CheatCode] Force: 13.4N | Phase: INSERT | z_off: -0.0150

[CheatCode] Force: 12.9N | Phase: INSERT | z_off: -0.0150

[CheatCode] Force: 13.4N | Phase: INSERT | z_off: -0.0150

[CheatCode] Force: 13.0N | Phase: INSERT | z_off: -0.0150

[CheatCode] Force: 13.4N | Phase: INSERT | z_off: -0.0150

[CheatCode] Force: 13.7N | Phase: INSERT | z_off: -0.0150

[CheatCode] Force: 4.9N | Phase: INSERT | z_off: -0.0150

[CheatCode] Force: 6.6N | Phase: INSERT | z_off: -0.0150

[CheatCode] Force: 7.9N | Phase: INSERT | z_off: -0.0150

[CheatCode] Force: 9.6N | Phase: INSERT | z_off: -0.0150

[CheatCode] Force: 11.3N | Phase: INSERT | z_off: -0.0150

[CheatCode] Force: 13.1N | Phase: INSERT | z_off: -0.0150

(insertion completed)

Teleop loop time: 16.86ms (59 Hz))

```

現在の state machine には force feedback が組み込まれていて、この run では挿入中に何が起きているかを以前よりずっとよく観察できました。

## 今日やった変更

### A) 将来の agent 向けに、完全版 PROJECT_CONTEXT.md を追加

次の場所に、かなり包括的な context file を作りました:

`mystuff/for_agents/PROJECT_CONTEXT.md`

全216行で、内容は次を含みます:

- コンペ概要
- パイプライン戦略
- リポジトリ構成
- `aic_teleop.py` のアーキテクチャ
- スコアリングシステム
- trial 構成
- 技術的詳細
- AI agent 向けの実務メモ

狙いは単純です。今後どんな AI agent がこの repo に入ってきても、毎回最初から背景説明をしなくて済むようにすることです。

### B) `aic_teleop.py` 内の AICCheatCodeTeleop (v2) を書き直した

cheatcode teleop の state machine にかなり大きなアップグレードを入れました:

- 専用の **ALIGN phase** を追加
- port 上 5cm の hover 高度で最低 1 秒 dwell
- INSERT に入る前に、XY 誤差 < 3 mm、角度誤差 < 0.05 rad を要求
- 挿入下降速度を **0.07 m/s → 0.02 m/s** に低下
- **force 比例の挿入速度制御**を追加
- 5N から減速開始
- 15N で完全停止
- 旧来の二値 go/stop 挙動を置き換え
- recovery 挙動を改善
- retreat は 20cm ではなく 5cm まで
- 復帰先は APPROACH ではなく ALIGN
- 最大 3 回まで再試行
- INSERT 中に plug TF を継続再読込して live XY 補正を追加
- phase 遷移を厳格化
- `dist_to_target < 0.01` を使用（以前の recovery ロジックにあった緩い 0.2 を廃止）
- INSERT 前に角度収束ゲートを追加
- INSERT 中の最大 linear velocity を 60% に制限して、より穏やかな動きにした
- gains を調整:
- `kp_linear`: 1.0 → 1.2
- `ki_linear`: 0.15 → 0.2
- `kp_angular`: 1.5 → 2.0
- `max_linear_vel`: 0.1 → 0.08

### C) retreat の閾値を 18N から 10N に下げた

最初は 18N で retreat する recovery mode を書いていましたが、上のターミナル証拠を見る限り、ロボットは 18N に達しなくても **10-13N** 付近でかなり悪く詰まることがあります。

つまり古い閾値のままだと、詰まっていても押し続けて運任せにするしかありませんでした。

新しい挙動チェーンはこうです:

- **0-5N**: フルスピード
- **5-15N**: 線形に減速
- **≥15N**: 完全停止
- **≥10N が 0.5 秒続く**: retreat して再試行

これで動作はかなり安全になり、force penalty の事故も減らせるはずです。

## サイドプロジェクトの話

数か月前に始めて、ずっと継続できてはいないものの、並行してこんなプロジェクトにも取り組んでいます:

https://discourse.openrobotics.org/t/energy-efficient-autonomous-navigation-benchmarking/53208?u=rocky_shao

## ツールの話 (Slate)

それから、Slate という新しいコーディングツールも見つけました。今日の変更の多くは、それを使って巨大でまとまりのないプロンプトを1本渡しながら進めた結果です。自分はまだコーディング、エンジニアリング、AI agent ワークフローのどれにおいても専門家ではありませんが、このツールは最初の1時間で期待以上の働きをしてくれたので、共有したくなりました。

![image](https://us1.discourse-cdn.com/flex022/uploads/ros/optimized/3X/2/4/24d21be5bb1f2311ac3098f0fd89961ce188f69e_2_690x359.jpeg)

Slate からは、自分の今の cheatcode teleop は **position control** ではなく **velocity control** を使っているとも指摘されました。自分ではまだ完全には検証できていないので、今日の時点では 100% 断言できませんが、明日掘り下げるつもりです。

全体として今日はかなり前進できました。force を意識した挙動、よりきれいな state transition、そして今後の反復のための project context が手に入りました。

## 投稿 11
- 投稿番号: `13`
- 投稿日時: `2026-03-16T03:18:05.339Z`
- 元投稿: [Discourse Post #13](https://discourse.openrobotics.org/t/rockys-open-source-build-thread-ai-for-industry-challenge/53155/13)

# **3月15日 更新**

今日は宿題と中間試験の復習に追われたうえ、午後は自分の Autonomous Car Team の活動もありました（これはまた別の話題ですね！）。人生にはいろいろあるので、このチャレンジに触れられたのは短い時間だけでした。

短いデモはこちらです:  
https://youtu.be/IrH-5pMJCRs

**State Machine Logic (AICCheatCodeTeleop):** arm が plug 挿入をどう扱うか、そのロジックの流れを整理しました。

* **INIT:** 必要な TF フレーム（port, plug, gripper）が揃うまで待つ。
* **APPROACH ➡️ ALIGN:** PI velocity controller を使って、まず port の 20cm 上に hover する。1cm 以内に入ったら、5cm hover に下げて XY 位置（誤差 < 3mm）と角度姿勢を微調整する。
* **INSERT:** 力に応じて速度を変えながら port に下降する（15N の抵抗に当たると完全停止するまで減速）。挿入深さが -1.5cm に達したら終了。
* **RECOVERY:** Align/Insert 中に有効。力が 10N を 0.5 秒超えたら、5cm まで retract して PI integrator をリセットし、再試行する（最大 10 回で abort）。
* **DONE:** 終端状態（ゼロ速度を出力）。
* **Controller Details:** linear velocity には PI controller、angular には P controller を使用。速度には clamp をかけ、挿入中は 40% 減にする。さらに world frame の速度を TCP frame に変換している。

**振り返り:** 少し引いて見ると、このチャレンジでの自分の旅路について考えることが増えてきました。自分は ROS や「現実世界の」ロボティクスにまだまだ新しいので、「AI に plug を port へ挿させる」という巨大な目標を、自分と AI が本当に理解しながら実行できる小さな塊に分解することが一番難しいです。

時間管理ももう一つの大きな壁です。学校がある間は、ロボティクスの時間を少しでも確保するために宿題を片づけるレースになります。休みに入ると逆に、「プレッシャーも監視もない中で、どう健康的な生活リズムを保ちつつ作業を続けるか」という別の問題になります。

自分の主目標はとにかく学ぶことです（もちろん勝てたら最高ですが）。だからこそ、AI コーディングツールで開発速度を上げることと、自分でコードを書いてロジックや実装を深く理解することのちょうどよいバランスを、ずっと探しています。

この AI 時代では、「学ぶこと」と「できるだけ早く届けること」の両立は本当に難しいです。

## 投稿 12
- 投稿番号: `14`
- 投稿日時: `2026-03-17T03:48:14.961Z`
- 元投稿: [Discourse Post #14](https://discourse.openrobotics.org/t/rockys-open-source-build-thread-ai-for-industry-challenge/53155/14)

# 3月16日

データ収集について、今はこんなジレンマに直面しています:

1. PID controller を完璧にチューニングして、毎回 1 回目で確実に挿入できるようにする。
2. 「そこそこ動く」PID controller に、複雑な state machine と magic number と例外処理を重ねて、何とか回ることを期待する。

そして自分は、2番を選びました。

昔の制御 state machine は、要するにこうでした。ロボットが挿入して、詰まって、上まで持ち上げて、また試す。自分はこれを "spam and pray" 方式と呼んでいます。結果はかなりひどく、次の動画のとおり、挿入成功率は 8 回に 1 回くらいでした（最後には一応成功しますが）:
https://youtube.com/shorts/H_iYKAh8NQc

もちろん、この "spam and pray" アプローチには無理がありました。そこで考えました。*自分がスマホを充電するとき、USB-C ケーブルをどう挿すだろう？* 合わなかったからといって、ただ無理やり押し込んだり、大きく引き抜いたりはしません。少し差し込んで、噛み合わなければ軽く wiggle して位置を合わせます。この "wiggle" をロボットで再現できないか、と。

そこで、古い攻撃的な pop-out ループをやめて、新しい **SEARCH ("wiggle")** 動作に置き換えました。これで recovery がかなり落ち着いたものになりました。実装はこうです。

plug が詰まった瞬間に引き抜くのではなく、ロボットは挿入力を監視します。力が **17N を 0.3 秒以上**超えたら、**半径 5 mm の水平円運動 wiggle** を行いながら、**0.002 m/s** でゆっくり下向きに進みます。もし力が下がれば、開口部を見つけた可能性が高いので、通常の挿入をすぐ再開します。**3 回の wiggle cycle** すべてが失敗したときだけ、**小さな持ち上げ（約 3 cm）** を行います。これで controller は、最初から何度もリセットする代わりに、port 近傍に長く留まれるようになります。

この動作を滑らかにするため、通常の挿入負荷がおよそ 14N であることを基準に、閾値を次のように調整しました:

* **15N:** 減速開始
* **17N:** SEARCH/wiggle を開始
* **19N:** 強制 recovery
* **<20N:** penalty 領域に入らないよう安全側に保つ

この更新前は、攻撃的な recovery がスループットを自分で壊していました。閾値を超えるたびに arm が必要以上に上まで持ち上がり（最初は 5 cm、後に 2 cm）、ALIGN phase に戻ってまた試す、ということを繰り返していたのです。ひどいときには、運良く成功するまで *insert → recover → insert* ループを 8 回以上回していました。

wiggle を作る過程で、そもそも robot を recovery に追い込んでいた根本バグもいくつか見つけて直しました。下降速度が frame-rate 依存になっていて（実質 0.6m/s、つまり約 60 倍速すぎた）、それが即座の force spike を生んでいたことに気づきました。下降速度は今は正しく dt スケーリングされています。また、良い seat-in を見逃さないように DONE 状態がどの phase からでも発火できるようにし、APPROACH 中には XY overshoot を避けるため integrator をゼロ化し、さらに robot が延々と wiggle し続けないよう recovery timeout 5 秒も追加しました。

ただ、"wiggle" 戦略にも限界があることがすぐ分かってきました。SC plug では、小さな "horns" が port の外側に引っかかると、水平方向に wiggle するだけではあまり効きません（だいたいそういうときに 3 回上限までいって失敗します）。失敗原因が本当に horns なのか、それとも 5 mm の wiggle 半径が小さすぎるのかはまだ分かりません。collision box を表示して port を確認したところ、その horns は実際には port の collision box を通り抜けているようにも見えました。

![image](https://us1.discourse-cdn.com/flex022/uploads/ros/optimized/3X/7/6/7680516acff6edc298105f667eb54367e8356c96_2_362x375.jpeg)

もう一つ腹立たしい失敗パターンもありました。plug が半分差さった状態で、本来ならそのまま下向きに挿入を続けるべきなのに、gripper の force feedback が 14N 付近にあるため、ハードコードした safety threshold が発火して挿入が止まってしまうのです。
![image](https://us1.discourse-cdn.com/flex022/uploads/ros/optimized/3X/4/9/499ac0a6b93108e6366ca4d56620d2f33436d76f_2_479x500.jpeg)

### Magic Number だらけ
今は state machine を頑健にしようとして、どんどん hard-coded な "magic number" を足してしまっています。でも正直、考えが変わりつつあります。無理にこの state machine に edge case と magic number を継ぎ足し続けるより、思い切って Path 1 に戻り、本当に頑健な control loop を作ったほうがいいのではないか、と感じ始めています。
https://github.com/Rocky0Shao/IntrinsicAIChallenge/blob/9389ff432bee01d1ebeb6f0503bbe6189fa365ab/aic_utils/lerobot_robot_aic/lerobot_robot_aic/aic_teleop.py#L361-L400

## 投稿 13
- 投稿番号: `15`
- 投稿日時: `2026-03-18T03:39:38.844Z`
- 元投稿: [Discourse Post #15](https://discourse.openrobotics.org/t/rockys-open-source-build-thread-ai-for-industry-challenge/53155/15)

# 3月17日
### ACT 学習用のクリーンな訓練データに向けた、簡素化 CheatCode Teleop

**今日の焦点: State Machine の整理**  
cheatcode teleop を、recovery のないシンプルな state machine (`INIT → APPROACH → ALIGN → INSERT → DONE`) にまで削ぎ落としました。`SEARCH`/wiggle と `RECOVERY`/retreat phase をすべて削除し、ACT model が失敗と回復を学ぶのではなく、きれいな一発挿入を学べるようにするのが狙いです。

**主な技術変更:**

* **Wrench Feedback:** コンプライアンスゲイン (`[0.5, 0.5, 0.5, 0, 0, 0]`) を有効化し、挿入中の横方向補正を自動化した。
* **XY-Only Integrator:** plug tip から port までの誤差だけを追うようにした（公式 CheatCode の構造と一致）。
* **State Transitions:** `ALIGN → INSERT` 遷移でも integrator を保持し、蓄積した XY 補正をそのまま引き継ぐようにした。さらに挿入中は binding torque を防ぐため angular gain を 25% に落とした。
* **Tighter Tolerances:** XY 0.5mm、角度 0.03rad、最低 dwell 2 秒に設定。
* **Descent Profile:** force 比例の rampdown をやめ、一定速度下降（12mm/s）+ 19.5N でのシンプルな safety hold に切り替えた。

**次のステップと blocker:** port 形状の違いに対応するため、別々の cheatcode teleop 設定を2種類実装する予定です。

* **SFP Connectors (Trials 1 & 2):** 長方形形状と狭い chamfer に合わせて調整。
* **SC Connectors (Trial 3):** 丸い形状と spring-loaded latch に合わせて調整。こちらは自然な force spike に対応するため、より遅い挿入速度、長い alignment dwell、高めの force threshold が必要。

**目標:** 2つの設定がどちらも sim 上で 100% 一発挿入できるようになったら、ACT model 学習に必要な trial ごと 50 本以上の demonstration episode を集め始めます。

## 投稿 14
- 投稿番号: `18`
- 投稿日時: `2026-03-19T17:41:31.501Z`
- 元投稿: [Discourse Post #18](https://discourse.openrobotics.org/t/rockys-open-source-build-thread-ai-for-industry-challenge/53155/18)

> Hui_Liu より:
> 共有ありがとう！ どうして CheatCode policy 自体の中で、動画や sensor/action 情報をそのまま保存しないのでしょうか？ LeRobot にデータ収集機能（それに学習フレームワークまで）が組み込まれているからですか？

*(まず大前提として、以下の内容はかなり割り引いて受け取ってください。自分は LeRobot を使った経験がほとんどありません！)*

はい、その通りです！ LeRobot には学習データ記録用の CLI ツールと、組み込みの学習フレームワークがあります。さらに Hugging Face という、機械学習データ版 GitHub みたいなサービスにも自動アップロードできます。

こちらが[チャレンジ公式ドキュメント](https://github.com/intrinsic-dev/aic/tree/main/aic_utils/lerobot_robot_aic)の例です。下へスクロールすると学習方法が出てきて、その先で LeRobot ドキュメントに飛びます。

```Bash
cd ~/ws_aic/src/aic
pixi run lerobot-record \
  --robot.type=aic_controller --robot.id=aic \
  --teleop.type=<teleop-type> --teleop.id=aic \
  --robot.teleop_target_mode=<mode> --robot.teleop_frame_id=<frame_id> \
  --dataset.repo_id=<hf-repo> \
  --dataset.single_task=<task-prompt> \
  --dataset.push_to_hub=false \
  --dataset.private=true \
  --play_sounds=false \
  --display_data=true
```

実際に記録されるデータは、今のところ見る限り次のようなものです:

* gripper tip の線形・角速度
* 3 台のカメラのフィード

[自分の Hugging Face dataset](https://huggingface.co/datasets/rockyshao22/Intrinsic_AI/viewer/default/train) で見ると、こんな感じです:
![image](https://us1.discourse-cdn.com/flex022/uploads/ros/optimized/3X/a/b/abbe5e0eecffbb2c0e5b5ecdf33066afd1ec5acc_2_690x442.png)

今のところ、自分が確認できたのは「データ記録が動くこと」だけで、集めたデータが本当に正しいかまではまだ検証できていません。

現在は、将来の ACT model のために、ground-truth ベースの teleop を使って、優雅で、1 回目で、100% 成功する挿入データを集める方法を作ろうとしています（LeRobot には組み込みの学習ツールがあるので）。

自分は "garbage in, garbage out" の流儀をかなり強く信じています。笑

## 投稿 15
- 投稿番号: `19`
- 投稿日時: `2026-03-19T17:47:35.494Z`
- 元投稿: [Discourse Post #19](https://discourse.openrobotics.org/t/rockys-open-source-build-thread-ai-for-industry-challenge/53155/19)

Ludvig さん、ほんとうにありがとうございます！
優しい言葉と寛大なお申し出にとても感謝しています。自分にとっては本当にゲームチェンジャーになりそうです。

LinkedIn でも連絡しました。ぜひもっとお話しできたらうれしいです！
Rocky Shao
