# Scoring

各 trial は tiered scoring system で採点されます。score は全 trial にわたって累積されます。

## Scoring Tier 概要

| Tier | Name | Description |
|------|------|-------------|
| Tier 1 | Model Validity | model がロードされ、期待どおりに振る舞うことを確認する前提チェック |
| Tier 2 | Performance & Convergence | motion quality に関する定量評価 |
| Tier 3 | Cable Insertion | 主目的。完全挿入または部分挿入を確認 |

## Tier 1: Model Validity（前提条件）

提出物が error なくロードされ、正常に動作することを確認する sanity check です。

- model は提出した policy を正常に activate し、`InsertCable` action request に応答できなければなりません。また提出 policy は `MotionUpdate`（target position / velocity）または `JointMotionUpdate`（target joint state）を通じて、ロボットアーム controller に有効な command を送る必要があります。
- policy は [Challenge Rules](./challenge_rules.md#aic_model) に定義されたすべての挙動要件に準拠していなければなりません
- このチェックに失敗した提出物は採点されません

| Outcome | Score |
|---------|-------|
| Validation passed | 1 |
| Validation failed | 0 |

## Tier 2: Performance & convergence

task 実行中のロボット motion 品質を測る定量指標です。

### Trajectory smoothness（0-6 points）

エンドエフェクタ軌道の滑らかさを測定します。jerk が低いほど、より滑らかで制御された motion であることを示します。jerk はアームが動いているとき（speed > 0.01 m/s）にのみ蓄積されるため、静止時間によって平均が薄まることはありません。task が成功した場合、または plug の最終位置が target port に十分近い場合（Tier 3 score > 0）にのみ加点されます。

- **Metric**: linear jerk magnitude（m/s³）の時間重み付き平均。Savitzky–Golay filter（15 サンプル window 上の局所 2 次多項式近似）で計算
- **Scoring**: jerk に反比例
  - Jerk = 0 m/s³ → 6 points（最大）
  - Jerk ≥ 50 m/s³ → 0 points（最小）
  - 閾値間は線形補間
- **Not awarded**: plug の最終位置が target port から許容最大距離の外にある場合は 0 points（Tier 3 score <= 0）

### Task duration（0-12 points）

task 完了が速いほど高得点になります。task が成功した場合、または plug の最終位置が target port に十分近い場合（Tier 3 score > 0）にのみ加点されます。

- **Metric**: task 開始から task 終了までの経過時間
- **Scoring**: duration に反比例
  - Duration ≤ 5 seconds → 12 points（最大）
  - Duration ≥ 60 seconds → 0 points（最小）
  - 閾値間は線形補間
- **Not awarded**: plug の最終位置が target port から許容最大距離の外にある場合は 0 points（Tier 3 score <= 0）

### Trajectory efficiency（0-6 points）

task 実行中にエンドエフェクタが移動した総距離を測定します。より短く直接的な経路ほど高得点です。task が成功した場合、または plug の最終位置が target port に十分近い場合（Tier 3 score > 0）にのみ加点されます。

- **Metric**: end-effector position の累積 Euclidean distance（meters）
- **Scoring**: 総経路長に反比例
  - Path length ≤ 初期 plug-port 距離 → 6 points（最大）
  - Path length ≥ 1 m + 初期 plug-port 距離 → 0 points（最小）
  - 閾値間は線形補間
- 完全点に必要な最小 path length は、trial 開始時の plug と port の初期 Euclidean distance に基づいて動的に設定されます
- **Not awarded**: plug の最終位置が target port から許容最大距離の外にある場合は 0 points（Tier 3 score <= 0）

### Insertion force penalty（0 から -12 points）

穏やかな操作を促すため、挿入中の過大 force を減点します。
force sensor の読み値は起動時にゼロ点調整されるため、baseline は 0 N 近傍です。

- **Force threshold**: 20 N
- **Duration threshold**: 1 second
- **Penalty**: force が duration threshold を超えて閾値を超えた場合、-12 points
- **No penalty**: 過大 force が検出されない、または閾値超過が duration threshold 以内の場合

### Off-Limit contact penalty（0 から -24 points）

環境の制限領域（enclosure または task board）との衝突を減点します。

- **Penalty**: off-limit entity との接触が 1 回でも検出されると -24 points
- **No penalty**: 禁止された接触が発生しない場合

## Tier 3: Task Success

ケーブル挿入成功を確認する主目的です。scoring では 2 段階のアプローチを取り、完全挿入と port への部分的進捗の両方を評価します。

### Successful insertion（-12 から 75 points）

ケーブルコネクタが **正しい** target port に完全挿入された場合、contact sensor により検証されます。

| Outcome | Score |
|---------|-------|
| Correct port insertion | 75 |
| Wrong port insertion | -12 |

### Partial insertion and proximity（0-50 points）

完全挿入が検出されない場合、task 完了時の plug と port の近さに基づいて採点されます。

- **Partial insertion**（38-50 points）: plug が port 入口から port 底面までの bounding box 内にあり（x-y 方向 5 mm tolerance 以内）、挿入深さに比例して score が決まります。より深い挿入ほど高得点です。
- **Proximity**（0-25 points）: plug が port 内にない場合、score は port からの最大許容距離に反比例して決まります。最大距離は、plug と port の初期距離の半分に設定されます。
  - port 入口位置 → 25 points（最大）
  - 最大距離の外側 → 0 points（最小）
  - 閾値間は線形補間

## Total Score Calculation

```
Total Score = Tier 1 + Tier 2 + Tier 3
```

内訳:
- **Tier 1**: 0 または 1 point
- **Tier 2**: smoothness（0-6）、duration（0-12）、efficiency（0-6）、penalty（force: 0 から -12、contact: 0 から -24）の合計
- **Tier 3**: 挿入成功（最大 75）、または partial insertion / proximity score（最大 50）
- **1 trial あたりの最大 score**: 100 points（1 + 6 + 12 + 6 + 75）

## 最終順位

最終順位は、全 trial の score を累積して決まります。Tier 2 の定量的 performance 指標と Tier 3 の task success score を合わせて評価します。

## See Also

各 scoring tier を再現可能に検証する例については [Scoring Test & Evaluation Guide](./scoring_tests.md) を参照してください。
