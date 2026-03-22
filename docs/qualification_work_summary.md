# 予選フェーズ作業サマリ

## 目的

この文書は、詳細な実験ログよりも一段高いレベルで、これまでの予選フェーズ作業を要約する。

このサマリの目的は、次の問いに答えやすくすることにある。

- これまで何を試してきたか
- どの run がスコア改善につながり、どれがつながらなかったか
- 実験で実際に裏づけられた結論は何か
- 現在の blocker は何か
- 次に何をすべきか

このファイルは、意図的に
[qualification_experiment_log.md](/home/masa/ws_aic/src/aic/docs/qualification_experiment_log.md)
および
[qualification_strategy_notes.md](/home/masa/ws_aic/src/aic/docs/qualification_strategy_notes.md)
と内容を重ねている。実験ログは生の時系列記録であり、strategy notes は設計と計画の文書である。このファイルは、その両者をつなぐ叙述的な橋渡しである。

## 現時点の結論

現在の状態は次のとおり。

- 提出可能な構成での最高スコアは、依然として `126.58206055565613 / 300`
- その best run は `S2` で、保存先は
  `/home/masa/aic_results/qual_submission_safe_v6_20260321_220649/scoring.yaml`
- `SFP` は十分に強く、繰り返し 40 点台後半に着地している
- `SC` は依然として主な bottleneck である
- `center_uvz-only` は本線戦略としては失敗したと判断している
- 最初の ground-truth 教師実現性ループもゲートを通過できなかった

したがって、現時点でのプロジェクト結論は次のとおり。

- `center_uvz-only` の調整は続けない
- まだ生徒データを追加収集しない
- まず教師/controller を再設計する
- 教師自身が挿入主導になるまで、教師誘導の生徒学習は再開しない

## 重要な運用上の事実

実運用では、次の環境上の事実が重要だと分かった。

- GUI を使う実行では、明示的に `DISPLAY=:1` を設定するべき
- `src/aic` 配下の package を変更した場合、検証前に rebuild が必要
- `distrobox enter -r aic_eval` は、この環境では内部で対話式の `sudo`
  を使おうとするため扱いにくかった
- `docker exec` は実用的な代替手段であり、ドキュメントもその方向へ
  修正した

これらはローカルの workflow メモや docs 更新にも反映済みである。

## 初期に確認されたこと

本格的な policy 反復に入る前に、いくつかの環境・scoring 問題を確認した。

1. docs に書かれた end-to-end pipeline を実行できた。
2. `DISPLAY=:1` による GUI 起動が機能した。
3. 以前の `Tier 3 tf between cable and port not found` 問題は、source state
   だけでなく古い evaluation artifact にも起因していると分かった。
4. 一部の upstream fix を実際に反映させるには、evaluation path 側で関連
   package を rebuild する必要があった。

これが重要だったのは、環境問題と実際の policy 性能問題を切り分けられたからである。

## スコアの現実

後続の計画に大きく影響した重要な確認事項がある。

- スコアは合計 `100` ではなく、trial ごとに `100`
- ここで扱う qualification sample run は通常 `3` trial なので、意味のある
  総スケールは `300`
- したがって `120` 前後は「かなり良い」ではなく、本気の提出に必要な
  trial あたり `80+` の水準にはまだ遠い

この認識が、作業の重心を「小さな改善」から「挿入の崖を越える」方向へ移した。

## 全体タイムライン

作業は自然にいくつかのフェーズに分かれた。

1. `public-sample` 実験と開発基盤の整備
2. 最初の提出可能ベースライン
3. より良い legal `SC` acquisition
4. 学習ベースの acquisition 実験
5. `center_uvz` 停滞後のピボット計画
6. 最初の教師実現性実験

以下で各フェーズを説明する。

## フェーズ1: `public-sample` 実験と開発基盤の整備

### `public-sample` リプレイ参照

初期には、参照用として強い `public-sample` 指向の replay policy を構築した。

- Score: `202.68080794003197`
- これは「`public-sample` 特化 policy をローカルで十分に詰めるとどこまで
  出せるか」の上限参照として有用だった
- これは提出可能な構成とは見なしていない

このフェーズの価値は、スコア自体の提出価値ではなく、初期の legal baseline
よりはるかに高いローカルスコアを環境側が出せることを示した点にあった。

### M0 から M7

`M0-M7` の並びは、開発指向の milestone ladder だった。

`M0-M7` の主眼は、提出可能性に厳密にこだわる前に、可観測性を整備し、
制御された形でアイデアを検証することだった。

代表的なスコアは以下のとおり。

| マイルストーン | 目的 | スコア |
| --- | --- | ---: |
| `M0` | 可観測性ハーネス | `3` |
| `M1` | 開発用 target controller ハーネス | `100.11421244878404` |
| `M2` | 最初の `SFP` center-camera localizer | `3` |
| `M3` | 最初の task-conditioned `SFP` template tuning | `-21` |
| `M4` | `public-sample` full pipeline baseline | `93.084972039114518` |
| `M5` | multi-camera late fusion | `109.96488164803876` |
| `M6` | `SC` force-guided refinement | `110.0571545895495` |
| `M7` | fresh-observation gating 付き residual refinement | `111.43384153152546` |

`M0-M7` から得られた主な知見は次のとおり。

- 可観測性と debug artifact への投資は価値があった
- 古い observation の扱いと sim-time pacing が重要だった
- `SFP` はかなり安定して押し上げられた
- `SC` は依然として挿入を拒み続けた
- 開発専用の path は診断には有用だったが、最終 policy には使えなかった

このフェーズの最終結論は、開発用足場は有用だったが、プロジェクトには依然として本物の提出可能トラックが必要だということだった。

## フェーズ2: 最初の提出可能ベースライン

### S0: 最初の `legal-only` end-to-end run

`S0` は、初めて実際の提出可能 baseline として扱われた run だった。

- Score: `98.041744964269498`
- trial ごと:
  - `t1=48.061318260132214`
  - `t2=48.980426704591785`
  - `t3=1.0`

`S0` が示したこと:

- 許可された実行時入力だけで 3 trial すべてを完了できる
- legal な `SFP` path はすでにかなり安定している
- `SC` が明確な弱点である

これは最初の大きな転換点だった。問題はもはや「legal pipeline を作れるか」
ではなく、「なぜ legal な `SC` acquisition はここまでひどく失敗するのか」
になった。

### S1: triangulated `SC` translation-only acquisition

`S1` では、三角測量ベースの translation-only stage により、legal な `SC`
acquisition を改善した。

- Score: `122.08548930859087`
- trial ごと:
  - `t1=48.72334879624358`
  - `t2=48.987096754294114`
  - `t3=24.375043758053183`

`S1` が示したこと:

- legal な `SC` acquisition でも意味のあるスコア増分を出せる
- 以前の `SC` orientation heuristic の一部は、かえって悪化要因だった
- 挿入に至らなくても、より良い legal acquisition によって `SC` を
  Tier 1 only よりかなり先へ進められる

これにより、legal framework 自体が問題ではないことが明確になった。

### S2: learned `SC` acquisition

`S2` は、現在の提出可能ベスト結果になった。

- Score: `126.58206055565613`
- trial ごと:
  - `t1=48.21030515038397`
  - `t2=48.982548911699574`
  - `t3=29.389206493572582`

`S2` が示したこと:

- 学習ベースの legal `SC` acquisition は、最良の hand-crafted legal version
  を上回れる
- legal path は `~100` に張り付いているわけではない
- ただし、システム全体は依然として挿入主導ではなく近接主導である

これが現在の公式な提出可能ベスト参照である。

## フェーズ3: 全体像を変えなかった小規模な legal 追試

### X1: `tool-frame` force search probe

分離した `SC` の `tool-frame` force search を試した。

- Score: 分離 probe 上で `33.404680405165706`

ここから分かったこと:

- force search という発想自体は無意味ではない
- ただし、すぐに本線へ昇格させるには遅すぎ、壊れやすすぎる
- `S2` を置き換える根拠にはならなかった

ここでの教訓は、局所的な force search だけでは欠けているピースにならない、
ということだった。

## フェーズ4: `center_uvz` 学習ベース `SFP` ルート

legal `SC` acquisition の作業の後、次の大きな分岐は、学習ベースの `SFP`
acquisition、特に `center_uvz` target representation に集中した。

### データ収集

より大きなランダム化 `SFP` dataset を収集した。

- `20` randomized trial
- `840` 総サンプル
- 5 本の `SFP` rail すべてでバランスを取った
- phase には initial、teacher hover、teacher insert sweep を含めた

これは、`center_uvz` が本当に legal な本線になりうるかを試すためだった。

### 訓練

その dataset で、より大きな `center_uvz` model を訓練した。

- training は `160` epoch まで実行
- best checkpoint は `epoch 152`
- best validation loss は `0.006016482987130682`
- best validation metrics は概ね次の範囲だった
  - `u ~= 4.18 px`
  - `v ~= 11.59 px`
  - `depth ~= 2.42 mm`

紙の上では、これは無視できない改善だった。

### 実行時バグと修正

重要な runtime bug が見つかった。

- 学習済み model は runtime auxiliary feature を期待していた
- しかし runtime inference では、それが実際には入力されていなかった
- そのため、無意味な予測、あるいはまったく動かない挙動が起きていた

この bug を修正し、学習 branch を再評価した。

### P0 ゲート結果

再訓練と runtime bugfix の後でも、より大きな `center_uvz` ルートは本線
ゲートを通過できなかった。

- Score: `71.320068644735088 / 200` on `SFP-only`
- trial ごと:
  - `t1=35.220362710871615`
  - `t2=36.099705933863473`

これが重要だった理由:

- これは `S2` の旧 `SFP` pair baseline よりかなり低い
- model は動ける程度には acquisition を改善したが、挿入主導の scoring に
  入るには足りなかった
- そのため、`center_uvz-only` は本線ルートとして失敗と見なしている

これは 2 回目の大きな転換点だった。

## フェーズ5: 教師誘導による挿入へのピボット

`center_uvz-only` が失敗した後、戦略は「点をよりうまく予測する」ことから
離れ、「挿入そのものを学習可能にする」方向へピボットした。

このピボットの核となる考え方は次のとおり。

- 学習で局所的な挿入フレームを推定する
- そのフレーム内で、教師誘導の近接接触 policy を使う

ここで重要だった概念上の転換は次のとおり。

- 問題は acquisition だけではない
- 真の bottleneck は、近接から挿入へ移ることにある

このピボットは、
[qualification_strategy_notes.md](/home/masa/ws_aic/src/aic/docs/qualification_strategy_notes.md)
の `Pivot Plan If center_uvz Stalls` セクションに明示的に書き込んである。

## フェーズ6: 最初の教師実現性ループ

ピボット計画は `T0` から始めた。生徒を訓練する前に、ground-truth 教師
そのものが randomized `SFP` で十分なスコアを出せるか確認するためである。

これは意図的な順序だった。教師が安定して挿入できないなら、生徒ラベルを
さらに集めるのは時期尚早である。

### T0 v0

最初の GT 教師実現性 run では、randomized `SFP-only` config を使った。

- Score: `86.653740214899685 / 200`
- trial ごと:
  - `t1=41.779318087841951`
  - `t2=44.874422127057734`

ここから分かったこと:

- 教師は両方の task を完了できた
- port の口元近くまでは到達できた
- しかし挿入イベントは起こせなかった

つまり GT を使っても、controller は依然として近接主導だった。

### T0 v1

2 回目の教師実現性 run では、最終 push の前に GT terminal-contact loop を
追加した。

- Score: `86.562241559624113 / 200`
- trial ごと:
  - `t1=42.659702234323316`
  - `t2=43.902539325300786`

これは実質的に `T0 v0` と変わらなかった。

これが意味すること:

- 最後の固定 push を少し GT-aware な loop に置き換えても、結果は十分には
  変わらなかった
- 問題は小さな局所パッチでは解決しない
- 教師自体に、より根本的な再設計が必要である

これは 3 回目の大きな転換点だった。

## 現時点の証拠が実際に示していること

現時点では、いくつかの点について証拠はかなり強い。

### 1. Legal runtime input は制約要因ではない

`S0`、`S1`、`S2` は、legal runtime input だけでも無視できない性能が出せる
ことを示した。

### 2. `SFP` は未解決だが、最悪の部分ではない

legal run を繰り返しても、`SFP` はおおむね 40 点台後半を維持している。
これは十分ではないが、現在の `SC` path よりは明らかに強い。

### 3. `SC` は提出可能 path における主 blocker のままである

legal な total の best は、依然として `SC` が挿入に届かないことにより頭打ちになっている。

### 4. `center_uvz-only` は正しい本線ではない

より良いデータと runtime bug 修正があっても、`SFP-only` ゲートを通過できなかった。

### 5. 現在の GT 教師も十分ではない

これが最近の結論として最も重要である。

現在の GT 教師は:

- 近接接触までは到達する
- task は完了する
- しかし挿入しない

つまり、次の bottleneck は生徒 model の容量ではなく、教師/controller 設計である。

## 何を確認済みで、何にまだ追加作業が必要か

### すでに確認したもの

ここまでの作業では、次を使って確認してきた。

- scoring output
- `/home/masa/ws_aic_runtime/qualification_debug` 配下の debug snapshot
- 各 run の log
- bag の有無と bag metadata

最新の `T0` run については、bag metadata から有用な signal が存在することも確認できている。

- `/aic_controller/pose_commands`
- `/aic_controller/controller_state`
- `/fts_broadcaster/wrench`
- `/tf`
- `/scoring/tf`

bag metadata の例:

- `trial_1` bag duration: `60.869616868 s`
- `trial_1` pose commands: `213`
- `trial_1` controller_state messages: `1098`
- `trial_1` wrench messages: `109`
- `trial_2` bag duration: `76.147619956 s`
- `trial_2` pose commands: `426`
- `trial_2` controller_state messages: `2215`
- `trial_2` wrench messages: `222`

### まだより深い作業が必要な点

次のループでは、最近のいくつかの反復よりも、より明確に bag-first で進めるべきである。

特に、次の controller 再設計は以下に基づくべきである。

- 近接接触付近での pose command と実際の動きの関係
- `wrench` がどこで立ち上がり、その立ち上がりが有効な挿入に対応しているのか、
  それとも無意味な衝突に対応しているのか
- contact 後に end-effector が意図した挿入軸に沿って前進しているか
- 最終の教師 phase がジャーク過多ではないか、あるいはアライメントがずれて
  いないか

これは「あとで気が向いたら見よう」という話ではなく、明確な次ステップ要件である。

## なぜ今すぐ生徒データを増やすべきではないのか

データ収集の infrastructure はすでにあるため、学習データを増やし続けること自体は簡単である。しかし、それは現時点の優先事項ではない。

現在の判断根拠は次のとおり。

1. `center_uvz-only` はすでに大幅なデータ増強を受けたにもかかわらず、ゲートに失敗した。
2. GT 教師も自身のゲートに失敗した。
3. したがって、今すぐ生徒ラベルをさらに集めても、真の bottleneck 解消にはつながりにくい。

そのため、現在のデータ方針は次のとおり。

- 今は point-label acquisition の追加を優先しない
- GT 教師が挿入主導になるまで、大規模な生徒データ収集は再開しない
- それが実現したら、疎な target label だけでなく、高密度な教師誘導 insertion trace を収集する

## 次の PDCA ループで重視すべきこと

次のループは、「また小さな tweak を試す」ではない。

次にやるべきことは以下である。

1. 最新の `T0` bag と debug artifact を、近接接触 phase に注目して解析する
2. GT 教師を次の性質を持つよう再設計する
   - 低ジャーク
   - 軸を意識している
   - リカバリ可能である
   - 明示的に挿入指向である
3. randomized `SFP-only` で教師実現性を再実行する
4. 教師誘導の生徒作業を再開する前提として `>= 150 / 200` を要求する

このゲートを通過して初めて、プロジェクトは次へ進むべきである。

- 成功した教師挿入データの高密度収集
- 局所挿入フレームの生徒学習
- 制約付きの教師誘導挿入 policy 訓練

## スコア表

以下の表が、ここまでで最も有用な高レベルのスコア要約である。

| ラベル | 意味 | スコア | 補足 |
| --- | --- | ---: | --- |
| public replay ref | `public-sample` 指向の参照 | `202.68080794003197` | 提出可能ではない |
| `M0` | 可観測性ハーネス | `3` | debug-first の足場 |
| `M1` | 開発用 target controller ハーネス | `100.11421244878404` | 開発専用 |
| `M2` | 最初の `SFP` center localizer | `3` | まだ有効な実行なし |
| `M3` | 最初の task-conditioned `SFP` tuning | `-21` | 大きく悪化 |
| `M4` | `public-sample` full-pipeline baseline | `93.084972039114518` | なお開発寄り |
| `M5` | multi-camera late fusion | `109.96488164803876` | 改善したがまだ開発寄り |
| `M6` | `SC` force-guided refinement | `110.0571545895495` | 改善は最小 |
| `M7` | fresh observation gating 付き residual refine | `111.43384153152546` | なお提出可能ではない |
| `S0` | 最初の提出可能 baseline | `98.041744964269498` | `legal-only` pipeline の存在を確認 |
| `S1` | triangulated legal `SC` acquisition | `122.08548930859087` | 大きな legal 改善 |
| `S2` | learned legal `SC` acquisition | `126.58206055565613` | 現在の提出可能ベスト |
| `X1` | legal `SC` force-search probe | `33.404680405165706` | 分離した `SC` probe |
| `P0` | より大きくバランスした `center_uvz` ゲート | `71.320068644735088 / 200` | `SFP-only` 本線ゲート失敗 |
| `T0 v0` | randomized GT teacher feasibility | `86.653740214899685 / 200` | GT teacher はまだ挿入主導ではない |
| `T0 v1` | terminal contact loop 付き GT teacher | `86.562241559624113 / 200` | 意味のある改善なし |

## 現時点の判断

現時点の判断は次のとおり。

- `S2` を提出可能ベストの参照として維持する
- `center_uvz-only` を本線として扱うのをやめる
- 教師が改善するまで生徒データの増量を止める
- 次に進む前に GT 教師を再設計する

以上が、ここまでの作業を最も正確に要約した内容である。
