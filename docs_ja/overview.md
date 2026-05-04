# Challenge Overview

![](../../media/aic_overview.png)

**AI for Industry Challenge** は、現代製造業における重要なボトルネックである電子機器組立を対象にしています。特に、器用なケーブルハンドリングと挿入に焦点を当てています。この作業は、現状では依然として多くが手作業で、かつ反復的です。

ロボティクスの観点では、このタスクは非常に難しいことで知られています。柔軟なケーブルを扱う際の複雑な物理と、コネクタを認識・把持・挿入するために必要な極めて高い精度の両方が要求されるためです。

**目標:**
参加者は、ROS を通信基盤として利用しながら、open-source simulator（例: Isaac Sim、MuJoCo、Gazebo）を用いて AI model を学習します。これは sim-to-real gap を埋め、実世界で重要な問題に対して実質的な前進を生み出す機会です。

**報酬:**
Finalist は、Intrinsic 本社にある physical workcell へ自分たちの model を simulation から展開します。上位 5 チームで **$180,000 の prize pool** を分配します。

---

## フェーズ

この challenge は **2026 年 3 月 2 日** に開始し、**2026 年 9 月 8 日** まで続きます。3 つの distinct phase で構成されます。

* **Qualification (3/2 - 5/15):** 参加者は simulation 内でケーブル組立 model を学習・テストします。評価期間は 5/18 - 5/27、Top 30 の発表は 5/28。
* **Phase 1 (5/28 - 7/14):** 通過チームは Intrinsic Flowstate にアクセスし、完全なケーブルハンドリングソリューションを開発します。評価期間は 7/14 - 7/21、Top 10 の発表は 7/22。
* **Phase 2 (7/27 - 8/25):** 上位チームは Intrinsic 提供の physical workcell 上でソリューションを展開・洗練し、実環境で評価されます。評価期間は 8/26 - 9/4、優勝者の発表は 9/8。

期待事項と deliverable の詳細は [Competition Phases](./phases.md) を参照してください。

## 評価

3 フェーズすべてで scoring は自動化されています。順位は次の基準の組み合わせで決まります。

* **Model Validity:** 提出物が error なしでロードされ、必要な ROS topic に対して有効なロボット command を生成できること。無効な提出物は失格になります。
* **Task Success:** ケーブル挿入成功ごとに適用される binary 指標。
* **Precision:** コネクタが目標 pose にどれだけ近く挿入されたかに基づく score。
* **Safety:** collision や、connector / cable に加えた過大な force に対する penalty。
* **Efficiency:** 組立タスク全体を完了する cycle time の計測。速いソリューションほど高く評価されます。

詳細は [Scoring](./scoring.md) と [Scoring Test & Evaluation Guide](./scoring_tests.md) を参照してください。

## 提出

challenge を進み、賞金対象であり続けるには、各 phase の終了時に model を提出する必要があります。
* **Authentication:** 各 team leader には、upload 用の固有 authentication token が配布されます。
* **Frequency:** 締切前であれば複数回提出できます。最終提出物が scoring に使われます。

upload 手順の詳細は [Submission Guidelines](./submission.md) を参照してください。

## ベースライン Policy

開始しやすいように、最小例、デバッグ用の ground truth ベース policy、ACT（Action Chunking with Transformers）policy を含む複数の baseline policy 実装を提供しています。

これらの policy の実行方法については [Example Policies README](./aic_example_policies/README.md) と [Policy Integration Guide](./policy.md#baseline-policies) を参照してください。

---

## Getting Started

始める準備ができたら、[Getting Started Guide](./getting_started.md) を参照してください。
