# Submission guidelines

## Introduction

**AI for Industry Challenge** へようこそ。この文書では、評価用にソリューションを package 化し、container 化し、upload するための技術要件を説明します。ここで示す手順に従うことで、model が自分のローカル環境とまったく同じ形で自動評価環境上でも動作するようになります。

> [!IMPORTANT]
> registry への upload を完了するには、team leader 宛の **onboarding email** で提供される credential が必要です。これには、固有の AWS access credential と、team に割り当てられた ECR Repository URI が含まれます。

> [!NOTE]
> ローカルで使っている repo 固有の daily operating procedure（branch freeze、artifact vendoring、ローカル再検証、submission log の取り方など）については、[Daily Submission Workflow](../docs/daily_submission_workflow.md) を参照してください。

---

## 1. image を準備して build する

すべての提出物は、Docker や Podman のような OCI 準拠 image builder を使って container 化する必要があります。project は、すべての policy logic と依存関係要件を custom policy package 内に直接置く形で構成してください。

追加 package や依存関係がない場合は、policy code を [policy.py](../aic_model/aic_model/policy.py) に置いたまま、`aic_model` directory とその [Dockerfile](../docker/aic_model/Dockerfile) を再利用できます。この場合は Dockerfile を更新し、`CMD ["--ros-args", "-p", "policy:=aic_example_policies.ros.CheatCode", "-p", "use_sim_time:=true"]` を `CMD ["--ros-args", "-p", "policy:=aic_model.MyPolicy", "-p", "use_sim_time:=true"]` に変更してください。その後、[Build the Image](#build-the-image) セクションへ進めます。

出発点として、サンプル `aic_model` Dockerfile を使うことを強く推奨します。

```bash
mkdir -p docker/my_policy
cp docker/aic_model/Dockerfile docker/my_policy/
```

その後、custom policy package を追加するために `docker/my_policy/Dockerfile` を編集します。

```dockerfile
# 他の依存関係を追加
COPY my_policy_node /ws_aic/src/aic/my_policy_node # <-- この行を追加
```

policy を実行するよう `CMD` を編集します。

```dockerfile
CMD ["--ros-args", "-p", "policy:=my_policy_node.MyPolicy"]
CMD ["--ros-args", "-p", "policy:=my_policy_node.MyPolicy", "-p", "use_sim_time:=true"]
```

### `docker-compose.yaml` を更新

`docker/docker-compose.yaml` を開き、model service の設定があなたの Dockerfile と policy を使うよう更新します。

```yaml
    model:
        image: my-solution:v1
        build:
            dockerfile: docker/my_policy_node/Dockerfile # <-- この行を置き換える
            context: ..
```

<a id="build-the-image"></a>
### image を build する

提出 image を build するには、**root directory** から次のコマンドを実行します。

```bash
docker compose -f docker/docker-compose.yaml build model
```

<a id="verify-locally"></a>
### ローカルで検証する

server へ push する前に、その container が正しく初期化され、期待どおりにデータを処理することを必ず確認してください。

`docker compose` を使ってローカル評価を実行できます。

```bash
docker compose -f docker/docker-compose.yaml up
```

> [!WARNING]
> ローカル検証は省略しないでください。ローカル評価時に container が起動失敗したり crash したりする場合、submission portal でも自動 reject され、その日の submission 上限に影響する可能性があります。

> [!IMPORTANT]
> simulator の内部データ構造を subscribe するだけの最小限の "cheating" solution を防ぐための Zenoh access control については、[Access Control](./access_control.md) を参照してください。

---

<a id="2-upload-your-image-to-our-registry"></a>
## 2. image を registry に upload する

team 用 OCI image のホスティングには Amazon Elastic Container Registry（ECR）を使用します。[AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html) をインストールしておく必要があります。

<a id="authenticate"></a>
### 認証

team leader 宛の onboarding email で提供された credential を使って、ローカル環境を設定します。

#### A. AWS Profile を設定
次のコマンドを実行します。`<team_name>` は email に記載された slug（例: `team123`）に置き換えてください。

```bash
aws configure --profile <team_name>
```

prompt が出たら、以下を入力します。

- **Access Key ID:** （email からコピー）
- **Secret Access Key:** （email からコピー）
- **Default region name:** us-east-1
- **Default output format:** json（または Enter で既定値）

#### B. Environment Variable を設定

以降の command が正しい credential を使うよう、shell を新しい profile に向けます。

```bash
export AWS_PROFILE=<team_name>
```

#### C. registry に認証

最後に、ローカル Docker client を private registry に対して認証します。

```bash
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 973918476471.dkr.ecr.us-east-1.amazonaws.com
```

### image に tag を付ける

ローカル image は、team に提供された remote repository URI に一致するよう tag を付ける必要があります。以下のダミー URI を、実際の team 用 URI に置き換えてください。

```bash
docker tag localhost/my-solution:v1 973918476471.dkr.ecr.us-east-1.amazonaws.com/aic-team/<team_name>:v1
```

> [!IMPORTANT]
> ECR registry 上の image tag は immutable です。既存 tag の上書きはできません。新しい submission や build ごとに、version tag（例: `:v2`, `:v3`）を増やすか、Git commit SHA のような一意識別子を使う必要があります。すでに存在する tag で push すると失敗します。

### image を push する

tag 付けした image を challenge registry に upload します。

```bash
docker push 973918476471.dkr.ecr.us-east-1.amazonaws.com/aic-team/<team_name>:v1
```

---

## 3. submission を登録する

ECR へ image を push しただけでは評価は始まりません。新しい version が採点可能であることを platform に通知する必要があります。

> [!NOTE]
> submission portal はまもなく公開予定です。portal の login credential は 3 月末までに team leader 宛へ送付されます。

1. 先ほど push した完全な Image URI（例: `973918476471.dkr.ecr.us-east-1.amazonaws.com/aic-team/<team_name>:v1`）をコピーする
2. submission portal にログインする
3. `AI for Industry Challenge` をクリックし、`Submit` へ進む
4. `Qualification` phase を選び、submission の `OCI Image` field に URI を貼り付ける
5. `Submit` をクリックして進める

---

### 4. 評価を監視する

OCI Image URI を登録すると、platform は専用の隔離された評価環境へ container を起動します。このプロセスは自動化されていますが、portal の monitoring dashboard から進行状況を追跡できます。

#### Dashboard へのアクセス
1. portal の **My Submissions** page へ移動する
2. "Phase" dropdown で `Qualification` filter を適用し、現在の提出物だけを見る
3. table の先頭にある最新 submission を見つける

#### 評価ライフサイクル

**Status** 列には、評価 cluster 内で container がたどる状態がリアルタイムで表示されます。これを理解することは、1 日の submission 上限を管理するうえで重要です。

| Status | Technical Context |
| :--- | :--- |
| **Submitted** | platform が Image URI を受け取った状態 |
| **Queued** | execution buffer 内で待機中。cluster 上で利用可能な評価 node を待っている状態 |
| **Running** | image が ECR から pull され、ROS 2 node が simulation 環境で challenge logic を実行中 |
| **Finished** | 評価が自然終了し、成功指標が計算されて Leaderboard に反映された状態 |
| **Failed** | container が途中終了した状態。多くは runtime crash（例: Python `ImportError`）、依存関係不足、または timeout を意味する |

> [!TIP]
> cluster の負荷や policy の複雑さにもよりますが、**Queued** から **Finished** への遷移は通常 **5〜15 分** 程度です。status が "Queued" または "Running" の間に再提出する必要はありません。page を refresh して最新状態を確認してください。

---

## FAQs

**I cannot use the example dockerfile**: サンプル dockerfile は `aic_model` を使って policy を実行する前提です。`aic_model` を使わない場合は [create a custom dockerfile](./custom_dockerfile.md) を参照してください。

**My push failed with "no basic auth credentials"**: Docker の login session が期限切れになっている可能性があります。ECR login token の有効期限は 12 時間です。Section 2 の [Authenticate](#authenticate) をもう一度実行してください。

**Where can I see my results?** 過去の結果や log はすべて portal の "My submissions" セクションで確認できます。Leaderboard を見れば他 team との比較もできます。

**Can I submit multiple times?** はい。ただし 1 日あたり 1 submission に制限されています。競技期間全体での総 submission 回数には制限はありません。

---

## Questions?

- **Issues**: 問題報告は [GitHub Issues](https://github.com/intrinsic-dev/aic/issues)
- **Community**: 議論への参加は [Open Robotics Discourse](https://discourse.openrobotics.org/c/competitions/ai-for-industry-challenge/)
