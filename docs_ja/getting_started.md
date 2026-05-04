# Getting Started

AI for Industry Challenge へようこそ。このガイドに従って toolkit の構成を把握し、環境を準備し、クイックスタートの例を実行してセットアップを確認してから、自分のソリューション開発に進んでください。

> [!NOTE]
> **ROS 2 Distribution:** 提出物の公式評価はすべて **ROS 2 Kilted Kaiju** 上で実施されます。異なる ROS 2 distribution（例: Humble や Jazzy）で policy を開発またはテストする場合、互換性確保とサポートはすべて参加者の責任になります。**異なる distribution 間の通信は保証されず、公式にはサポートされません。** ROS 2 に不慣れな場合は、まず [公式 ROS 2 チュートリアル](https://docs.ros.org/en/kilted/Tutorials.html) を一通り終えることを強く推奨します。

## アーキテクチャ概要

この challenge は 2 コンポーネント構成を採用しています。

1. **Evaluation Component**（提供済み）- シミュレーション、ロボット、センサー、スコアリングシステムを実行
2. **Participant Model**（参加者が実装）- センサーデータを処理し、ロボットへコマンドを送るあなたの ROS 2 node

**両コンポーネントの source code はこの toolkit に含まれています。** Evaluation Component は競技期間中に変わらないため、再利用できるよう Docker image（`aic_eval`）を提供しています。これが **推奨ワークフロー** です。source から build したい上級ユーザーは [Building from Source](./build_eval.md) を参照してください。

アーキテクチャ、package、interface の詳細説明については、README の [Toolkit Architecture](./README.md#toolkit-architecture) を参照してください。

---

## 要件

**最小計算環境:**

- **OS:** Ubuntu 24.04
- **CPU:** 4-8 cores
- **RAM:** 32GB+
- **GPU:** NVIDIA RTX 2070+ または同等品
- **VRAM:** 8GB+

> [!NOTE]
> GPU なしでも challenge は実行できますが、性能は大きく低下します。CPU のみのシステム向け最適化のヒントは [Troubleshooting](./troubleshooting.md#no-gpu-available) を参照してください。

**クラウド評価インスタンス:**

クラウド評価では、すべての参加者提出物が次の仕様の同一インスタンスタイプで評価されます。

- **vCPU:** 64 cores
- **RAM:** 256 GiB
- **GPU:** 1 x NVIDIA L4 Tensor Core
- **VRAM:** 24 GiB

---

## セットアップ

まず、次のツールをインストールしてください。
* [Docker](#setup-docker)（必須）
* [Distrobox](#setup-distrobox)（必須）
* [Pixi](#setup-pixi)（必須）
* [NVIDIA Container Toolkit](#setup-and-configure-nvidia-container-toolkit)（任意 - NVIDIA GPU ユーザー向け）

<a id="setup-docker"></a>
### Docker をセットアップ

1. 使用中のプラットフォーム向け [Docker Engine](https://docs.docker.com/engine/install/) をインストールします。
2. [Linux post-installation steps for Docker Engine](https://docs.docker.com/engine/install/linux-postinstall/) を完了し、非 root ユーザーで Docker を管理できるようにします。

<a id="setup-and-configure-nvidia-container-toolkit"></a>
### NVIDIA Container Toolkit をセットアップして設定する（任意）

> [!NOTE]
> この手順は NVIDIA GPU があり、最適な性能のために GPU acceleration を使いたい場合にのみ必要です。

1. Docker Engine から NVIDIA GPU にアクセスできるようにするため、[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) をインストールします。

2. インストール後、Docker が NVIDIA runtime を使うよう設定します。
    ```bash
    sudo nvidia-ctk runtime configure --runtime=docker
    sudo systemctl restart docker
    ```

<a id="setup-distrobox"></a>
### Distrobox をセットアップ

`aic_eval` container をホストシステムと密に統合するために [Distrobox](https://distrobox.it/) を使います。Distrobox は package manager からインストールすることを推奨します。利用中の distribution が Distrobox をサポートしているかは [supported distros](https://distrobox.it/#installation) を確認してください。

Ubuntu では次を実行します。
```bash
sudo apt install distrobox
```

他の distribution については [Alternative methods](https://distrobox.it/#alternative-methods) を参照してください。

<a id="setup-pixi"></a>
### Pixi をセットアップ

package と依存関係の管理に [Pixi](https://pixi.prefix.dev/latest/) を使います。ROS 2 もここで管理します。

Ubuntu では次を実行します。
```bash
curl -fsSL https://pixi.sh/install.sh | sh
# インストール後にターミナルを再起動
```

他の OS については [Alternative Installation Methods](https://pixi.prefix.dev/latest/installation/#alternative-installation-methods) を参照してください。

> [!IMPORTANT]
> Pixi 環境内の package 変更は自動では追跡されません。変更を反映するには `pixi reinstall <package_name>` を実行する必要があります。

<a id="quick-start"></a>
## クイックスタート

このセクションでは以下を案内します。
1. **workspace のセットアップ** - challenge repository を clone し、Pixi で依存関係を install
2. **Evaluation Component の実行** - `aic_eval` container を起動し、シミュレーション環境、ロボット、センサー、スコアリングシステムを立ち上げる
3. **サンプル policy の実行** - ローカル workspace から用意されたサンプル policy を評価 container に対して実行する

ここまで完了して提出準備に進む場合は、[Submission Guidelines](./submission.md) を参照して participant workspace の container 化方法を確認してください。

---
### Step 1: workspace をセットアップ

```bash
# この repo を clone
mkdir -p ~/ws_aic/src
cd ~/ws_aic/src
git clone https://github.com/intrinsic-dev/aic

# 依存関係を install して build
cd ~/ws_aic/src/aic
pixi install
```

**想定される結果:**
- Pixi が ROS 2 package と依存関係を download / install する
- 完了時にインストール成功メッセージが表示される
- workspace に、すべての依存関係を含む `.pixi` directory が作成される

---
### Step 2: 評価 container を起動

```bash
# distrobox が Docker を container manager として使うよう指定
export DBX_CONTAINER_MANAGER=docker

# eval container を作成して入る
docker pull ghcr.io/intrinsic-dev/aic/aic_eval:latest
# NVIDIA GPU が *ない* 場合は、GPU サポート用の --nvidia flag を外す
distrobox create -r --nvidia -i ghcr.io/intrinsic-dev/aic/aic_eval:latest aic_eval
distrobox enter -r aic_eval

# container 内で環境を起動
/entrypoint.sh ground_truth:=false start_aic_engine:=true
```

[`entrypoint.sh`](../docker/aic_eval/Dockerfile) script は Zenoh router を実行し、`aic_engine` 付きで [`aic_gz_bringup.launch.py`](./aic_bringup/README.md#1-aic_gz_bringuplaunchpy) を起動します。

> [!NOTE]
> 評価 container は本質的には build 済み workspace であり、`/entrypoint.sh` だけが利用方法ではありません。container に入り、workspace を source（`source /ws_aic/install/setup.bash`）したうえで、各 package README やドキュメントに書かれている任意の command を実行 / launch できます。

**想定される結果:**
- **Gazebo**（simulation）と **RViz**（visualization）の 2 つの window が開く
- Gazebo 上には、テーブルに取り付けられた Universal Robots UR5e manipulator を含む workcell が表示される
- ターミナルには、AIC engine が初期化され `aic_model` node を待っていることを示す log（`No node with name 'aic_model' found. Retrying...`）が出る
- まだロボットは動かない（あなたの policy が接続するのを待っている状態）

![Evaluation Environment](../../media/eval_environment_waiting.png)

シミュレーション環境の詳細は [Scene Description](./scene_description.md) を参照してください。

> [!Note]
> `docker pull` が失敗する場合は、[ghcr.io にログイン](https://docs.github.com/en/packages/working-with-a-github-packages-registry/working-with-the-container-registry#authenticating-with-a-personal-access-token-classic) する必要があるかもしれません。

> [!NOTE]
> 評価 container 内の `aic_engine` node は、30 秒以内に `aic_model` node（Step 3 参照）が見つかることを期待しており、それを過ぎるとタイムアウトします。一方で、Zenoh router は評価 container 側が起動するため、この手順（`/entrypoint.sh`）は Step 3 の `aic_model` node 起動 **前** に実行する必要があります。

---

### Step 3: サンプル policy を実行

シミュレーション環境が起動した状態（Step 2）で、次の policy を実行します。
```bash
cd ~/ws_aic/src/aic
pixi run ros2 run aic_model aic_model --ros-args -p use_sim_time:=true -p policy:=aic_example_policies.ros.WaveArm
```

> [!NOTE]
> `pixi run` は独自の環境を作り、その中で `aic_model` を実行します。そのため `pixi run` 自体は Docker や distrobox の外で実行できます。通常はこちらの方が簡単で速く、扱いやすいです。

`aic_model` node が起動すると、AIC engine は Gazebo window にタスクボードと、グリッパに取り付けられたケーブルを生成します。その後 eval container 側のターミナルで 3 回連続の trial を追跡し、各 score を表示します。詳細は [Scoring](./scoring.md) を参照してください。

**Note:** `WaveArm` policy は、ロボットアームを前後に手を振るように動かすだけのダミー例です。ケーブル挿入タスクを解こうとはしません。この例の目的は、[`aic_engine`](./aic_engine/README.md) が [sample configuration](../aic_engine/config/sample_config.yaml) に基づいてどのように trial を進行させ、その性能に基づいて policy を採点するかを示すことです（この場合、当然スコアは低くなります）。

**想定される結果:**
- **Gazebo 上**: タスクボードと、グリッパに取り付けられたケーブルがシミュレーションに現れる
- **ロボット上**: アームが前後に手を振るように動く
- **eval container ターミナル上**:
  - trial の進行を示す log（Trial 1/3, Trial 2/3, Trial 3/3）
  - 各 trial 後の scoring 情報
  - 全 trial の合計 score を含む最終 summary
- ロボットが自動で 3 回連続の trial を実行する
- **結果の保存先**: `$HOME/aic_results/`（`$AIC_RESULTS_DIR` が設定されていればその値）

![Wave Arm Policy](../../media/wave_arm_policy.gif)

ロボットが動かない、または想定動作が見えない場合は [Troubleshooting](./troubleshooting.md) を参照してください。

さらに多くのサンプル policy と想定 score 結果については [Scoring Test & Evaluation Guide](./scoring_tests.md) を参照してください。

---

## 🎉 おめでとうございます

クイックスタートガイドを完了しました。これで次の状態になっています。
- ✅ Gazebo と RViz を含む評価環境が起動している
- ✅ すべての依存関係がインストールされたローカル Pixi workspace がある
- ✅ サンプル policy を実行し、AIC engine がどのように trial を管理するかを確認できた

**次のステップ:** ソリューションを提出する段階になったら、participant workspace を container 化する必要があります。policy のパッケージ化と提出の詳細は [Submission Guidelines](./submission.md) を参照してください。

---

## 次のステップ

環境の準備ができたら、同じ evaluation container を異なる [baseline solutions](./aic_example_policies/README.md) で実行してみてください。
その後、[Toolkit Guide](./README.md#toolkit-guide) の **💻 Develop Your Policy** セクションへ進んでください。
