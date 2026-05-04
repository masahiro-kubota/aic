# カスタム Dockerfile の作成（上級者向け）

サンプルの dockerfile は、`aic_model` を使って policy を実行する前提になっています。`aic_model` を使わない場合は、独自の dockerfile を作成する必要があります。

この文書は、ROS 2 middleware の概念、Zenoh、Docker について高度な知識があることを前提としています。

## 1. ROS 2 と rmw_zenoh_cpp

policy node は本質的には、observation を subscribe し、実行すべき action を publish する ROS 2 node です。
利便性のため、サンプル policy では `aic_model` という Python ROS node を使い、その中で `Policy` class を生成して実際の policy を実装することで、できるだけ多くの boilerplate を切り離しています。

ROS 2 自体は middleware 非依存ですが、AI for Industry Challenge では `rmw_zenoh_cpp` のみを使用します。あなたの dockerfile は **必ず** `rmw_zenoh_cpp` を使って policy node を実行しなければなりません。通常は `RMW_IMPLEMENTATION` environment variable を `rmw_zenoh_cpp` に設定して対応します。

この設定は image 実行時に自動で与えられます。したがって dockerfile 側でそれを上書きしないこと、また `rmw_zenoh_cpp` を利用可能な状態にしておくことが必要です。

## 2. Zenoh

評価時には、あなたの image は次の environment variable 付きで実行されます。

| Variable                    | Description                                                                                                     |
| :-------------------------- | :-------------------------------------------------------------------------------------------------------------- |
| RMW_IMPLEMENTATION          | 使用する ROS 2 middleware。常に `rmw_zenoh_cpp` が設定される                                                   |
| ZENOH_ROUTER_CHECK_ATTEMPTS | 常に `-1` が設定される。model router の準備前にコンテナが起動しても error にならないようにする                |
| AIC_MODEL_ROUTER_ADDR       | policy node が接続すべき Zenoh router address                                                                  |
| AIC_MODEL_PASSWD            | policy node の認証に使用すべき password                                                                        |

policy node は **必ず** 次を満たす必要があります。

1. `AIC_MODEL_ROUTER_ADDR` environment variable で与えられた Zenoh router に接続すること
2. ユーザー `model` と、`AIC_MODEL_PASSWD` で与えられた password を使った user-password 認証を行うこと

もっとも簡単な方法は、`ZENOH_CONFIG_OVERRIDE` を次のように設定することです。

```bash
ZENOH_CONFIG_OVERRIDE='connect/endpoints=["tcp/'"$AIC_MODEL_ROUTER_ADDR"'"];transport/auth/usrpwd/user="model";transport/auth/usrpwd/password="'"$AIC_MODEL_PASSWD"'";transport/auth/usrpwd/dictionary_file="/credentials.txt"'
```

さらに credentials file を作成する必要があります。たとえば次のようにできます。

```bash
echo "model:$AIC_MODEL_PASSWD" >> /credentials.txt
```

詳細は https://github.com/ros2/rmw_zenoh および https://zenoh.io/docs/manual/access-control/ を参照してください。

## 3. Entrypoint

image の entrypoint は policy node を起動しなければなりません。追加のコマンドライン引数は渡されません。
