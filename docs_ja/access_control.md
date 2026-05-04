# アクセス制御

評価システムでは、Zenoh のアクセス制御リスト（ACL）を使って、提出物がシミュレーション内の部品 pose を参照するだけで不正に解けてしまうような ROS topic や service を利用できないようにしています。

提出環境で提供されるセキュリティを再現するには、Zenoh security を有効にした状態で環境とモデル提出物を実行できます。これにより、不正防止のため `gz_server` namespace 内のものなど、一部の topic や service へのアクセスが Zenoh によって遮断されます。

## 別ターミナルでテストする

以下のデモでは、Zenoh のアクセス制御が実際に機能していることを示します。これらの手順は提出ポータルでは Docker により自動実行されますが、ここでは対話性と分かりやすさのため、コマンドライン上で手動実行する形で示しています。ローカルテスト時に、未許可の topic や service を使っていないことを確認するのにも役立ちます。

### Terminal 1: ACL 付きで Zenoh router を起動

```bash
. install/setup.bash
. src/aic/docker/aic_eval/zenoh_config_router.sh
ros2 run rmw_zenoh_cpp rmw_zenohd
```

### Terminal 2: シミュレーション環境を起動

以下のコマンドで、いくつかの entity を生成した状態のシミュレーション環境を起動します。
```bash
. install/setup.bash
. src/aic/docker/aic_eval/zenoh_config_eval_session.sh
export RMW_IMPLEMENTATION=rmw_zenoh_cpp
ros2 launch aic_bringup aic_gz_bringup.launch.py nic_card_mount_0_present:=true sc_port_0_present:=true ground_truth:=false spawn_task_board:=true spawn_cable:=true attach_cable_to_gripper:=true sfp_mount_rail_0_present:=true cable_type:=sfp_sc_cable
```

### Terminal 3: service が遮断されることを確認

```bash
. install/setup.bash
. src/aic/docker/aic_model/zenoh_config_model_session.sh
export RMW_IMPLEMENTATION=rmw_zenoh_cpp
ros2 service call /gz_server/get_entities_states simulation_interfaces/srv/GetEntitiesStates
```

この呼び出しは ACL によって遮断されるため、成功しません。

一方、このターミナルで `eval` identity に対応する environment variable を使うと、シミュレーション内の全 entity の pose と velocity 一覧が返ってきます。
```bash
. install/setup.bash
. src/aic/docker/aic_eval/zenoh_config_eval_session.sh
export RMW_IMPLEMENTATION=rmw_zenoh_cpp
ros2 service call /gz_server/get_entities_states simulation_interfaces/srv/GetEntitiesStates
```

`eval` identity はパスワードで保護されており、そのパスワードは提出ポータルで実行されるときには異なる値になります :smile:

## docker-compose でテストする

Docker-compose を使うと、評価コンテナとモデルコンテナの相互作用を手軽にテストできます。

まず、コンテナを build します。
```bash
docker compose -f docker/docker-compose.yaml build
```

次に、それらを起動します。
```bash
docker copose -f docker/docker-compose.yaml up
```
