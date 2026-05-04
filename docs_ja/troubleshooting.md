# Troubleshooting

## Gazebo の real-time factor が低い

シミュレーションは **1.0 RTF（100% real-time factor）** で動作するよう設定されています。つまり simulation time は wall-clock time と一致するはずです。RTF がこれより低い場合は、以下のセクションが原因調査と解決に役立つかもしれません。

### Gazebo が dedicated GPU を使っていない

マシンに GPU が 2 つある場合（または CPU に内蔵 GPU がある場合）、OpenGL が *integrated* GPU を使って rendering していることがあり、その場合 RTF が非常に低くなります。これを直すには、手動で *discrete* GPU を使うよう設定する必要があるかもしれません。

OpenGL が discrete GPU を使っているか確認するには `glxinfo -B` を実行します。出力に discrete GPU の詳細が表示されるはずです。さらに `nvidia-smi` を実行すれば GPU ごとの process も確認できます。AIC sim が動作中なら、process list に `gz sim` が表示されるはずです。

誤った GPU が選ばれている場合は、`sudo prime-select nvidia` を実行してください。
**Note**: 変更を有効にするには、一度 log out して再度 log in する必要があります。その後 `glxinfo -B` を再実行し、discrete GPU が有効になっていることを確認してください。

[Problems with dual Intel and Nvidia GPU systems](https://gazebosim.org/docs/latest/troubleshooting/#problems-with-dual-intel-and-nvidia-gpu-systems) も参考になります。

<a id="no-gpu-available"></a>
### GPU が利用できない場合

システムに dedicated GPU がない場合、real-time factor（RTF）の性能が低くなることがあります。これは、AIC scene が GPU acceleration を前提とする [GlobalIllumination (GI)](https://gazebosim.org/api/sim/9/global_illumination.html) ベース rendering を使っているためです。

**GPU のないシステムで simulation 性能を改善する方法:**

[`aic.sdf`](../aic_description/world/aic.sdf) を編集し、global illumination 設定内の `<enabled>` を `false` にすることで GlobalIllumination を無効化できます。対象箇所は [こちら](https://github.com/intrinsic-dev/aic/blob/c8aa4571d9dc4bd55bbefc02b0a160ba0e8e1e90/aic_description/world/aic.sdf#L39) と [こちら](https://github.com/intrinsic-dev/aic/blob/c8aa4571d9dc4bd55bbefc02b0a160ba0e8e1e90/aic_description/world/aic.sdf#L109) です。画質は落ちますが、CPU のみのシステムでは RTF が大きく改善する場合があります。

> [!WARNING]
> GI を無効にすると scene の見た目が変わるため、vision-based policy に影響する可能性があります。

## Zenoh Shared Memory Watchdog Warning

system 実行中、次のような warning が出ることがあります。

```
WARN Watchdog Validator ThreadId(17) zenoh_shm::watchdog::periodic_task:
error setting scheduling priority for thread: OS(1), will run with priority 48.
This is not an hard error and it can be safely ignored under normal operating conditions.
```

**この warning は無害であり、通常は無視して構いません。** Zenoh の shared memory watchdog thread が、より高い scheduling priority を設定できなかったことを示しています（これには高い権限が必要です）。system 自体は正常に動作し続けます。

**なぜ起きるのか:**
- watchdog thread は shared memory の健全性を監視している
- 高い priority を設定するには `CAP_SYS_NICE` capability または root 権限が必要
- それがない場合、thread は既定 priority（48）で動作する

**問題になる可能性がある場面:**
- 極端に高い CPU 負荷下では、watchdog がたまに deadline を逃す可能性がある
- その結果、shared memory operation でまれに timeout が起こるかもしれない
- ただし実際には、一般的な workload ではほぼ問題にならない

**shared memory が動いていることを確認するには:**
```bash
# Zenoh の shared memory file を確認
ls -lh /dev/shm | grep zenoh

# network traffic を監視（最小限であるべき）
sudo tcpdump -i lo port 7447 -v
```

`/dev/shm` に Zenoh file があり、port 7447 の traffic が最小限なら、この warning が出ていても shared memory は正常に機能しています。

<a id="nvidia-rtx-50xx-cards-not-supported-on-pytorch-version-locked-in-pixi"></a>
## Pixi に固定された PyTorch version が NVIDIA RTX 50xx card を未サポート

```
UserWarning:
NVIDIA GeForce RTX 5090 with CUDA capability sm_120 is not compatible with the current PyTorch installation.

The current PyTorch install supports CUDA capabilities sm_50 sm_60 sm_70 sm_75 sm_80 sm_86 sm_90.
If you want to use the NVIDIA GeForce RTX 5090 GPU with PyTorch, please check the instructions at https://pytorch.org/get-started/locally/
```

`pixi.toml` にある `lerobot` version は、より古い `pytorch` version（古い cuda 向け build）に依存しています。
そのため `pixi install` では、新しい NVIDIA RTX 50xx card の sm_120 architecture をサポートしない古い version が入ります。

私たちは、`pixi.toml` に次を追加することで Nvidia RTX 5090 上でこの policy を動かせました。
```
[pypi-options.dependency-overrides]
torch = ">=2.7.1"
torchvision = ">=0.22.1"
```

詳細はこの [LeRobot issue](https://github.com/huggingface/lerobot/issues/2217) を参照してください。

## Error: no such container aic_eval

`distrobox enter -r aic_eval` 実行時に、次の error が出ることがあります。
```bash
Error: no such container aic_eval
```

distrobox は既定で podman を使いますが、このセットアップでは docker を使っています。既定 container manager を `DBX_CONTAINER_MANAGER` environment variable で設定していることを確認してください。
```bash
export DBX_CONTAINER_MANAGER=docker
```
