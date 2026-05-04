# lerobot_robot_aic

この package には、AIC ロボット向けの [LeRobot](https://huggingface.co/lerobot) interface が含まれています。

## Usage

ここでは LeRobot でできることの一部を説明します。さらに詳しくは公式の [LeRobot docs](https://huggingface.co/docs/lerobot/en/index) を参照してください。

LeRobot driver は [pixi](https://prefix.dev/tools/pixi) workspace 内にインストールされます。基本的には command の先頭に `pixi run` を付けるか、`pixi shell` で環境に入って実行します。

<a id="teleoperating-with-lerobot"></a>
### LeRobot で teleoperation する

```bash
cd ~/ws_aic/src/aic
pixi run lerobot-teleoperate \
  --robot.type=aic_controller --robot.id=aic \
  --teleop.type=<teleop-type> --teleop.id=aic \
  --robot.teleop_target_mode=<mode> --robot.teleop_frame_id=<frame_id> \
  --display_data=true
```

`--teleop.type` の選択肢（および対応する `--robot.teleop_target_mode` の設定）は以下です。

- `aic_keyboard_ee` は cartesian-space のキーボード制御（`--robot.teleop_target_mode=cartesian` に設定）
- `aic_spacemouse` は cartesian-space の SpaceMouse 制御（`--robot.teleop_target_mode=cartesian` に設定）
- `aic_keyboard_joint` は joint-space 制御（`--robot.teleop_target_mode=joint` に設定）

`--robot.teleop_target_mode` が `cartesian` のときの `--robot.teleop_frame_id` の選択肢:
- `base_link` はロボット base link 基準で cartesian target を送る
- `gripper/tcp` はロボット gripper に付いた `tcp` frame 基準で cartesian target を送る

例:
```bash
cd ~/ws_aic/src/aic
pixi run lerobot-teleoperate \
  --robot.type=aic_controller --robot.id=aic \
  --teleop.type=aic_keyboard_ee --teleop.id=aic \
  --robot.teleop_target_mode=cartesian --robot.teleop_frame_id=base_link \
  --display_data=true
```

:warning: Note: `--teleop.type` を設定するだけでなく、`--robot.teleop_target_mode` も必ず設定してください。`AICRobotAICController` class は controller にどの種類の action を送るべきかを知る必要がありますが、`--teleop.type` にはアクセスできないためです。

#### Cartesian space control

Cartesian 制御では、`--teleop.type` と `--robot.teleop_target_mode` に加えて、`teleop_frame_id`（Cartesian 制御の参照 frame）も設定できます。これは gripper TCP（`"gripper/tcp"`、既定値）または robot base link（`"base_link"`）のどちらかに設定します。

##### Keyboard

> Shift+&lt;key&gt; command の使い方に関する Note: 停止するには、Shift を離す *前に* &lt;key&gt; を離してください。そうしないと、Shift と &lt;key&gt; の両方を離した後でもロボットが回転し続けることがあります。

| Key     | Cartesian      |
| ------- | ---------- |
| w       | -linear y  |
| s       | +linear y  |
| a       | -linear x  |
| d       | +linear x  |
| r       | -linear z  |
| f       | +linear z  |
| q       | -angular z |
| e       | +angular z |
| shift+w | +angular x |
| shift+s | -angular x |
| shift+a | -angular y |
| shift+d | +angular y |

`t` を押すと slow mode と fast mode を切り替えます。

key mapping と speed setting は `aic_teleop.py` 内の `AICKeyboardJointTeleop` と `AICKeyboardJointTeleopConfig` で確認・編集できます。

##### SpaceMouse

:warning: Note: 私たちの経験では、SpaceMouse による teleoperation は keyboard より遅延が大きめでした。

私たちは 3Dconnexion SpaceMouse と [pyspacemouse](https://github.com/JakubAndrysek/PySpaceMouse?tab=readme-ov-file#dependencies) library を使用しました。USB 権限を有効にするには、`/etc/udev/rules.d/99-spacemouse.rules` に次を追加する必要があるかもしれません。
``` bash
# 3Dconnexion device の全 hidraw node に適用
KERNEL=="hidraw*", ATTRS{idVendor}=="046d", MODE="0666", GROUP="plugdev"
# USB device 本体に適用
SUBSYSTEM=="usb", ATTRS{idVendor}=="046d", MODE="0666", GROUP="plugdev"
```
その後、次を実行します。
``` bash
sudo udevadm control --reload-rules
sudo udevadm trigger
```
axis mapping と speed setting は `aic_teleop.py` 内の `AICSpaceMouseTeleop` と `AICSpaceMouseTeleopConfig` で確認・編集できます。

#### Joint space control

| Key | Joint          |
| --- | -------------- |
| q   | -shoulder_pan  |
| a   | +shoulder_pan  |
| w   | -shoulder_lift |
| s   | +shoulder_lift |
| e   | -elbow         |
| d   | +elbow         |
| r   | -wrist_1       |
| f   | +wrist_1       |
| t   | -wrist_2       |
| g   | +wrist_2       |
| y   | -wrist_3       |
| h   | +wrist_3       |

`u` を押すと slow mode と fast mode を切り替えます。

key mapping と speed setting は `aic_teleop.py` 内の `AICKeyboardEETeleop` と `AICKeyboardEETeleopConfig` で確認・編集できます。

<a id="recording-training-data"></a>
### 学習データを記録する

```bash
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

:warning: Note（`lerobot-teleoperate` と同じ）: `--teleop.type` だけでなく `--robot.teleop_target_mode` も必ず設定してください。`AICRobotAICController` class は controller にどの種類の action を送るかを知る必要がありますが、`--teleop.type` にはアクセスできないためです。

command の開始時に `WARN   Watchdog Validator ThreadId(13) zenoh_shm::watchdog::periodic_task: Some("Watchdog Validator")` が表示されることがありますが、これは無視して構いません。`INFO ... ls/utils.py:227 Recording episode 0` が出ていることを確認してください。

LeRobot recording key:

| Key         | Command          |
| ----------- | ---------------- |
| Right Arrow | 次の episode     |
| Left Arrow  | 現在の episode をキャンセルして再記録 |
| ESC         | 記録停止   |

<!-- TODO: lerobot-record doesn't load the hil processor to handle teleop events (lerobot bug?) -->

### 学習

LeRobot dataset が用意できたら、学習については [LeRobot tutorials](https://huggingface.co/docs/lerobot/en/index) を参照してください。

```bash
cd ~/ws_aic/src/aic
pixi run lerobot-train \
  --dataset.repo_id=${HF_USER}/your_dataset \
  --policy.type=your_policy_type \
  --output_dir=outputs/train/act_your_dataset \
  --job_name=act_your_dataset \
  --policy.device=cuda \
  --wandb.enable=true \
  --policy.repo_id=${HF_USER}/act_policy
```
