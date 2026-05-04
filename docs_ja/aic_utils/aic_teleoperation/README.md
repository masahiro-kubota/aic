# aic_teleoperation

joint-space と Cartesian-space の両方の control mode でロボットを操作するための、キーボードベースの teleoperation package です。

## Prerequisites

1. 開発環境のセットアップのために [Getting Started Guide](../../getting_started.md) を実施する
2. X11 display server（`pynput` は Wayland で既知の問題あり）
3. native build の場合: `sudo apt install python3-pynput`（pixi では自動インストールされる）

## 利用可能な Script

### 1. Joint Space Teleoperation (`joint_keyboard_teleop`)

各ロボット関節を直接制御します。

**Key Mappings:**
- `q/a` - Joint 1 (`shoulder_pan_joint`): +/-
- `w/s` - Joint 2 (`shoulder_lift_joint`): +/-
- `e/d` - Joint 3 (`elbow_joint`): +/-
- `r/f` - Joint 4 (`wrist_1_joint`): +/-
- `t/g` - Joint 5 (`wrist_2_joint`): +/-
- `y/h` - Joint 6 (`wrist_3_joint`): +/-

**Speed Control:**
- `k` - Slow mode（0.075 rad/s）
- `l` - Fast mode（0.2 rad/s）

**Exit:**
- `ESC` - teleoperation を終了

### 2. Cartesian Space Teleoperation (`cartesian_keyboard_teleop`)

end-effector の pose（位置と向き）を制御します。

**Linear Movement:**
- `a/d` - X axis: -/+
- `w/s` - Y axis: -/+
- `r/f` - Z axis: -/+

**Angular Movement:**
- `Shift + s/w` : -/+ Angular X
- `Shift + a/d` : -/+ Angular Y
- `q/e` : -/+ Angular Z

**Speed Control:**
- `k` - Slow mode（linear: 0.02 m/s、angular: 0.02 rad/s）
- `l` - Fast mode（linear: 0.1 m/s、angular: 0.1 rad/s）

**Frame Toggle:**
- `n` - Tool frame（`gripper/tcp`）
- `m` - Global frame（`base_link`）

**Exit:**
- `ESC` - teleoperation を終了

## Usage

まず、[Getting Started - Quick Start](../../getting_started.md#quick-start) の手順に従って評価環境を起動してください。

### pixi を使う場合（推奨）

```bash
cd ~/ws_aic/src/aic

# teleoperation を実行
pixi run ros2 run aic_teleoperation joint_keyboard_teleop
# または
pixi run ros2 run aic_teleoperation cartesian_keyboard_teleop
```

### native ROS 2 build を使う場合

workspace の build と Zenoh のセットアップについては [Building the Evaluation Component from Source](../../build_eval.md) を参照してください。

```bash
# teleoperation を実行
ros2 run aic_teleoperation joint_keyboard_teleop
# または
ros2 run aic_teleoperation cartesian_keyboard_teleop
```

## Notes

- script は起動時に controller を適切な control mode（joint または Cartesian）へ自動切替する
- ESC で teleoperation を安全に終了できる
- キーボード入力は、ターミナル window にフォーカスがなくても取得される
