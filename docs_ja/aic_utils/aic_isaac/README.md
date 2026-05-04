# AIC Isaac Lab Integration

この package は、AI for Industry Challenge（AIC）環境を Isaac Lab でセットアップするためのドキュメント、script、utility を提供します。


## Overview

[Isaac Lab](https://isaac-sim.github.io/IsaacLab/main/index.html) は、ロボット学習のための統一的かつモジュール型の framework で、ロボティクス研究における一般的な workflow（reinforcement learning、demonstration からの学習、motion planning など）を簡素化することを目的としています。  
**NVIDIA** との協力により、この integration では参加者は次を行えます。

- AIC 環境で teleoperation を行う
- Imitation Learning 用に episode を記録・再生する
- `rsl-rl` library を使って policy 学習用の Reinforcement Learning を行う


## Workflow

> [!TIP]
> **Isaac Lab** 自体に起因する問題（例: framework の挙動、Docker セットアップ、Isaac Lab API）と思われるものに遭遇した場合は、[Isaac Lab GitHub repository](https://github.com/isaac-sim/IsaacLab) に issue を作成してください。そちらの maintainer が最も適切に対応できます。AIC integration や challenge asset 固有の問題については、この repo の issue tracker を使ってください。

**推奨:** NVIDIA team が準備した asset を使用してください。指示どおりに download / 配置した後、container を起動して task を実行します。

| Step | やること | Section |
|------|-------------|---------|
| 1 | Docker と NVIDIA Container Toolkit をインストールする | [Prerequisites](#prerequisites) |
| 2 | Isaac Lab を clone / build し、その中に AIC repo を clone する | [Installation & Setup](#installation--setup) |
| 3 | NVIDIA が準備した asset を download して `Intrinsic_assets` に配置する | [Assets](#assets) |
| 4 | Isaac Lab container を起動し、その中に入る | [Assets](#assets) |
| 5 | container 内から teleoperation または reinforcement learning を実行する | [Usage](#usage) |


<a id="prerequisites"></a>
## Prerequisites

### Docker

1. 利用中の platform 向け [Docker Engine](https://docs.docker.com/engine/install/) をインストールします。
2. [Linux post-installation steps for Docker Engine](https://docs.docker.com/engine/install/linux-postinstall/) を完了し、非 root ユーザーで Docker を管理できるようにします。

### NVIDIA Container Toolkit（任意）

1. Docker Engine から NVIDIA GPU にアクセスできるよう、[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) をインストールします。

2. インストール後、Docker が NVIDIA runtime を使うよう設定します。
    ```bash
    sudo nvidia-ctk runtime configure --runtime=docker
    sudo systemctl restart docker
    ```


<a id="installation--setup"></a>
## Setup

> [!NOTE]
> このセクションの command はすべて **host machine 上**（Docker の中ではなく）で実行します。

> [!WARNING]
> この integration は Isaac Lab version *2.3.2* で検証されています。

home directory に Isaac Lab repository を clone します。
```bash
cd ~
git clone git@github.com:isaac-sim/IsaacLab.git
```

`IsaacLab` directory の中に AIC repository を clone します。
```bash
cd ~/IsaacLab
git clone git@github.com:intrinsic-dev/aic.git
```

<a id="assets"></a>
## Assets

challenge に必要な asset は **NVIDIA team が準備済み** です。[用意された asset pack を download](https://developer.nvidia.com/downloads/Omniverse/learning/Events/Hackathons/Intrinsic_assets.zip) し、展開したうえで、以下のように配置してください。

展開した `Intrinsic_assets` directory は `aic_task` 内の次の場所へ配置します。

```bash
~/IsaacLab/aic/aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/
```

**Intrinsic_assets directory の中身**（download pack 内）:
```
Intrinsic_assets/
├── aic_unified_robot_cable_sdf.usd
├── assets
│   ├── NIC Card
│   │   ├── nic_card.usd
│   │   ├── nic_card_visual.usd
│   │   └── textures
│   │       ├── Image_0.jpg
│   │       ├── Image_1.jpg
│   │       ├── Image_2.jpg
│   │       └── NIC_Albedo.jpg
│   ├── NIC Card Mount
│   │   ├── nic_card_mount_visual.usd
│   │   ├── nic_card_visual.usd
│   │   └── textures
│   │       ├── Image_0.jpg
│   │       ├── Image_1.jpg
│   │       ├── Image_2.jpg
│   │       └── NIC_Albedo.jpg
│   ├── SC Plug
│   │   ├── sc_plug_visual.usd
│   │   └── textures
│   │       ├── Image_1.png
│   │       └── sc_plug_visual_image1.png
│   ├── SC Port
│   │   ├── sc_port.usd
│   │   ├── sc_port_visual.usd
│   │   └── textures
│   │       ├── Image_0.png
│   │       └── Image_1.png
│   └── Task Board Base
│       ├── base_visual.usd
│       └── task_board_rigid.usd
├── scene
│   └── aic.usd
└── scene.usd
```

asset pack に world、enclosure、robot の USD や個別の配置手順が含まれている場合は、それに従ってください。そうでなければ、この prepared pack だけで完結します。

asset の調整や import に関する Isaac Sim Documentation:
- [Tutorial: Import URDF](https://docs.isaacsim.omniverse.nvidia.com/4.5.0/robot_setup/import_urdf.html)
- [Tuning Joint Drive Gains](https://docs.isaacsim.omniverse.nvidia.com/4.5.0/robot_setup/joint_tuning.html)
- [Gain Tuner Extension](https://docs.isaacsim.omniverse.nvidia.com/4.5.0/robot_setup/ext_isaacsim_robot_setup_gain_tuner.html)
- [Physics Inspector](https://docs.isaacsim.omniverse.nvidia.com/4.5.0/physics/joint_inspector.html)
- [Simulation Data Visualizer](https://docs.isaacsim.omniverse.nvidia.com/4.5.0/physics/ext_isaacsim_inspect_physics.html)


## Installation

`base` profile を build します（これにより `isaac-lab-base` Docker image が作成されます）。
```bash
cd ~/IsaacLab
./docker/container.py build base
```

container を起動し、shell を attach します（Isaac Lab repo から実行）。
```bash
cd ~/IsaacLab
./docker/container.py start base
./docker/container.py enter base
```

Isaac Lab container 内で `aic_task` を editable mode で install します。
```
 python -m pip install -e aic/aic_utils/aic_isaac/aic_isaaclab/source/aic_task
 ```


<a id="usage"></a>
## Usage

> [!NOTE]
> 以下の command は、Isaac Lab container を起動して中に入った後、**container 内で** 実行します。

### Environment と Sensor Reading
利用可能な environment を一覧表示します（`AIC-Task-v0` RL Environment は参考用として提供されています）。
```bash
isaaclab -p aic/aic_utils/aic_isaac/aic_isaaclab/scripts/list_envs.py
```

### Teleoperation と Imitation Learning
keyboard でロボットを teleoperate します。
```bash
isaaclab -p aic/aic_utils/aic_isaac/aic_isaaclab/scripts/teleop.py \
    --task AIC-Task-v0 --num_envs 1 --teleop_device keyboard --enable_cameras
```

> [!NOTE]
> keyboard teleop の感度は `aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/aic_task_env_config.py` を更新することで調整できます。
> ```python
> "keyboard": Se3KeyboardCfg(
>     pos_sensitivity=0.08,
>     rot_sensitivity=0.05,
>     gripper_term=False,
>     sim_dev=self.sim.device,
> ),
> ```

データ収集:
```bash
isaaclab -p aic/aic_utils/aic_isaac/aic_isaaclab/scripts/record_demos.py \
    --task AIC-Task-v0 --teleop_device keyboard --enable_cameras \
    --dataset_file ./datasets/dataset.hdf5 --num_demos 10
```

収集した episode を再生するには:
```bash
isaaclab -p aic/aic_utils/aic_isaac/aic_isaaclab/scripts/replay_demos.py \
    --dataset_file ./datasets/dataset.hdf5
```


> [!NOTE]
> teleoperation したデータを記録するには、外部 environment と Isaac Lab を接続する必要があります。

追加リソース:
1. [Teleoperation using Keyboard, Spacemouse and XR](https://isaac-sim.github.io/IsaacLab/main/source/overview/imitation-learning/teleop_imitation.html#teleoperation)
2. [Recording Teleoperation data](https://isaac-sim.github.io/IsaacLab/main/source/overview/imitation-learning/teleop_imitation.html#collecting-demonstrations)
3. [Imitation Learning in Isaac Lab](https://isaac-sim.github.io/IsaacLab/main/source/overview/imitation-learning/teleop_imitation.html#imitation-learning-with-isaac-lab-mimic)

### Reinforcement Learning
次の command で training script を実行します。
```bash
isaaclab -p aic/aic_utils/aic_isaac/aic_isaaclab/scripts/rsl_rl/train.py \
    --task AIC-Task-v0 --num_envs 1 --enable_cameras
```

その他のリソース:
1. [Gear Assembly Task](https://isaac-sim.github.io/IsaacLab/main/source/policy_deployment/02_gear_assembly/gear_assembly_policy.html)
2. [Creating a manager-based RL environment](https://isaac-sim.github.io/IsaacLab/main/source/tutorials/03_envs/create_manager_rl_env.html)
3. [Task Curation, VLA training and Policy Evaluation using Isaac Lab Arena](https://isaac-sim.github.io/IsaacLab-Arena/release/0.1.1/index.html)



### `aic_isaaclab` の Directory Structure

```bash
aic_isaac/
├── README.md
└── aic_isaaclab
    ├── pyproject.toml
    ├── scripts
    │   ├── list_envs.py
    │   ├── random_agent.py
    │   ├── record_demos.py
    │   ├── replay_demos.py
    │   ├── rsl_rl
    │   │   ├── cli_args.py
    │   │   ├── play.py
    │   │   └── train.py
    │   ├── teleop.py
    │   └── zero_agent.py
    └── source
        └── aic_task
            ├── aic_task
            │   ├── __init__.py
            │   ├── extension.py
            │   └── tasks
            │       ├── __init__.py
            │       └── manager_based
            │           ├── __init__.py
            │           └── aic_task
            │               ├── __init__.py
            │               ├── agents
            │               │   ├── __init__.py
            │               │   └── rsl_rl_ppo_cfg.py
            │               ├── aic_task_env_cfg.py
            │               └── mdp
            │                   ├── __init__.py
            │                   ├── events.py
            │                   ├── observations.py
            │                   └── rewards.py
            ├── config
            │   └── extension.toml
            ├── docs
            │   └── CHANGELOG.rst
            ├── pyproject.toml
            └── setup.py
```



## Future Work

workflow に対する今後の改善予定:
- [ ] SDF World から USD asset への export pipeline を追加


## Resources

- [Isaac Lab Documentation](https://isaac-sim.github.io/IsaacLab/main/index.html)
- [AIC Getting Started Guide](../../getting_started.md)
- [AIC Scene Description](../../scene_description.md)
