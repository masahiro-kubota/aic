# AIC Task Board Description

**AI for Industry Challenge (AIC)** の task board は、特に server や data center infrastructure に見られる high-mix electronics manufacturing の現実的なケーブル管理課題を再現するために設計された、モジュール式で再構成可能なプラットフォームです。この環境は、高密度な光ファイバ配線と、密集した network hardware への transceiver 挿入という複雑なタスクを再現します。

この task board は、challenge の各 phase を通じて、器用な操作、知覚、motion planning を評価するための主要環境として機能します。


## 1. Board Overview

task board は、**[SFP (Small Form-factor Pluggable) module、LC (Lucent Connector) fiber optic、SC (Subscriber Connector) fiber optic](https://www.versitron.com/pages/sfp-sc-and-lc-connectors-transceivers-defined-and-analyzed-in-detail)** コネクタを操作するための標準化された物理 interface を提供します。task board は 4 つの distinct zone に分かれており、assembly target と初期 component pick location を分離しています。

![AIC Task Board](../../media/aic_task_board.png)

## 2. Zone Descriptions

AIC task board は、電子機器組立ワークフロー全体を模擬する 4 つの機能 zone で構成されています。Zone 1 と 2 は assembly target です。
* [Zone 1](#zone-1-network-interface-cards-nic) には SFP port を持つ Network Interface Card（NIC）が配置され、server compute tray を表現します
* [Zone 2](#zone-2-sc-optical-ports) は SC port を備えた optical patch panel を模擬します
* [Zone 3](#zone-3--4-pick-locations) と 4 は Pick Location として機能し、SFP module や fiber optic plug を adjustable mount 上に配置する high-mix supply area を提供します

この modular layout により、ロボットは Zone 3 および 4 での整然とした pick から、Zone 1 および 2 での高精度かつ器用な挿入へ遷移する必要があります。

<a id="zone-1-network-interface-cards-nic"></a>
### Zone 1: Network Interface Cards (NIC)
この zone は、data link が確立される networking switch または server compute tray を表します。

![AIC Task Board](../../media/aic_board_zone_1.png)

* **Components:** dual-port network card（NIC）を最大 5 枚まで搭載可能
* **Ports:** 各 card は 2 つの SFP port を備える
* **Flexibility:** card は mount rail 上をスライドできるよう設計されており、位置と向きのランダム offset を許容する
  * Card translation limits: [-0.0215, 0.0234] meters
  * Card orientation limits: [-10, +10] degrees

![AIC Task Board](../../media/aic_board_zone_1_legend.png)

<a id="zone-2-sc-optical-ports"></a>
### Zone 2: SC Optical Ports
この zone は、server rack の optical patch panel または backplane を模擬します。

![AIC Task Board](../../media/aic_board_zone_2.png)

* **Ports:** 2 本の rail に分散して、最大 5 つの SC port を搭載可能
* **Flexibility:** port は rail 上をスライドできるため、位置 offset をランダム化できる
  * SC port translation limits: [-0.06, 0.055] meters

![AIC Task Board](../../media/aic_board_zone_2_legend.png)

<a id="zone-3--4-pick-locations"></a>
### Zone 3 & 4: Pick Locations
Zone 3 と 4 は、component（LC plug、SC plug、SFP module）を配線・挿入する前に並べておく整理された supply area です。

![AIC Task Board](../../media/aic_board_zone_3.png)
![AIC Task Board](../../media/aic_board_zone_4.png)

* **Mounts:** LC/SC plug と SFP module 用 fixture を保持する
* **Customization:** fixture は任意の rail に任意の順序で配置でき、高 mix な環境を作れる
  * Fixture translation limits: [-0.09425, 0.09425] meters
  * Fixture orientation limits: [-60, +60] degrees


![AIC Task Board](../../media/aic_board_zone_3_legend_1.png)
![AIC Task Board](../../media/aic_board_zone_3_legend_2.png)

![AIC Task Board](../../media/aic_board_zone_4_lengend_1.png)
![AIC Task Board](../../media/aic_board_zone_4_legend_2.png)

## 3. Bill of Material (BOM)

task board は、入手しやすい部品と 3D print によって比較的簡単に構築できるよう設計されています。完全な task board を作るには次が必要です。

* **Off-the-shelf components:**
  * NIC Card（Quantity: 5）- [Amazon link](https://a.co/d/5lkWCj4)
  * SFP module（Quantity: 5）- [Amazon link](https://a.co/d/7RGkdZO)
  * LC to SC cable（Quantity: 5）- [Amazon link](https://a.co/d/edbwgg2)
  * SC-SC connectors（Quantity: 1 pack）- [Amazon link](https://a.co/d/4PgnstS)
* **Task board BOM:** *3D printed chassis, rails, and component mounts.* **TODO**

## 4. Configuration Structure

各 trial における board の状態は YAML configuration で定義されます。これにより、上記 rail limit の範囲で component の位置と向きをランダム化できるため、ロボットはハードコード座標ではなく知覚に依存する必要があります。AIC engine の [sample config](../aic_engine/config/sample_config.yaml) を参照してください。
