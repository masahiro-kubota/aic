# Rocky_Shao Build Thread Dump

- Source thread: [Rocky's Open-Source Build Thread (AI for Industry Challenge)](https://discourse.openrobotics.org/t/rockys-open-source-build-thread-ai-for-industry-challenge/53155)
- Extracted posts: `Rocky_Shao` only
- Post count: `15`
- Timestamps: Discourse ISO 8601 in `UTC`
- Snapshot basis: current visible Discourse raw text with quote/image normalization

## Post 01
- Post #: `1`
- Created: `2026-03-12T01:31:09.557Z`
- Source: [Discourse Post #1](https://discourse.openrobotics.org/t/rockys-open-source-build-thread-ai-for-industry-challenge/53155/1)

Hello there! I’m Rocky Shao, a freshman at The Ohio State University.

I have very little “real-world” robotics experience, and I joined this competition as a way to stretch myself and push my learning in a short timeframe (as the competition has a deadline, of course…).

Nevertheless, I believe sharing my progress is a great way to connect with like-minded people passionate about robotics, AI, and cool, fun technology—and hopefully get help from experts when AI can’t answer my questions well enough!

My current progress: [https://github.com/Rocky0Shao/IntrinsicAIChallenge](https://github.com/Rocky0Shao/IntrinsicAIChallenge)

## Post 02
- Post #: `2`
- Created: `2026-03-12T01:38:47.549Z`
- Source: [Discourse Post #2](https://discourse.openrobotics.org/t/rockys-open-source-build-thread-ai-for-industry-challenge/53155/2)

**My Plan:**

1. Add a "Data Collection" teleop class on top of the [`lerobotTeleop`](https://github.com/intrinsic-dev/aic/blob/main/aic_utils/lerobot_robot_aic/lerobot_robot_aic/aic_teleop.py) code to generate "perfect" datasets using the same logic as the CheatCode example policy.

2. Modify the CheatCode policy logic to trigger a stop when force feedback is detected, rather than using hard-coded Z-offsets.

3. Save the recorded data to Hugging Face and use the LeRobot Google Colab notebook template—along with my $300 in student credits—to train the LeRobot ACT modle.

## Post 03
- Post #: `3`
- Created: `2026-03-12T02:09:42.437Z`
- Source: [Discourse Post #3](https://discourse.openrobotics.org/t/rockys-open-source-build-thread-ai-for-industry-challenge/53155/3)

#  March 11

### Today's Progress
I added a "CheatCode" LeRobot Teleop mode that uses Ground-Truth TF frames to insert cables, so I can record nice training data without needing to teleop with keyboard.

It works for Trial 1 and Trial 2, but not Trial 3. I suspect it's a naming-mismatch between different plug types.

https://github.com/Rocky0Shao/IntrinsicAIChallenge/blob/19a0ac32e01445bcadf8a08b789dcf8ac1a17217/aic_utils/lerobot_robot_aic/lerobot_robot_aic/aic_teleop.py#L340-L591

Here's a video of above code working (Note this is NOT the provided CheatCode policy, but above code :slight_smile: 
https://youtube.com/shorts/aCZ-L5IKyE4


### Small Pixi Hack
Also I realized I  have to manually make this version number larger:
https://github.com/Rocky0Shao/IntrinsicAIChallenge/blob/19a0ac32e01445bcadf8a08b789dcf8ac1a17217/aic_utils/lerobot_robot_aic/package.xml#L5-L5
And then run 
```bash
pixi install
```
to make sure `pixi run ...` behaves correctly.
### My Hardware Setup
* Asus Zypherus G16 2024 (32GB RAM, Intel Ultra 9 CPU)
* I dual Boot Ubuntu 24, with 300GB partition. 
I currently haven't ran into huge compatibility issues (my life would be much harder if I'm using WSL or Arch I can imagine)

### Tomorrow's Plan
My next steps is to learn how to correctly record multiple episodes of data in LeRobot.


Also here's a short video for yall to enjoy:
https://youtube.com/shorts/pViucR-X3vQ

## Post 04
- Post #: `4`
- Created: `2026-03-12T17:51:02.853Z`
- Source: [Discourse Post #4](https://discourse.openrobotics.org/t/rockys-open-source-build-thread-ai-for-industry-challenge/53155/4)

# March 12: Resolved TF Lookup Issue for Trial 3

**The Issue:** The LeRobot CheatCode Teleop (discussed above) wasn't working for the Trial 3 SC-Port (short plug). Even though the port naming appeared correct, the robot simply wouldn't move.
> Rocky_Shao wrote:
> It works for Trial 1 and Trial 2, but not Trial 3. I suspect it’s a naming-mismatch between different plug types.

**The Root Cause:** The issue stemmed from a naming mismatch. My code was looking up TF frames under `cable_1`, but the manual spawner was creating the entity as `cable_0`. Pointing my code to `cable_0` resolved the TF lookup issue and got it working.

### Detailed Breakdown: Why this happens

The entity name changes depending on how the scene is spawned:

* **Automatic Spawning (`aic_engine`):** When using `start_aic_engine:=true`, the engine dynamically spawns cables using the YAML key from `sample_config.yaml` as the entity name. Therefore, Trial 3's `cable_1` key correctly spawns an entity named `cable_1` 

https://github.com/intrinsic-dev/aic/blob/5596c67374ee2b56a815152548f35f46f29f9d78/aic_engine/src/aic_engine.cpp#L1216-L1218
* **Manual Spawning (`spawn_cable.launch.py`):** When manually spawning via `/entrypoint.sh` using `spawn_cable:=true`, the launch file strictly hardcodes the entity name as `cable_0` 

https://github.com/intrinsic-dev/aic/blob/5596c67374ee2b56a815152548f35f46f29f9d78/aic_bringup/launch/spawn_cable.launch.py#L64-L67

Because manual scene spawning completely ignores the `sample_config.yaml` ([see source](https://github.com/intrinsic-dev/aic/blob/5596c67374ee2b56a815152548f35f46f29f9d78/aic_engine/config/sample_config.yaml#L320-L320)), it will always create `cable_0`, even for Trial 3 which expects `cable_1`.

### Summary Takeaway

> **If you want to use ground-truth TF when manually spawning custom scenes, always use `cable_0`!**

## Post 05
- Post #: `6`
- Created: `2026-03-12T21:29:58.298Z`
- Source: [Discourse Post #6](https://discourse.openrobotics.org/t/rockys-open-source-build-thread-ai-for-industry-challenge/53155/6)

Thank you for the kind words, and thanks for providing the Robotics community such an exicing and well-organized challenge!

## Post 06
- Post #: `7`
- Created: `2026-03-13T02:12:37.863Z`
- Source: [Discourse Post #7](https://discourse.openrobotics.org/t/rockys-open-source-build-thread-ai-for-industry-challenge/53155/7)

# March 12: Resolved TF Lookup Issue for Trial 3

**The Issue:** The LeRobot CheatCode Teleop (discussed above) wasn't working for the Trial 3 SC-Port (short plug). Even though the port naming appeared correct, the robot simply wouldn't move.
> Rocky_Shao wrote:
> It works for Trial 1 and Trial 2, but not Trial 3. I suspect it’s a naming-mismatch between different plug types.

**The Root Cause:** The issue stemmed from a naming mismatch. My code was looking up TF frames under `cable_1`, but the manual spawner was creating the entity as `cable_0`. Pointing my code to `cable_0` resolved the TF lookup issue and got it working.

### Detailed Breakdown: Why this happens

The entity name changes depending on how the scene is spawned:

* **Automatic Spawning (`aic_engine`):** When using `start_aic_engine:=true`, the engine dynamically spawns cables using the YAML key from `sample_config.yaml` as the entity name. Therefore, Trial 3's `cable_1` key correctly spawns an entity named `cable_1` 

https://github.com/intrinsic-dev/aic/blob/5596c67374ee2b56a815152548f35f46f29f9d78/aic_engine/src/aic_engine.cpp#L1216-L1218
* **Manual Spawning (`spawn_cable.launch.py`):** When manually spawning via `/entrypoint.sh` using `spawn_cable:=true`, the launch file strictly hardcodes the entity name as `cable_0` 

https://github.com/intrinsic-dev/aic/blob/5596c67374ee2b56a815152548f35f46f29f9d78/aic_bringup/launch/spawn_cable.launch.py#L64-L67

Because manual scene spawning completely ignores the `sample_config.yaml` ([see source](https://github.com/intrinsic-dev/aic/blob/5596c67374ee2b56a815152548f35f46f29f9d78/aic_engine/config/sample_config.yaml#L320-L320)), it will always create `cable_0`, even for Trial 3 which expects `cable_1`.

### Summary Takeaway

> **If you want to use ground-truth TF when manually spawning custom scenes, always use `cable_0`!**

## Post 07
- Post #: `8`
- Created: `2026-03-13T02:31:09.587Z`
- Source: [Discourse Post #8](https://discourse.openrobotics.org/t/rockys-open-source-build-thread-ai-for-industry-challenge/53155/8)

# March 12 – Part 2

Finished homework early. What more fun can I have than working on robotics?

## Problem with Old "CheatCode" LeRobot Teleop

My old **CheatCode LeRobot teleop** tries to insert the plug **straight downward blindly**.  
That means if the plug doesn't align with the port, the robot **keeps pushing downward instead of re-aligning**.

This causes some bad behavior:

https://www.youtube.com/shorts/zupveMsGg2o

## Goal

My goal is to modify the custom **LeRobot "CheatCode" Teleop** to use **force feedback** so that:

- When force > **20N**, the robot **lifts up** instead of continuing to push down.

Currently, this **is not working** 
(although I added code to github).

My speculation is that my **hacked LeRobot CheatCode Teleop** is **not subscribing to the `/observations` ROS topic correctly**.

---

## Checking the Force Data

To visualize the current force readings, I used the following command:
```bash
ros2 topic echo /observations | grep -A 14 wrist_wrench
```
This produced the following output:
```yaml
wrist_wrench:
  header:
    stamp:
      sec: 86
      nanosec: 142000000
    frame_id: ati/tool_link
  wrench:
    force:
      x: -1.213755535327079
      y: -13.563044908263814
      z: 27.869689319898924
    torque:
      x: 1.8214097057249585
      y: -0.6879985208072665
      z: -0.17330135150952428
```
This is a good example of the issue:

* **Force z ≈ 27.8N (>20N)**
* However, the robot **still keeps pushing downward** instead of lifting.

---

## Questions
Here's a couple questions I'll figure out tmr:

### 1. What does LeRobot `record` actually record?

Does the **force feedback** get included in the dataset, or is it only recording:

* images
* joint states
* actions

### 2. `ros2 bag record` not working

Even inside the **pixi shell**, `ros2 bag record` does not work.

I tried installing the bag recording functionality, but it **still fails**.

---

## Gazebo vs RViz Spawn Bug

I also noticed a small bug:

* **Taskboard / Cable appears in RViz**
* But **does not spawn in Gazebo**
![image](https://us1.discourse-cdn.com/flex022/uploads/ros/optimized/3X/4/a/4af8b945507d4f5f32892fdcc36bcfd905d9fe6e_2_690x421.jpeg)
Re-Launching `/entrypoint.sh` seems to solve this issue.


A lot to do still.
Learned a lot as well.

## Post 08
- Post #: `9`
- Created: `2026-03-14T02:30:33.742Z`
- Source: [Discourse Post #9](https://discourse.openrobotics.org/t/rockys-open-source-build-thread-ai-for-industry-challenge/53155/9)

# March 13

As mentioned yesterday, my `CheatCode LeRobot Teleop` was not subscribing to `/obesrvations` ros topic —it turns out I just forgot to prefix the command with `pixi run`.

Running the following confirmed that my `cheatcode_teleop` is now successfully subscribed:
```bash
pixi run ros2 topic info /observations --verbose
```

### Force Sensor Strange Behavior
I'm noticing some strange behavior with the force sensor. When the robot is first spawned, the `wrist_wrench` z-axis immediately reads around 20N:
```yaml
wrist_wrench:
  header:
    stamp:
      sec: 10
      nanosec: 146000000
    frame_id: ati/tool_link
  wrench:
    force:
      x: 0.19647534913152367
      y: 0.5709232984170273
      z: 20.372605348811124
    torque:
      x: 0.15718355450545468
      y: -0.19727286686559287
      z: 0.006627747694545185
```
I tried taring the force sensor with this service call:
```bash
pixi run ros2 service call /aic_controller/tare_force_torque_sensor std_srvs/srv/Trigger
```
But the z-axis stubbornly remains at 20N.

Even more interestingly, when the robot presses down onto a hard surface and gets stuck, the z-axis value actually *decreases*:
![image](https://us1.discourse-cdn.com/flex022/uploads/ros/optimized/3X/9/6/96c7a976fc6b5613bf89b75dfc4689c39f6a462b_2_459x500.jpeg)
```yaml
wrist_wrench:
  header:
    stamp:
      sec: 72
      nanosec: 384000000
    frame_id: ati/tool_link
  wrench:
    force:
      x: -1.0565478791114393
      y: -2.8273204030726795
      z: 7.123609734869961
    torque:
      x: -0.9946708484649165
      y: -1.1107599148782015
      z: -0.07033790872857923
```
### My Current Hypotheses:

1. **Wrong Topic:** I might be looking at the wrong subtopic for the published TCP force feedback.
2. **Gripper Weight vs. Normal Force:** The constant 20N downward force might just be reading the mass of the gripper itself. When the robot presses against a surface, the normal force pushing back offsets the weight, which explains why the sensor reads *less* downward force.

So, while my custom teleop is subscribing properly now, it's definitely still not using the force-feedback correctly.

Spring Break officially starts today at Ohio State, so I’ll have the next 9 days of uninterrupted time to keep playing around with this! (Assuming I finish my midterm reviews first, of course 😅). Let me know if anyone has ideas on the 20N tare issue!

## Post 09
- Post #: `11`
- Created: `2026-03-14T20:07:33.651Z`
- Source: [Discourse Post #11](https://discourse.openrobotics.org/t/rockys-open-source-build-thread-ai-for-industry-challenge/53155/11)

Hey @jlamperez,

OMG amaaazinggg write up I really appreciate this! This is exactly what I'm looking for for the 20N issue!

I'm also really happy that great minds think alike -- using cheatcode to generate training data haha

I'm definitely going to implement your `get_observation` snippet to manually and get extract raw contact force out.

Thanks again for sharing your workflow and the video examples. This is a huge help!

Best, Rocky

## Post 10
- Post #: `12`
- Created: `2026-03-15T03:08:14.118Z`
- Source: [Discourse Post #12](https://discourse.openrobotics.org/t/rockys-open-source-build-thread-ai-for-industry-challenge/53155/12)

# March 14 Worklog Update

## Big Win Today: Force Feedback is Working

Thanks to @jlamperez's amazing writeup, I finally got force feedback working in my cheatcode teleop flow.

![image](https://us1.discourse-cdn.com/flex022/uploads/ros/optimized/3X/1/e/1ebe1673281e02eaa5c093f42202fc567bda8d2a_2_690x359.jpeg)

As you can see in the terminal output on the right side:

```yaml

TFs found! Starting APPROACH phase.

[CheatCode] Force: 0.2N | Phase: APPROACH | z_off: 0.2000

[CheatCode] Force: 3.1N | Phase: APPROACH | z_off: 0.2000

[CheatCode] Force: 1.5N | Phase: APPROACH | z_off: 0.2000

[CheatCode] Force: 1.6N | Phase: APPROACH | z_off: 0.2000

[CheatCode] Force: 1.3N | Phase: APPROACH | z_off: 0.2000

Hover reached (err=0.0063m). Entering ALIGN phase.

[CheatCode] Force: 1.2N | Phase: ALIGN | z_off: 0.0500

[CheatCode] Force: 1.1N | Phase: ALIGN | z_off: 0.0500

Aligned! (xy=0.0024m, ang=0.000rad, dwell=1.7s). Starting INSERT.

[CheatCode] Force: 0.6N | Phase: INSERT | z_off: 0.0500

[CheatCode] Force: 0.6N | Phase: INSERT | z_off: 0.0100

[CheatCode] Force: 0.9N | Phase: INSERT | z_off: -0.0100

[CheatCode] Force: 1.1N | Phase: INSERT | z_off: -0.0100

[CheatCode] Force: 5.8N | Phase: INSERT | z_off: -0.0150

[CheatCode] Force: 8.6N | Phase: INSERT | z_off: -0.0150

[CheatCode] Force: 10.5N | Phase: INSERT | z_off: -0.0150

[CheatCode] Force: 12.9N | Phase: INSERT | z_off: -0.0150

[CheatCode] Force: 12.6N | Phase: INSERT | z_off: -0.0150

[CheatCode] Force: 12.6N | Phase: INSERT | z_off: -0.0150

[CheatCode] Force: 12.4N | Phase: INSERT | z_off: -0.0150

[CheatCode] Force: 10.9N | Phase: INSERT | z_off: -0.0150

[CheatCode] Force: 12.7N | Phase: INSERT | z_off: -0.0150

[CheatCode] Force: 12.7N | Phase: INSERT | z_off: -0.0150

[CheatCode] Force: 12.9N | Phase: INSERT | z_off: -0.0150

[CheatCode] Force: 13.4N | Phase: INSERT | z_off: -0.0150

[CheatCode] Force: 12.9N | Phase: INSERT | z_off: -0.0150

[CheatCode] Force: 13.4N | Phase: INSERT | z_off: -0.0150

[CheatCode] Force: 13.0N | Phase: INSERT | z_off: -0.0150

[CheatCode] Force: 13.4N | Phase: INSERT | z_off: -0.0150

[CheatCode] Force: 13.7N | Phase: INSERT | z_off: -0.0150

[CheatCode] Force: 4.9N | Phase: INSERT | z_off: -0.0150

[CheatCode] Force: 6.6N | Phase: INSERT | z_off: -0.0150

[CheatCode] Force: 7.9N | Phase: INSERT | z_off: -0.0150

[CheatCode] Force: 9.6N | Phase: INSERT | z_off: -0.0150

[CheatCode] Force: 11.3N | Phase: INSERT | z_off: -0.0150

[CheatCode] Force: 13.1N | Phase: INSERT | z_off: -0.0150

(insertion completed)

Teleop loop time: 16.86ms (59 Hz))

```

The current state machine now incorporates force feedback, and this run gave me much better visibility into what is actually happening during insertion.

## What I Changed Today

### A) Added a full PROJECT_CONTEXT.md for future agents

I created a comprehensive context file at:

`mystuff/for_agents/PROJECT_CONTEXT.md`

It is 216 lines and includes:

- competition overview

- pipeline strategy

- repository structure

- `aic_teleop.py` architecture

- scoring system

- trial configurations

- technical details

- practical notes for AI agents

The goal is simple: if any future AI agent jumps into this repo, it should have enough context to contribute effectively without me re-explaining everything from scratch.

### B) Rewrote AICCheatCodeTeleop (v2) in `aic_teleop.py`

I made major upgrades to the cheatcode teleop state machine:

- Added a dedicated **ALIGN phase**

- Dwell at 5 cm hover height for at least 1 second

- Require XY error < 3 mm and angular error < 0.05 rad before INSERT

- Slowed insertion descent from **0.07 m/s → 0.02 m/s**

- Added **force-proportional insertion speed control**

- starts slowing at 5N

- fully pauses at 15N

- replaces old binary go/stop behavior

- Improved recovery behavior

- retreat only to 5 cm (not 20 cm)

- return to ALIGN (not APPROACH)

- maximum 3 retries

- Added live XY correction during INSERT by continuously re-reading plug TF

- Tightened phase transitions

- `dist_to_target < 0.01` (instead of the loose 0.2 value used before in recovery logic)

- Added angular convergence gate before entering INSERT

- Clamped max linear velocity during INSERT to 60% for gentler motion

- Tuned gains:

- `kp_linear`: 1.0 → 1.2

- `ki_linear`: 0.15 → 0.2

- `kp_angular`: 1.5 → 2.0

- `max_linear_vel`: 0.1 → 0.08

### C) Lowered retreat threshold from 18N to 10N

I originally wrote a recovery mode that retreats at 18N, but looking at the terminal evidence above, the robot can get badly stuck around **10-13N** without ever reaching 18N.

So with the old threshold, it could keep pushing while stuck and just hope for the best.

New behavior chain is now:

- **0-5N**: full speed

- **5-15N**: linear slowdown

- **≥15N**: full stop

- **≥10N for 0.5s**: retreat and retry

This should make the behavior much safer and reduce the chance of force-penalty disasters.

## Side Project Note

I have also been working on a parallel project I started a couple of months ago, but not consistently:

https://discourse.openrobotics.org/t/energy-efficient-autonomous-navigation-benchmarking/53208?u=rocky_shao

## Tooling Note (Slate)

I also found a new coding tool called Slate. A lot of today’s changes came from working through one giant, unorganized prompt with it. I’m definitely not an expert in coding, engineering, or AI-agent workflows yet, but this tool exceeded my expectations in the first hour, so I wanted to share that.

![image](https://us1.discourse-cdn.com/flex022/uploads/ros/optimized/3X/2/4/24d21be5bb1f2311ac3098f0fd89961ce188f69e_2_690x359.jpeg)

Slate also pointed out that my current cheatcode teleop is using **velocity control** instead of **position control**. I haven’t fully verified that myself yet, so I can’t confirm 100% today, but I’ll dig into it tomorrow.



Overall, today was a solid step forward: better force-aware behavior, cleaner state transitions, and better project context for future iterations.

## Post 11
- Post #: `13`
- Created: `2026-03-16T03:18:05.339Z`
- Source: [Discourse Post #13](https://discourse.openrobotics.org/t/rockys-open-source-build-thread-ai-for-industry-challenge/53155/13)

# **March 15 Update**

Caught up with homework and midterm review today, and spent the afternoon working with my Autonomous Car Team (definitely another topic for later!). Life gets in the way sometimes, so I only found a short window to work on this challenge today haha.

Here’s a quick demo: 
https://youtu.be/IrH-5pMJCRs

**State Machine Logic (AICCheatCodeTeleop):** I compacted the logic flow for how the arm handles the plug insertion:

* **INIT:** Waits for necessary TF frames (port, plug, gripper) before moving.
* **APPROACH ➡️ ALIGN:** Moves to hover 20cm above the port using a PI velocity controller. Once within 1cm, it drops to a 5cm hover to fine-tune the XY position (<3mm error) and angular orientation.
* **INSERT:** Descends into the port using a force-modulated speed (ramping down to a full stop if it hits 15N of resistance). Finishes when it reaches the -1.5cm insertion depth.
* **RECOVERY:** Active during Align/Insert. If force exceeds 10N for 0.5s, it retracts back to 5cm, resets the PI integrator, and tries again (max 10 retries before aborting).
* **DONE:** Terminal state (outputs zero velocity).
* **Controller Details:** Uses a PI controller for linear velocity and a P controller for angular. Speeds are clamped (and reduced by 40% during insertion), with world-frame velocities transformed into the TCP frame.

**Some Reflections:** Taking a step back, I'm reflecting on my journey with this challenge so far. Being so new to ROS and "real-world" robotics, the hardest part is taking a massive goal like "Train an AI to insert plugs into ports" and breaking it down into small enough chunks that I (with the help of AI) can actually execute and understand.

Time management is the other big hurdle. During school, it’s a race to finish homework so I can squeeze in robotics time. During breaks, the challenge flips entirely: how do you stick to a healthy routine and keep working when there's no pressure or anyone watching?

My main goal is to learn as much as possible (though winning is obviously nice!), so I’m constantly trying to find the sweet spot between letting AI coding tools accelerate my delivery and writing the code myself so I deeply understand the logic and implementation. 

Balancing "learning" with "delivering as fast as possible" is tough in this AI age.

## Post 12
- Post #: `14`
- Created: `2026-03-17T03:48:14.961Z`
- Source: [Discourse Post #14](https://discourse.openrobotics.org/t/rockys-open-source-build-thread-ai-for-industry-challenge/53155/14)

# March 16

When it comes to data collection, I've been facing a dilemma:

1. Spend the time to perfectly tune a PID controller to guarantee insertion 1st-try every time.
2. Rely on a "good enough" PID controller backed by a complex state machine, magic numbers, and exceptions, hoping it works.

Well, I went for path 2.

The old control state machine was basically this: the robot inserts, gets stuck, lifts all the way up, and tries again. I call this the "spam and pray" method. It worked very poorly. As you can see in this video, the insertion only succeeds about 1 out of 8 tries (though it did finally work at the end):
https://youtube.com/shorts/H_iYKAh8NQc

Obviously, the "spam and pray" approach was flawed. I started thinking: *How do I insert my USB-C cable to charge my phone?* I don't just mash it in and yank it all the way out if it fails. I insert the plug, and if it doesn't align with the port and gets stuck, I wiggle it a little bit to let it line up. Could I replicate this "wiggle" idea for the robot?

I decided to replace that old, aggressive pop-out loop with a new **SEARCH ("wiggle")** behavior, and it made recovery feel way less chaotic. Here is how I implemented it:

Instead of yanking the plug out the moment it gets stuck, the robot monitors the insertion force. If the force stays above **17N for 0.3 seconds**, we run a small **5 mm horizontal circular wiggle** while slowly creeping downward at **0.002 m/s**. If the force drops, it means we likely found the opening, and standard insertion resumes immediately. We only do a **mini-lift (~3 cm)** if three full wiggle cycles fail. This ensures the controller spends more time actually near the port instead of constantly resetting from scratch.

To make this work smoothly, I tuned the thresholds around a normal insertion load of ~14N:

* **15N:** Rampdown
* **17N:** Trigger the SEARCH/wiggle
* **19N:** Hard recovery
* **<20N:** Keep everything safely under the penalty zone.

Before this update, the aggressive recovery was basically sabotaging throughput. Once the force crossed the threshold, the arm would lift out way too far (initially 5 cm, later 2 cm), re-enter the ALIGN phase, and try again. It would easily get stuck in an infuriating *insert → recover → insert* loop 8+ times before getting lucky.

While building the wiggle, I also found and fixed a few underlying bugs that were kicking the robot into recovery in the first place. I realized the descent speed was frame-rate dependent (making it effectively 0.6m/s ~60x too fast!), which was causing immediate force spikes. Descent is now properly dt-scaled. I also made sure the DONE state can trigger from any phase (so a good seat-in doesn't get missed), zeroed out the integrators during APPROACH to avoid XY overshoot, and added a 5-second recovery timeout so the robot doesn't just sit there wiggling forever.


However, I'm quickly finding that the "wiggle" strategy has its own limitations. For the SC plug, if the little "horns" get caught outside the port, just wiggling in the horizontal plane doesn't do much (which is usually why it hits the max 3 attempts and fails). I'm not entirely sure if it's the horns causing the failure, or if the 5 mm wiggle radius is just too small. I checked with the collision box turned on for the port, and the horns actually seem to pass right through the port's collision box.
 
![image](https://us1.discourse-cdn.com/flex022/uploads/ros/optimized/3X/7/6/7680516acff6edc298105f667eb54367e8356c96_2_362x375.jpeg) 

Below is another frustrating failure mode I ran into: the plug is half-inserted, meaning the robot *should* keep inserting downward; however the force feedback on the gripper is around 14N, and a hard-coded safety threshold got triggered instead, halting the insertion process. 
![image](https://us1.discourse-cdn.com/flex022/uploads/ros/optimized/3X/4/9/499ac0a6b93108e6366ca4d56620d2f33436d76f_2_479x500.jpeg)

### Lots of Magic Numbers
Right now, I'm finding myself adding more and more hard-coded "magic numbers" to the state machine in the hopes of making data collection more robust. Honestly, I'm starting to change my mind. I think it might actually be better to bite the bullet, go back to Path 1, and build a truly robust control loop rather than constantly patching this state machine with edge cases and magic numbers.
https://github.com/Rocky0Shao/IntrinsicAIChallenge/blob/9389ff432bee01d1ebeb6f0503bbe6189fa365ab/aic_utils/lerobot_robot_aic/lerobot_robot_aic/aic_teleop.py#L361-L400

## Post 13
- Post #: `15`
- Created: `2026-03-18T03:39:38.844Z`
- Source: [Discourse Post #15](https://discourse.openrobotics.org/t/rockys-open-source-build-thread-ai-for-industry-challenge/53155/15)

#March 17
### Simplified CheatCode Teleop for Clean ACT Training Data**

**🎯 Today's Focus: State Machine Cleanup** Stripped the cheatcode teleop down to a clean, no-recovery state machine (`INIT → APPROACH → ALIGN → INSERT → DONE`). Removed all `SEARCH`/wiggle and `RECOVERY`/retreat phases to ensure the ACT model learns clean, first-try insertions rather than learning to fail and recover.

**🛠️ Key Technical Changes:**

* **Wrench Feedback:** Enabled compliance gains (`[0.5, 0.5, 0.5, 0, 0, 0]`) for automatic lateral correction during insertion.
* **XY-Only Integrator:** Now tracking plug-tip-to-port error (matching the official CheatCode architecture).
* **State Transitions:** The integrator is now preserved across the `ALIGN → INSERT` transition so built-up XY correction carries through. Angular gain is also reduced to 25% during insertion to prevent binding torques.
* **Tighter Tolerances:** Set to 0.5mm XY, 0.03rad angular, with a 2s minimum dwell.
* **Descent Profile:** Switched to a constant-speed descent (12mm/s) with a simple safety hold at 19.5N, replacing the force-proportional rampdown.

**🚀 Next Steps & Blockers:** I plan on implementing two separate cheatcode teleop configs to handle different port geometries:

* **SFP Connectors (Trials 1 & 2):** Tuned for rectangular geometry and tight chamfers.
* **SC Connectors (Trial 3):** Tuned for round geometry and spring-loaded latches. This requires slower insertion speeds, longer alignment dwell, and higher force thresholds to handle natural force spikes.

**Target:** Once both configs hit 100% first-try insertions in sim, I will start collecting the 50+ demonstration episodes per trial needed for ACT model training.

## Post 14
- Post #: `18`
- Created: `2026-03-19T17:41:31.501Z`
- Source: [Discourse Post #18](https://discourse.openrobotics.org/t/rockys-open-source-build-thread-ai-for-industry-challenge/53155/18)

> Hui_Liu wrote:
>  Thanks for sharing! Why we can’t just save off video sensor/action information directly in the CheatCode policy itself? Is it because LeRobot has the built-in data collection (and even training framework)? 

*(First of all, please take the information below with large chunks of salt, as I have no prior experience using LeRobot!)*

Yes, exactly! LeRobot has its own CLI tools for recording training data and a built-in training framework. It can also automatically upload data to a Git provider called Hugging Face (which feels like a GitHub specifically for machine-learning data).

Here is an example from the [challenge's official documentation](https://github.com/intrinsic-dev/aic/tree/main/aic_utils/lerobot_robot_aic). If you scroll down, you can see how to train, and it redirects you to the LeRobot documentation:

```Bash
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

As for what actually gets recorded, the data consists of:

* Linear and angular speed of the gripper tip
* Camera feeds from the 3 cameras

From [my Hugging Face dataset](https://huggingface.co/datasets/rockyshao22/Intrinsic_AI/viewer/default/train), it looks something like this:
![image](https://us1.discourse-cdn.com/flex022/uploads/ros/optimized/3X/a/b/abbe5e0eecffbb2c0e5b5ecdf33066afd1ec5acc_2_690x442.png)

So far, I've only verified that the data recording works—I haven't even verified if the collected data is correct yet XD.

Currently, I'm working on creating a ground-truth-based teleop method to collect graceful, 1st-try, 100% insertions for my future ACT model (since LeRobot has built-in tools for training).

I'm strictly following the "garbage in, garbage out" school of thought, haha!

## Post 15
- Post #: `19`
- Created: `2026-03-19T17:47:35.494Z`
- Source: [Discourse Post #19](https://discourse.openrobotics.org/t/rockys-open-source-build-thread-ai-for-industry-challenge/53155/19)

Hi Ludvig, thank you soooo much! 
I really appreciate the kind words genours offer, this will be a game changer for me haha!

I reached out to you on LinkedIn, would love to chat more!
Rocky Shao
