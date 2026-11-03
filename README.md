# Quantum Reinforcement Learning for DOOM (ViZDoom)

A research prototype exploring whether **variational quantum circuits (VQCs)** can serve as decision-making policies in reinforcement learning agents that play DOOM through **ViZDoom**.

The project uses **Proximal Policy Optimization (PPO)**, **PyTorch**, and **PennyLane**, with classical visual encoders and quantum policy components. It currently contains separate experiments for combat and room-level navigation, plus an experimental integration targeting the full **E1M1** level.


## Goals

1. Train a quantum-enhanced PPO agent to act from DOOM observations.
3. Investigate **hierarchical control**: a high-level navigation policy selects where to go, while a lower-level controller handles movement.
4. Explore combining specialized **navigation** and **combat** policies within one ViZDoom instance.
5. Ultimately navigate and complete **E1M1**, including doors, obstacles, enemies, and the exit.

## System overview

```text
                    ViZDoom observations
                             |
                   +---------+---------+
                   |                   |
             Navigation             Combat
             experiment            experiment
                   |                   |
          CNN + engineered       CNN visual encoder
          map/room features            |
                   |             Quantum policy
          Quantum room-option          |
               policy           Movement / shooting
                   |
         Classical local steering
         and waypoint following
                   |                   |
                   +---------+---------+
                             |
                     ViZDoom actions
```

The integrated E1M1 prototype is intended to coordinate these components, but integration and reliable level completion remain experimental.

## Evaluation plan

To understand whether the quantum component helps, compare policies under the **same maps, seeds, action budgets, and low-level controller**:

| Experiment | High-level decision maker | Low-level control |
| --- | --- | --- |
| Classical baseline | Exploration heuristic | Shared waypoint/steering controller |
| Quantum baseline | Pretrained navigation QNN | Shared waypoint/steering controller |
| Quantum fine-tuned | QNN further trained on E1M1 | Shared waypoint/steering controller |
| Classical learned baseline (planned) | Classical PPO policy | Shared waypoint/steering controller |

Suggested metrics include unique rooms visited, unique spatial cells explored, failed passages, recovery events, actions per successful transition, episode return, and **E1M1 completion rate**. Training speed and model parameter counts should also be reported when comparing quantum and classical approaches.

No comparative performance or level-completion claims are made here until measured experimentally.

