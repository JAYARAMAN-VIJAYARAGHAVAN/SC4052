# Hierarchical Quantum Reinforcement Learning for ViZDoom


## Project idea

This project explores **two complementary forms of hierarchy** in one quantum reinforcement learning (QRL) architecture:

1. **Hierarchical control:** A high-level decision layer selects goals or temporally extended behaviors; a low-level decision layer executes the movement and combat actions required to achieve them.
2. **Navigation–combat decision split at the higher layer:** The high-level system decides whether the agent should focus on **navigation** or **combat**, then routes control to the appropriate specialized behavior.

The intended end state is not simply two unrelated quantum agents. It is an **integrated, coordinated hierarchy** in which the navigation/combat split and high-/low-level control are both part of the architecture. The long-term plan is for the learned decision-making components—including low-level movement control currently implemented with heuristics—to be quantum policies.

## Intended architecture

```text
                    ViZDoom observation / game state
                                  |
                                  v
                  HIGH-LEVEL QUANTUM DECISION LAYER
                    /                         \
             NAVIGATION                      COMBAT
                 |                              |
        Select navigation goal          Select combat objective
        / room / subgoal                 or combat behavior
                 |                              |
                 +---------------+--------------+
                                 |
                                 v
                   LOW-LEVEL QUANTUM CONTROL
              Execute movement, turning, dodging,
                 attacking, and recovery actions
                                 |
                                 v
                          ViZDoom actions
                                 |
                         Feedback / rewards
```

**Design note:** This diagram represents the *target architecture*, not a claim that all of its quantum components have been trained or integrated. The precise division of responsibility between the high-level combat branch and the shared low-level controller is an intended design choice to validate during integration.

## Current implementation

### Navigation (`nav/`)

- An **8-qubit, 3-layer variational quantum circuit** is used in a PPO navigation policy.
- The policy receives visual features from a CNN plus **22 engineered navigation features**.
- It chooses among **four candidate neighboring sectors/passages**.
- A sector graph tracks candidate boundaries, observed transitions, visited areas, and failed attempts.
- A **classical steering controller** currently turns and moves toward selected passage targets, with basic stuck recovery. This is a temporary implementation and baseline, **not** the intended final quantum low-level controller.

### Combat (`corridor/`)

- A separate quantum PPO combat policy has been developed using ViZDoom's Deadly Corridor environment.
- Its action space includes forward movement, strafing, turning, and shooting.
- A CNN processes stacked game frames; a variational quantum circuit contributes to the policy output.
- An auxiliary enemy-detection signal has been explored for combat/navigation coordination.

### E1M1 integration (`e1m1/`)

- A shared ViZDoom game wrapper loads E1M1 (using Freedoom Phase 1 by default, or a supplied IWAD).
- Integration code connects navigation, combat, and a coordinator to the same game instance.
- Navigation-only playback and heuristic baselines are being tested.
- Current navigation can become stuck or circle; reliable completion, map-aware routing, and full quantum low-level control remain future work.

## Evaluation ideas

| Measurement | Purpose |
|---|---|
| Rooms / unique cells visited | Navigation coverage |
| Blocked passages / recovery events | Low-level control reliability |
| Enemy encounters / kills / survival | Combat effectiveness |
| Time and actions to objective | Control efficiency |
| E1M1 completion rate | End-to-end task success |
| Hierarchical vs. flat policy | Value of hierarchical design |
| Split navigation/combat vs. unified policy | Value of behavior specialization |
| Quantum vs. classical policy baselines | Quantum-model comparison |
