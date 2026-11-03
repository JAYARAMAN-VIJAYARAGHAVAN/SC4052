
import argparse
from collections import Counter

import numpy as np
import torch
from torch.distributions import Categorical

from . import config as C
from .environment import RoomNavigationEnv, OPTION_NAMES
from .model.networks import QuantumActorCritic
from .algo.ppo import tensor_obs


def load_model(checkpoint):
    model = QuantumActorCritic()

    # Initialize the lazy CNN layer before loading weights.
    dummy = {
        "image": np.zeros(
            (C.FRAME_STACK, C.SCREEN_SIZE, C.SCREEN_SIZE),
            dtype=np.float32
        ),
        "nav": np.zeros(C.N_NAV_FEATURES, dtype=np.float32),
        "mask": np.ones(C.N_OPTIONS, dtype=np.bool_)
    }

    with torch.no_grad():
        model(tensor_obs(dummy))

    ckpt = torch.load(
        checkpoint, map_location="cpu", weights_only=False
    )

    if ckpt.get("experiment") != C.EXPERIMENT_NAME:
        raise ValueError(
            f"Checkpoint experiment {ckpt.get('experiment')} "
            f"does not match {C.EXPERIMENT_NAME}"
        )

    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt


def select_action(policy, obs, model, rng, deterministic=False):
    valid = np.flatnonzero(obs["mask"])

    if len(valid) == 0:
        raise RuntimeError("No valid navigation options")

    if policy == "frontier":
        frontier = OPTION_NAMES.index("FRONTIER")
        return frontier if frontier in valid else int(valid[0])

    if policy == "random":
        return int(rng.choice(valid))

    if policy == "qnn":
        with torch.no_grad():
            logits, _ = model(tensor_obs(obs))
            if deterministic:
                return int(logits.argmax(dim=-1).item())

            probabilities = torch.softmax(
                logits.reshape(-1), dim=-1
            ).cpu().numpy()

        # Local RNG makes policy sampling reproducible.
        return int(rng.choice(len(probabilities), p=probabilities))

    raise ValueError(f"Unknown policy: {policy}")


def evaluate(policy, args, model=None):
    successes = []
    game_actions = []
    successful_actions = []
    rooms = []
    returns = []
    option_counts = []
    all_choices = Counter()
    end_reasons = Counter()

    print(f"\n{'=' * 65}")
    print(f"Policy: {policy}")
    print(f"Episodes: {args.episodes}")
    print(f"Seeds: {args.seed} to {args.seed + args.episodes - 1}")
    print(f"Maximum options: {C.MAX_OPTIONS}")
    print(f"{'=' * 65}")

    for ep in range(args.episodes):
        seed = args.seed + ep
        rng = np.random.default_rng(seed + 100000)

        # Recreate environment with the same seed for each policy.
        env = RoomNavigationEnv(
            visible=not args.headless, seed=seed
        )

        counts = Counter()
        total_reward = 0.0

        try:
            obs, _ = env.reset()
            info = None

            for step in range(C.MAX_OPTIONS):
                action = select_action(
                    policy, obs, model, rng,
                    deterministic=args.deterministic
                )

                counts[OPTION_NAMES[action]] += 1

                obs, reward, terminated, truncated, info = env.step(
                    action
                )
                total_reward += reward

                if terminated or truncated:
                    break

            if info is None:
                raise RuntimeError("Episode completed without a step")

            success = bool(info["success"])
            actions = int(info["game_actions"])

            successes.append(success)
            game_actions.append(actions)
            rooms.append(info["rooms"])
            returns.append(total_reward)
            option_counts.append(step + 1)
            all_choices.update(counts)

            if success:
                successful_actions.append(actions)

            reason = info.get("end_reason") or (
                "game_finished" if terminated
                else "truncated" if truncated
                else "option_limit"
            )
            end_reasons[reason] += 1

            print(
                f"Episode {ep + 1:3d} | "
                f"success={success} | "
                f"options={step + 1:2d} | "
                f"actions={actions:3d} | "
                f"rooms={info['rooms']:2d} | "
                f"return={total_reward:+.2f} | "
                f"end={reason}",
                flush=True
            )

        finally:
            env.close()

    print(f"\n--- {policy.upper()} SUMMARY ---")
    print(
        f"Success rate: {100 * np.mean(successes):.1f}% "
        f"({sum(successes)}/{len(successes)})"
    )
    print(f"Mean game actions (all): {np.mean(game_actions):.1f}")
    if successful_actions:
        print(
            f"Mean game actions (successes): "
            f"{np.mean(successful_actions):.1f}"
        )
    print(f"Mean rooms explored: {np.mean(rooms):.2f}")
    print(f"Mean options used: {np.mean(option_counts):.2f}")
    print(f"Mean return: {np.mean(returns):+.2f}")
    print(f"Action distribution: {dict(all_choices)}")
    print(f"End reasons: {dict(end_reasons)}")

    return {
        "policy": policy,
        "success_rate": float(np.mean(successes)),
        "mean_actions": float(np.mean(game_actions)),
        "mean_rooms": float(np.mean(rooms)),
        "mean_return": float(np.mean(returns)),
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--seed", type=int, default=800)
    parser.add_argument(
        "--policy",
        choices=["qnn", "frontier", "random", "all"],
        default="qnn"
    )
    parser.add_argument(
        "--checkpoint",
        default=str(C.CHECKPOINT_DIR / "latest.pt")
    )

    args = parser.parse_args()
    torch.set_num_threads(1)

    policies = (
        ["frontier", "random", "qnn"]
        if args.policy == "all"
        else [args.policy]
    )

    model = None
    if "qnn" in policies:
        model, ckpt = load_model(args.checkpoint)
        print(
            f"Experiment={C.EXPERIMENT_NAME} | "
            f"trained options={ckpt['steps']}"
        )

    results = []

    for policy in policies:
        results.append(evaluate(policy, args, model))

    if len(results) > 1:
        print("\nFINAL COMPARISON")
        print(f"{'Policy':<14} {'Success':>10} {'Actions':>10} {'Rooms':>10}")
        print("-" * 48)
        for r in results:
            print(
                f"{r['policy']:<14} "
                f"{r['success_rate'] * 100:>9.1f}% "
                f"{r['mean_actions']:>10.1f} "
                f"{r['mean_rooms']:>10.2f}"
            )


if __name__ == "__main__":
    main()
