import sys
from pathlib import Path

# Allow both of these invocation styles:
#   python env_check/preview_map.py
#   python -m env_check.preview_map
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# python -m statistic.curve_score --model models/ppo_metadrive.zip --start-seed 50 --count 50

import argparse
from collections import defaultdict

import numpy as np
from stable_baselines3 import PPO

from env_utils import make_metadrive_env

def get_curve_info(env):
    """
    Read curve geometry directly from MetaDrive PGBlock config.
    """
    base_env = env.unwrapped

    for block in base_env.current_map.blocks:
        if getattr(block, "ID", None) == "C":
            config = block.config

            direction = int(config["dir"])

            return {
                "direction": "right" if direction == 1 else "left",
                "dir": direction,
                "radius": float(config["radius"]),
                "angle": float(config["angle"]),
                "length": float(config["length"]),
            }

    return None

def evaluate_one_seed(model, seed):
    env = make_metadrive_env({
        "map": "SC",
        "start_seed": seed,
        "num_scenarios": 1,
    })
    obs, info = env.reset()

    curve_info = get_curve_info(env)

    if curve_info is None:
        env.close()
        raise RuntimeError(
            f"Seed {seed}: Curve block not found."
        )

    total_reward = 0.0
    steps = 0

    terminated = False
    truncated = False

    final_info = {}

    while not (terminated or truncated):
        action, _ = model.predict(
            obs,
            deterministic=True,
        )

        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += float(reward)
        steps += 1
        final_info = info

    env.close()

    # MetaDrive usually exposes arrive_dest / out_of_road
    arrive_dest = bool(
        final_info.get("arrive_dest", False)
    )

    out_of_road = bool(
        final_info.get("out_of_road", False)
    )

    route_completion = float(
        final_info.get("route_completion", 0.0)
    )

    if arrive_dest:
        status = "arrive_dest"
    elif out_of_road:
        status = "out_of_road"
    elif truncated:
        status = "truncated"
    else:
        status = "terminated"

    return {
        "seed": seed,
        "direction": curve_info["direction"],
        "dir": curve_info["dir"],
        "radius": curve_info["radius"],
        "angle": curve_info["angle"],
        "length": curve_info["length"],
        "success": arrive_dest,
        "status": status,
        "completion": route_completion,
        "reward": total_reward,
        "steps": steps,
    }

def print_group_statistics(name, results):
    if not results:
        print(f"\n===== {name} =====")
        print("No samples.")
        return

    success_count = sum(
        result["success"]
        for result in results
    )

    total = len(results)

    rewards = np.array(
        [result["reward"] for result in results],
        dtype=np.float32,
    )

    completions = np.array(
        [result["completion"] for result in results],
        dtype=np.float32,
    )

    steps = np.array(
        [result["steps"] for result in results],
        dtype=np.float32,
    )

    print(f"\n===== {name} =====")
    print(f"Episodes         : {total}")
    print(
        f"Success          : {success_count}/{total} "
        f"({success_count / total:.1%})"
    )

    print(
        f"Completion       : "
        f"mean={completions.mean():.3f}, "
        f"std={completions.std():.3f}"
    )

    print(
        f"Reward           : "
        f"mean={rewards.mean():.2f}, "
        f"std={rewards.std():.2f}"
    )

    print(
        f"Steps            : "
        f"mean={steps.mean():.1f}, "
        f"std={steps.std():.1f}"
    )

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to PPO model zip file.",
    )

    parser.add_argument(
        "--start-seed",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--count",
        type=int,
        default=50,
    )

    args = parser.parse_args()

    print("Loading model...")
    model = PPO.load(
        args.model,
        device="auto",
    )

    results = []

    end_seed = args.start_seed + args.count

    print(
        f"\nEvaluating seeds "
        f"{args.start_seed} ~ {end_seed - 1}"
    )

    for seed in range(
        args.start_seed,
        end_seed
    ):
        result = evaluate_one_seed(
            model,
            seed,
        )

        results.append(result)

        success_text = (
            "PASS"
            if result["success"]
            else "FAIL"
        )

        print(
            f"Seed {seed:3d} | "
            f"{result['direction']:5s} | "
            f"{success_text:4s} | "
            f"completion={result['completion']:.3f} | "
            f"reward={result['reward']:.2f} | "
            f"steps={result['steps']}"
        )

    left_results = [
        result
        for result in results
        if result["direction"] == "left"
    ]

    right_results = [
        result
        for result in results
        if result["direction"] == "right"
    ]

    print("\n")
    print("=" * 60)
    print("MetaDrive Curve Evaluation Summary")
    print("=" * 60)

    print_group_statistics(
        "OVERALL",
        results,
    )

    print_group_statistics(
        "LEFT",
        left_results,
    )

    print_group_statistics(
        "RIGHT",
        right_results,
    )

    print("\n===== Failed Seeds =====")

    failed = [
        result
        for result in results
        if not result["success"]
    ]

    if not failed:
        print("None")
    else:
        for result in failed:
            print(
                f"Seed {result['seed']:3d} | "
                f"{result['direction']:5s} | "
                f"{result['status']:12s} | "
                f"completion={result['completion']:.3f} | "
                f"radius={result['radius']:.2f} | "
                f"angle={result['angle']:.2f}"
            )

if __name__ == "__main__":
    main()
