import sys
from pathlib import Path

# Allow both of these invocation styles:
#   python env_check/preview_map.py
#   python -m env_check.preview_map
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
from collections import Counter

import numpy as np

from env_utils import make_metadrive_env

def get_curve_config(env):
    """
    Find the Curve block in the generated MetaDrive map
    and return its procedural-generation parameters.
    """
    base_env = env.unwrapped

    for block in base_env.current_map.blocks:
        if getattr(block, "ID", None) == "C":
            config = block.config

            return {
                "dir": int(config["dir"]),
                "radius": float(config["radius"]),
                "angle": float(config["angle"]),
                "length": float(config["length"]),
            }

    return None


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--start-seed",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--count",
        type=int,
        default=100,
    )

    args = parser.parse_args()

    results = []

    for seed in range(
        args.start_seed,
        args.start_seed + args.count
    ):
        env = make_metadrive_env({
            "map": "C",
            "start_seed": seed,
            "num_scenarios": 1,
        })

        env.reset()

        curve = get_curve_config(env)

        if curve is not None:
            # 目前已由 seed 0 人工確認：
            # dir=1 -> right
            # dir=0 -> left
            direction = (
                "right"
                if curve["dir"] == 1
                else "left"
            )

            curve["seed"] = seed
            curve["direction"] = direction

            results.append(curve)

        env.close()

    if not results:
        print("No curve blocks found.")
        return

    directions = Counter(
        result["direction"]
        for result in results
    )

    total = len(results)

    left_count = directions["left"]
    right_count = directions["right"]

    print("\n===== Curve Seed Statistics =====")

    print(f"Seed range : {args.start_seed} ~ "
          f"{args.start_seed + args.count - 1}")

    print(f"Total      : {total}")

    print(
        f"Left       : {left_count} "
        f"({left_count / total:.2%})"
    )

    print(
        f"Right      : {right_count} "
        f"({right_count / total:.2%})"
    )

    print("\n===== Geometry Statistics =====")

    for direction in ["left", "right"]:

        subset = [
            result
            for result in results
            if result["direction"] == direction
        ]

        if not subset:
            continue

        radius = np.array(
            [r["radius"] for r in subset]
        )

        angle = np.array(
            [r["angle"] for r in subset]
        )

        length = np.array(
            [r["length"] for r in subset]
        )

        print(f"\n[{direction.upper()}]")

        print(
            f"Radius : mean={radius.mean():.2f}, "
            f"std={radius.std():.2f}, "
            f"min={radius.min():.2f}, "
            f"max={radius.max():.2f}"
        )

        print(
            f"Angle  : mean={angle.mean():.2f}, "
            f"std={angle.std():.2f}, "
            f"min={angle.min():.2f}, "
            f"max={angle.max():.2f}"
        )

        print(
            f"Length : mean={length.mean():.2f}, "
            f"std={length.std():.2f}, "
            f"min={length.min():.2f}, "
            f"max={length.max():.2f}"
        )

    print("\n===== Seed Lists =====")

    left_seeds = [
        r["seed"]
        for r in results
        if r["direction"] == "left"
    ]

    right_seeds = [
        r["seed"]
        for r in results
        if r["direction"] == "right"
    ]

    print("Left seeds :", left_seeds)
    print("Right seeds:", right_seeds)


if __name__ == "__main__":
    main()