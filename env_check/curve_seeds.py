import sys

import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO

from config import MODEL_PATH
from env_utils import make_metadrive_env

def classify_curve(env):
    base_env = env.unwrapped

    for block in base_env.current_map.blocks:
        if getattr(block, "ID", None) == "C":
            direction = block.config.get("dir")

            if direction == 1:
                return "right"
            elif direction == 0:
                return "left"

    return "unknown"

def evaluate_seed(model, seed, max_steps=1000):
    env = make_metadrive_env({
        "map": "C",
        "start_seed": seed,
        "num_scenarios": 1,
    })

    try:
        obs, info = env.reset(seed=seed)
        curve_type = classify_curve(env)

        total_reward = 0.0
        steering_values = []
        steps = 0

        for _ in range(max_steps):
            action, _states = model.predict(obs, deterministic=True)
            steering_values.append(float(np.asarray(action).reshape(-1)[0]))

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            steps += 1

            if terminated or truncated:
                break

        if info.get("arrive_dest"):
            status = "arrive_dest"
        elif info.get("out_of_road"):
            status = "out_of_road"
        elif info.get("crash"):
            status = "crash"
        elif info.get("max_step") or steps >= max_steps:
            status = "max_step"
        else:
            status = "unknown"

        steering = np.asarray(steering_values, dtype=np.float32)
        result = {
            "seed": seed,
            "curve": curve_type,
            "status": status,
            "success": status == "arrive_dest",
            "completion": info.get("route_completion", 0.0) * 100,
            "reward": total_reward,
            "steps": steps,
            "steering_mean": float(steering.mean()) if steering.size else 0.0,
            "steering_std": float(steering.std()) if steering.size else 0.0,
            "steering_min": float(steering.min()) if steering.size else 0.0,
            "steering_max": float(steering.max()) if steering.size else 0.0,
        }

        return result
    finally:
        env.close()

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model-path",
        type=str,
        default=str(MODEL_PATH),
        help=f"PPO model path (default: {MODEL_PATH})",
    )

    parser.add_argument(
        "--start-seed",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--count",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--output",
        type=str,
        default="curve_seed_results.csv",
    )

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_path = Path(args.model_path)
    if not model_path.is_absolute():
        model_path = PROJECT_ROOT / model_path

    model = PPO.load(
        model_path,
        device=device,
    )

    results = []

    for seed in range(
        args.start_seed,
        args.start_seed + args.count
    ):
        print(f"\nTesting seed {seed}...")

        result = evaluate_seed(
            model,
            seed
        )

        results.append(result)

        print(
            f"Seed {seed:02d} | "
            f"{result['curve']:>8} | "
            f"{result['status']:>12} | "
            f"completion={result['completion']:.1f}% | "
            f"reward={result['reward']:.1f}"
        )

    df = pd.DataFrame(results)

    print("\n===== All Results =====")
    print(df)

    print("\n===== Curve Statistics =====")

    if "curve" in df:
        summary = (
            df.groupby("curve")
            .agg(
                episodes=("seed", "count"),
                success_rate=("success", "mean"),
                mean_completion=("completion", "mean"),
                mean_reward=("reward", "mean"),
                mean_steps=("steps", "mean"),
            )
        )

        summary["success_rate"] *= 100

        print(summary)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig"
    )

    print(f"\nSaved results to: {output_path}")

if __name__ == "__main__":
    main()