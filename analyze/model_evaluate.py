import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from stable_baselines3 import PPO

from config import METADRIVE_CONFIG, MODEL_PATH, OUTPUT_DIR
from env_utils import get_final_status, make_metadrive_env
from model_metadata import resolve_model_map


def run_one_episode(model, max_steps, map_name=None):
    config_override = {"map": map_name} if map_name else None
    env = make_metadrive_env(config_override)
    try:
        obs, info = env.reset()
        total_reward = 0.0
        steps = 0

        for _ in range(max_steps):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            steps += 1

            if terminated or truncated:
                break

        return {
            "reward": total_reward,
            "completion": info.get("route_completion", 0.0) * 100,
            "steps": steps,
            "status": get_final_status(info),
        }
    finally:
        env.close()


def evaluate_model(model, episodes=5, max_steps=1000, map_name=None):
    if episodes < 1:
        raise ValueError("episodes must be at least 1")
    if max_steps < 1:
        raise ValueError("max_steps must be at least 1")

    return [run_one_episode(model, max_steps, map_name) for _ in range(episodes)]


def format_evaluation_report(results, model_path, map_name=None):
    lines = [
        "MetaDrive Model Evaluation",
        "=" * 48,
        f"Model             : {model_path}",
        f"Map               : {map_name or METADRIVE_CONFIG['map']}",
        f"Episodes          : {len(results)}",
        "",
    ]

    for index, result in enumerate(results, start=1):
        lines.append(
            f"Episode {index:02d} | "
            f"reward={result['reward']:.2f} | "
            f"completion={result['completion']:.1f}% | "
            f"steps={result['steps']} | "
            f"status={result['status']}"
        )

    mean_reward = sum(item["reward"] for item in results) / len(results)
    mean_completion = sum(item["completion"] for item in results) / len(results)
    statuses = [item["status"] for item in results]

    lines.extend([
        "",
        "Evaluation Summary",
        "=" * 48,
        f"Mean reward       : {mean_reward:.2f}",
        f"Mean completion   : {mean_completion:.1f}%",
        f"Statuses          : {statuses}",
    ])
    return "\n".join(lines) + "\n"


def save_evaluation_report(results, model_path, output_path, map_name=None):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        format_evaluation_report(results, model_path, map_name),
        encoding="utf-8",
    )
    return output_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default=str(MODEL_PATH))
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument(
        "--map",
        dest="map_name",
        help="Override the model's stored MetaDrive map.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(OUTPUT_DIR / "model_evaluation.txt"),
    )
    args = parser.parse_args()

    model_path = Path(args.model_path)
    if not model_path.is_absolute():
        model_path = PROJECT_ROOT / model_path

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = PPO.load(model_path, device=device)
    map_name = resolve_model_map(
        model_path,
        args.map_name,
        model=model,
        default_map=METADRIVE_CONFIG["map"],
    )
    results = evaluate_model(model, args.episodes, args.max_steps, map_name)
    save_evaluation_report(results, model_path, output_path, map_name)
    print(f"Saved evaluation report to: {output_path}")


if __name__ == "__main__":
    main()
