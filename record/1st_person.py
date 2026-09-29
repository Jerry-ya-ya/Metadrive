import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import imageio
import numpy as np
import torch
from stable_baselines3 import PPO

from config import METADRIVE_CONFIG, MODEL_PATH, VIDEO_DIR
from env_utils import get_final_status, make_metadrive_env
from model_metadata import resolve_model_map


def _describe_observation(obs):
    if not isinstance(obs, dict):
        return [f"Observation type  : {type(obs).__name__}"]

    lines = []
    for key, value in obs.items():
        lines.append(
            f"Observation {key:<6}: shape={value.shape}, dtype={value.dtype}"
        )
    return lines


def _safe_stats(values):
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def record_first_person(
    model,
    output_path,
    report_path,
    *,
    model_path,
    map_name=None,
    steps=1000,
    fps=30,
    screen_size=672,
    seed=0,
):
    if steps < 1:
        raise ValueError("steps must be at least 1")
    if fps < 1:
        raise ValueError("fps must be at least 1")
    if screen_size < 1:
        raise ValueError("screen_size must be at least 1")

    output_path = Path(output_path)
    report_path = Path(report_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    env = make_metadrive_env({
        "map": map_name or METADRIVE_CONFIG["map"],
        "start_seed": seed,
        "num_scenarios": 1,
    })

    try:
        obs, info = env.reset()
        observation_lines = _describe_observation(obs)
        frames = []
        steering_values = []
        throttle_values = []
        total_reward = 0.0
        completed_steps = 0

        for _ in range(steps):
            action, _states = model.predict(obs, deterministic=True)
            steering_values.append(float(action[0]))
            throttle_values.append(float(action[1]))

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            completed_steps += 1

            frame = np.transpose(obs["image"], (1, 2, 0))
            frame = (frame * 255).clip(0, 255).astype("uint8")
            frame = cv2.resize(
                frame,
                (screen_size, screen_size),
                interpolation=cv2.INTER_NEAREST,
            )
            frames.append(frame)

            if terminated or truncated:
                break

        imageio.mimsave(output_path, frames, fps=fps)

        steering = _safe_stats(steering_values)
        throttle = _safe_stats(throttle_values)
        steering_changes = np.diff(steering_values)
        mean_abs_delta = (
            float(np.mean(np.abs(steering_changes)))
            if steering_changes.size
            else 0.0
        )
        max_abs_delta = (
            float(np.max(np.abs(steering_changes)))
            if steering_changes.size
            else 0.0
        )
        sign_changes = int(
            np.sum(
                np.sign(steering_values[1:])
                != np.sign(steering_values[:-1])
            )
        )

        result = {
            "status": get_final_status(info),
            "completion": info.get("route_completion", 0.0) * 100,
            "reward": total_reward,
            "steps": completed_steps,
            "steering": steering,
            "throttle": throttle,
            "mean_abs_steering_delta": mean_abs_delta,
            "max_abs_steering_delta": max_abs_delta,
            "steering_sign_changes": sign_changes,
        }

        lines = [
            "MetaDrive First-Person Recording",
            "=" * 48,
            f"Model             : {model_path}",
            f"Map               : {map_name or METADRIVE_CONFIG['map']}",
            f"Seed              : {seed}",
            f"Video             : {output_path}",
            f"Observation space : {env.observation_space}",
            f"Feature extractor : {model.policy.features_extractor}",
            *observation_lines,
            "",
            "Episode Result",
            "=" * 48,
            f"Final status      : {result['status']}",
            f"Route completion  : {result['completion']:.1f}%",
            f"Total steps       : {result['steps']}",
            f"Total reward      : {result['reward']:.2f}",
            "",
            "Steering",
            "=" * 48,
            f"Mean              : {steering['mean']:.6f}",
            f"Std               : {steering['std']:.6f}",
            f"Min               : {steering['min']:.6f}",
            f"Max               : {steering['max']:.6f}",
            f"Mean abs delta    : {mean_abs_delta:.6f}",
            f"Max abs delta     : {max_abs_delta:.6f}",
            f"Sign changes      : {sign_changes}",
            "",
            "Throttle",
            "=" * 48,
            f"Mean              : {throttle['mean']:.6f}",
            f"Std               : {throttle['std']:.6f}",
            f"Min               : {throttle['min']:.6f}",
            f"Max               : {throttle['max']:.6f}",
        ]
        report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return result
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default=str(MODEL_PATH))
    parser.add_argument(
        "--output",
        type=str,
        default=str(VIDEO_DIR / "metadrive_driving_1stp_video.mp4"),
    )
    parser.add_argument("--report-output", type=str)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--screen-size", type=int, default=672)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--map",
        dest="map_name",
        help="Override the model's stored MetaDrive map.",
    )
    args = parser.parse_args()

    model_path = Path(args.model_path)
    if not model_path.is_absolute():
        model_path = PROJECT_ROOT / model_path

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path

    report_path = (
        Path(args.report_output)
        if args.report_output
        else output_path.with_suffix(".txt")
    )
    if not report_path.is_absolute():
        report_path = PROJECT_ROOT / report_path

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = PPO.load(model_path, device=device)
    map_name = resolve_model_map(
        model_path,
        args.map_name,
        model=model,
        default_map=METADRIVE_CONFIG["map"],
    )
    record_first_person(
        model,
        output_path,
        report_path,
        model_path=model_path,
        map_name=map_name,
        steps=args.steps,
        fps=args.fps,
        screen_size=args.screen_size,
        seed=args.seed,
    )
    print(f"Saved first-person video to: {output_path}")
    print(f"Saved recording report to: {report_path}")


if __name__ == "__main__":
    main()
