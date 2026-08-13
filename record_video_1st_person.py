import argparse
from pathlib import Path

import imageio
import torch
from stable_baselines3 import PPO

from config import MODEL_PATH, VIDEO_DIR
from env_utils import make_metadrive_env, print_scoreboard

import cv2

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default=str(MODEL_PATH))
    parser.add_argument("--output", type=str, default=str(VIDEO_DIR / "metadrive_driving_video.mp4"))
    parser.add_argument("--steps", type=int, default=100000)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--screen-size", type=int, default=600)
    args = parser.parse_args()

    VIDEO_DIR.mkdir(exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = PPO.load(args.model_path, device=device)

    env = make_metadrive_env()
    obs, info = env.reset()

    print(env.observation_space)
    print(type(obs))

    if isinstance(obs, dict):
        for key, value in obs.items():
            print(key, value.shape, value.dtype)

    frames = []
    total_reward = 0.0
    steps = 0

    print("Recording 1st person driving video...")

    for _ in range(args.steps):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += float(reward)
        steps += 1

        frame = obs["image"][..., -1]

        frame = (frame * 255).clip(0, 255).astype("uint8")

        frame = cv2.resize(
            frame,
            (672, 672),
            interpolation=cv2.INTER_NEAREST
        )

        frames.append(frame)

        if terminated or truncated:
            break

    print_scoreboard(total_reward, steps, info)

    env.close()

    output_path = Path(args.output)
    imageio.mimsave(output_path, frames, fps=args.fps)
    print(f"Saved video to {output_path}")

if __name__ == "__main__":
    main()
