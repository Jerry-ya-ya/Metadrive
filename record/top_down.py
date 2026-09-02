import argparse
from pathlib import Path

import imageio
import torch
from stable_baselines3 import PPO

from config import MODEL_PATH, VIDEO_DIR
from env_utils import make_metadrive_env, print_scoreboard

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default=str(MODEL_PATH))
    parser.add_argument("--output", type=str, default=str(VIDEO_DIR / "metadrive_driving_td_video.mp4"))
    parser.add_argument("--steps", type=int, default=100000)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--screen-size", type=int, default=600)
    args = parser.parse_args()

    VIDEO_DIR.mkdir(exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = PPO.load(args.model_path, device=device)

    env = make_metadrive_env()
    obs, info = env.reset()

    frames = []
    total_reward = 0.0
    steps = 0

    print("Recording top-down driving video...")

    for _ in range(args.steps):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += float(reward)
        steps += 1

        frame = env.render(
            mode="top_down",
            window=False,
            screen_size=(args.screen_size, args.screen_size),
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
