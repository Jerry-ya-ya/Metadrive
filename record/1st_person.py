import argparse
from pathlib import Path
import numpy as np

import imageio
import torch
from stable_baselines3 import PPO

from config import MODEL_PATH, VIDEO_DIR
from env_utils import make_metadrive_env, print_scoreboard

import cv2

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default=str(MODEL_PATH))
    parser.add_argument("--output", type=str, default=str(VIDEO_DIR / "metadrive_driving_1stp_video.mp4"))
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

    steering_values = []
    throttle_values = []

    for _ in range(args.steps):
        action, _states = model.predict(obs, deterministic=True)

        steering = float(action[0])
        throttle_values.append(float(action[1]))
        steering_values.append(steering)
        
        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += float(reward)
        steps += 1

        # PPO receives CHW image: (3, 84, 84)
        frame = obs["image"]

        # Convert back to HWC for video: (84, 84, 3)
        frame = np.transpose(frame, (1, 2, 0))

        # float32 [0, 1] -> uint8 [0, 255]
        frame = (frame * 255).clip(0, 255).astype("uint8")

        # Enlarge video
        frame = cv2.resize(
            frame,
            (672, 672),
            interpolation=cv2.INTER_NEAREST
        )

        frames.append(frame)

        print("Action:", action)
        print("Action shape:", action.shape)

        if terminated or truncated:
            break

    print("\n===== Features Extractor =====")
    print(model.policy.features_extractor)
    print(env.observation_space)
    print(type(model.policy.features_extractor))

    print_scoreboard(total_reward, steps, info)

    print("\n===== Steering =====")
    print("Mean:", np.mean(steering_values))
    print("Std:", np.std(steering_values))
    print("Min:", np.min(steering_values))
    print("Max:", np.max(steering_values))

    print("\n===== Throttle =====")
    print("Mean:", np.mean(throttle_values))
    print("Std:", np.std(throttle_values))
    print("Min:", np.min(throttle_values))
    print("Max:", np.max(throttle_values))
    env.close()

    output_path = Path(args.output)
    imageio.mimsave(output_path, frames, fps=args.fps)
    print(f"Saved video to {output_path}")

if __name__ == "__main__":
    main()
