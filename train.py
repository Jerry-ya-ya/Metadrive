import argparse
from pathlib import Path

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

from config import MODEL_DIR, CHECKPOINT_DIR, LOG_DIR, MODEL_PATH
from env_utils import build_vec_env

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--model-path", type=str, default=str(MODEL_PATH))
    parser.add_argument("--checkpoint-freq", type=int, default=25_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    MODEL_DIR.mkdir(exist_ok=True)
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    LOG_DIR.mkdir(exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    env = build_vec_env()

    model = PPO(
        "MultiInputPolicy",
        env,
        verbose=1,
        device=device,
        seed=args.seed,
        tensorboard_log=str(LOG_DIR),
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=args.checkpoint_freq,
        save_path=str(CHECKPOINT_DIR),
        name_prefix="ppo_metadrive",
    )

    model.learn(
        total_timesteps=args.timesteps,
        callback=checkpoint_callback,
        progress_bar=True,
    )

    model.save(args.model_path)
    print(f"Saved model to {args.model_path}.zip")

    env.close()

if __name__ == "__main__":
    main()
