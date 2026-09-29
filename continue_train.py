# python continue_train.py --timesteps 50000 --learning-rate 1e-4

import argparse
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

from config import MODEL_DIR, CHECKPOINT_DIR, LOG_DIR, MODEL_PATH, METADRIVE_CONFIG
from env_utils import build_vec_env
from model_metadata import resolve_model_map, save_model_metadata, saved_model_path
from training_device import select_training_device
from training_progress import training_callbacks


PROJECT_ROOT = Path(__file__).resolve().parent


def resolve_model_path(path_value):
    model_path = Path(path_value)
    if not model_path.is_absolute():
        model_path = PROJECT_ROOT / model_path
    return model_path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=25_000)
    parser.add_argument("--model-path", type=str, default=str(MODEL_PATH))
    parser.add_argument("--checkpoint-freq", type=int, default=5_000)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument(
        "--map",
        dest="map_name",
        help="Override the stored MetaDrive map; omit to reuse the previous map.",
    )
    args = parser.parse_args()

    MODEL_DIR.mkdir(exist_ok=True)
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    LOG_DIR.mkdir(exist_ok=True)

    device = select_training_device()
    print(f"Using device: {device}")

    model_path = resolve_model_path(args.model_path)
    model = PPO.load(model_path, device=device)
    map_name = resolve_model_map(
        model_path,
        args.map_name,
        model=model,
        default_map=METADRIVE_CONFIG["map"],
    )
    print(f"Using map: {map_name}")

    env = build_vec_env({"map": map_name})
    model.set_env(env)
    model.metadrive_map = map_name

    model.learning_rate = 1e-4
    model.lr_schedule = lambda _: 1e-4

    if args.learning_rate is not None:
        lr = args.learning_rate

        model.learning_rate = lr
        model.lr_schedule = lambda _: lr

        print(f"Override learning rate: {lr}")

    checkpoint_callback = CheckpointCallback(
        save_freq=args.checkpoint_freq,
        save_path=str(CHECKPOINT_DIR),
        name_prefix="ppo_metadrive_continue",
    )
    callbacks = training_callbacks(checkpoint_callback, args.timesteps)

    try:
        print("Starting PPO training and collecting environment steps...", flush=True)
        model.learn(
            total_timesteps=args.timesteps,
            reset_num_timesteps=False,
            callback=callbacks,
            progress_bar=True,
        )
        model.save(model_path)
    finally:
        env.close()

    model_path = saved_model_path(model_path)
    metadata_path = save_model_metadata(model_path, map=map_name)
    print(f"Updated model saved to {model_path}")
    print(f"Saved model metadata to {metadata_path}")

if __name__ == "__main__":
    main()
