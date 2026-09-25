import argparse
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

from config import MODEL_DIR, CHECKPOINT_DIR, LOG_DIR, MODEL_PATH
from env_utils import build_vec_env


PROJECT_ROOT = Path(__file__).resolve().parent
TEST_RUNNER = PROJECT_ROOT / "run_model_tests.ps1"


def resolve_model_path(path_value):
    model_path = Path(path_value)
    if not model_path.is_absolute():
        model_path = PROJECT_ROOT / model_path
    return model_path


def saved_model_path(model_path):
    return model_path if model_path.suffix == ".zip" else Path(f"{model_path}.zip")


def run_post_training_tests(args, model_path):
    powershell = shutil.which("pwsh") or shutil.which("powershell")
    if powershell is None:
        raise RuntimeError("PowerShell is required to run run_model_tests.ps1.")
    if not TEST_RUNNER.is_file():
        raise FileNotFoundError(f"Test runner not found: {TEST_RUNNER}")

    test_name = args.test_name or datetime.now().strftime("training_%Y%m%d_%H%M%S")
    command = [
        powershell,
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(TEST_RUNNER),
        "-TestName",
        test_name,
        "-ModelPath",
        str(model_path),
        "-Episodes",
        str(args.test_episodes),
        "-MaxSteps",
        str(args.test_max_steps),
        "-RecordSteps",
        str(args.record_steps),
        "-Seed",
        str(args.record_seed),
        "-Fps",
        str(args.record_fps),
        "-ScreenSize",
        str(args.record_screen_size),
        "-PythonExecutable",
        sys.executable,
    ]

    print(f"\nStarting post-training test run: {test_name}")
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=50_000)
    parser.add_argument("--model-path", type=str, default=str(MODEL_PATH))
    parser.add_argument("--checkpoint-freq", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--test-name", type=str)
    parser.add_argument("--test-episodes", type=int, default=5)
    parser.add_argument("--test-max-steps", type=int, default=1000)
    parser.add_argument("--record-steps", type=int, default=1000)
    parser.add_argument("--record-seed", type=int, default=0)
    parser.add_argument("--record-fps", type=int, default=30)
    parser.add_argument("--record-screen-size", type=int, default=672)
    parser.add_argument("--skip-post-test", action="store_true")

    args = parser.parse_args()

    positive_values = {
        "timesteps": args.timesteps,
        "checkpoint-freq": args.checkpoint_freq,
        "test-episodes": args.test_episodes,
        "test-max-steps": args.test_max_steps,
        "record-steps": args.record_steps,
        "record-fps": args.record_fps,
        "record-screen-size": args.record_screen_size,
    }
    invalid_names = [name for name, value in positive_values.items() if value < 1]
    if invalid_names:
        parser.error(f"These values must be at least 1: {', '.join(invalid_names)}")

    model_path = resolve_model_path(args.model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    MODEL_DIR.mkdir(exist_ok=True)
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    LOG_DIR.mkdir(exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    env = build_vec_env()

    print("\n===== Observation Space =====")
    print(env.observation_space)

    model = PPO(
        "MultiInputPolicy",
        env,
        verbose=1,
        device=device,
        # seed=args.seed,
        tensorboard_log=str(LOG_DIR),
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        policy_kwargs=dict(
            normalize_images=False
        ),
    )

    print("\n===== Observation Features =====")
    print(model.policy.features_extractor)

    checkpoint_callback = CheckpointCallback(
        save_freq=args.checkpoint_freq,
        save_path=str(CHECKPOINT_DIR),
        name_prefix="ppo_metadrive",
    )

    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=checkpoint_callback,
            progress_bar=True,
        )

        model.save(model_path)
    finally:
        env.close()

    model_path = saved_model_path(model_path)
    print(f"Saved model to {model_path}")

    if args.skip_post_test:
        print("Skipped post-training tests.")
    else:
        run_post_training_tests(args, model_path)

if __name__ == "__main__":
    main()
