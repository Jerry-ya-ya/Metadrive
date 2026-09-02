import argparse

import torch
from stable_baselines3 import PPO

from config import MODEL_PATH
from env_utils import make_metadrive_env, print_scoreboard

def run_one_episode(model, max_steps):
    env = make_metadrive_env()
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

    env.close()
    return total_reward, steps, info

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default=str(MODEL_PATH))
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=1000)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = PPO.load(args.model_path, device=device)

    rewards = []
    completions = []
    statuses = []

    for ep in range(args.episodes):
        total_reward, steps, info = run_one_episode(model, args.max_steps)
        rewards.append(total_reward)
        completions.append(info.get("route_completion", 0.0) * 100)

        if info.get("arrive_dest"):
            status = "arrive_dest"
        elif info.get("crash"):
            status = "crash"
        elif info.get("out_of_road"):
            status = "out_of_road"
        else:
            status = "unknown"
        statuses.append(status)

        print(f"Episode {ep + 1}: reward={total_reward:.2f}, completion={completions[-1]:.1f}%, status={status}")

    print()
    print("=" * 48)
    print("Evaluation Summary")
    print("=" * 48)
    print(f"Mean reward      : {sum(rewards) / len(rewards):.2f}")
    print(f"Mean completion  : {sum(completions) / len(completions):.1f}%")
    print(f"Statuses         : {statuses}")
    print("=" * 48)

if __name__ == "__main__":
    main()
