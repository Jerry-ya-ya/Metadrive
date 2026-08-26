import numpy as np

from env_utils import make_metadrive_env


def main():
    env = make_metadrive_env()

    obs, info = env.reset()

    print("\n===== Observation Space =====")
    print(env.observation_space)

    print("\n===== Observation =====")
    print("Type:", type(obs))

    if isinstance(obs, dict):
        for key, value in obs.items():
            print(f"\n[{key}]")
            print("Shape:", value.shape)
            print("Dtype:", value.dtype)
            print("Min:", np.min(value))
            print("Max:", np.max(value))

    # ===== State inspection =====
    if isinstance(obs, dict) and "state" in obs:
        state = obs["state"]

        print("\n===== State Dimensions =====")
        print("Total dimensions:", len(state))

        for i, value in enumerate(state):
            print(f"state[{i:02d}] = {value:.6f}")

    # ===== Image inspection =====
    if isinstance(obs, dict) and "image" in obs:
        image = obs["image"]

        print("\n===== Image =====")
        print("Image shape:", image.shape)
        print("Image dtype:", image.dtype)

    env.close()


if __name__ == "__main__":
    main()