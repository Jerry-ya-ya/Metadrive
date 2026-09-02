from env_utils import make_metadrive_env

env = make_metadrive_env()
obs, info = env.reset()

for step in range(500):
    action = [0.0, 0.2]

    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        break

env.close()