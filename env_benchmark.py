import time

from config import METADRIVE_CONFIG
from metadrive import MetaDriveEnv

env = MetaDriveEnv(METADRIVE_CONFIG)

obs, info = env.reset()

steps = 1000

start = time.perf_counter()

for _ in range(steps):
    action = env.action_space.sample()

    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        obs, info = env.reset()

elapsed = time.perf_counter() - start

print(f"Steps: {steps}")
print(f"Time: {elapsed:.2f}s")
print(f"Environment FPS: {steps / elapsed:.2f}")

env.close()