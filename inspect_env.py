from env_utils import make_metadrive_env
from config import METADRIVE_CONFIG

env = make_metadrive_env()
obs, info = env.reset()

print("=" * 48)
print("MetaDrive Environment Specification")
print("=" * 48)
print("Config map:", METADRIVE_CONFIG.get("map"))
print("Traffic density:", METADRIVE_CONFIG.get("traffic_density"))
print()
print("Observation space:", env.observation_space)
print("Initial observation shape:", getattr(obs, "shape", None))
print()
print("Action space:", env.action_space)
print("Random action example:", env.action_space.sample())
print("=" * 48)

env.close()
