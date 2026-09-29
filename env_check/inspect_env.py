# Allow both of these invocation styles:
#   python env_check/preview_map.py
#   python -m env_check.preview_map
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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
