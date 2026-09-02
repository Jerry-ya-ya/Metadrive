import sys
from pathlib import Path

# Allow both of these invocation styles:
#   python env_check/preview_map.py
#   python -m env_check.preview_map
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    
from env_utils import make_metadrive_env

env = make_metadrive_env({
    "map": "C",
    "start_seed": 0,
    "num_scenarios": 1,
})

obs, info = env.reset()

base_env = env.unwrapped

print("Current map type:", type(base_env.current_map))

print("\n===== Map attributes =====")

for name in dir(base_env.current_map):
    if "block" in name.lower():
        try:
            value = getattr(base_env.current_map, name)
            print(name, "=", value)
        except Exception:
            pass

for i, block in enumerate(base_env.current_map.blocks):
    print(f"\nBlock {i}")
    print("Type:", type(block))
    print("Object:", block)

    print("\n--- Possible parameter attributes ---")

    for name in [
        "config",
        "_config",
        "sample_parameters",
        "_sample_parameters",
        "parameter_space",
        "_parameter_space",
        "block_network_type",
        "ID",
    ]:
        if hasattr(block, name):
            try:
                print(f"{name} =", getattr(block, name))
            except Exception as e:
                print(f"{name} = <error: {e}>")

    print("\n--- Attributes containing parameter/config/dir ---")

    for name in dir(block):
        lower = name.lower()

        if any(keyword in lower for keyword in [
            "param",
            "config",
            "dir",
            "angle",
            "radius",
        ]):
            try:
                value = getattr(block, name)

                if not callable(value):
                    print(f"{name} =", value)
            except Exception:
                pass

env.close()