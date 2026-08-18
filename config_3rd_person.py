from pathlib import Path
from metadrive.component.sensors.rgb_camera import RGBCamera

# Main folders
MODEL_DIR = Path("models")
CHECKPOINT_DIR = Path("checkpoints")
LOG_DIR = Path("logs")
VIDEO_DIR = Path("videos")
OUTPUT_DIR = Path("outputs")

MODEL_PATH = MODEL_DIR / "ppo_metadrive"

# Teacher-style default MetaDrive config
METADRIVE_CONFIG = dict(
    use_render=False,

    # Teacher demo used:
    # map="S" for simple environment test
    # map="XSSORC" for longer mixed road layout
    map="SO",

    # 0.0 = no NPC traffic, easier for first RL training.
    # Try 0.1 or 0.2 after your agent can drive on empty roads.
    traffic_density=0.0,

    # Make episode length clear and controllable.
    horizon=1000,

    # Useful when running in notebooks or repeated scripts.
    force_render_fps=30,

    num_scenarios=50,
    start_seed=0,

    # 降低高速獎勵
    speed_reward=0.05,

    # 保留前進獎勵
    driving_reward=1.0,

    # 提高失控懲罰
    out_of_road_penalty=10.0,
    crash_vehicle_penalty=10.0,
    crash_object_penalty=10.0,
)