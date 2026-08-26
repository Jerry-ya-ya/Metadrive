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
    map="S",

    # 0.0 = no NPC traffic, easier for first RL training.
    # Try 0.1 or 0.2 after your agent can drive on empty roads.
    traffic_density=0.0,

    # Make episode length clear and controllable.
    horizon=1000,

    # Useful when running in notebooks or repeated scripts.
    force_render_fps=30,

    # Number of random scenarios available from start_seed.
    # This must be at least 1; zero makes every scenario seed invalid.
    num_scenarios=1,

    start_seed=0,

     # ===== Vision observation =====
    image_observation=True,

    # RGB camera
    sensors={
        "rgb_camera": (
            RGBCamera,
            84,
            84,
        )
    },

    # Tell the vehicle which camera is used as observation
    vehicle_config={
        "image_source": "rgb_camera",
    },

    # Number of consecutive frames
    stack_size=1,

    # Normalize RGB values to 0~1
    norm_pixel=True,

    # ===== Reward =====
    # use_lateral_reward=True,
)