import sys
from pathlib import Path

# Allow both of these invocation styles:
#   python env_check/preview_map.py
#   python -m env_check.preview_map
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt

from config import OUTPUT_DIR, METADRIVE_CONFIG
from env_utils import make_metadrive_env

def main() -> None:
    output_dir = OUTPUT_DIR if OUTPUT_DIR.is_absolute() else PROJECT_ROOT / OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "map_preview.png"

    env = make_metadrive_env()
    try:
        env.reset()
        # Gymnasium's Wrapper.render() accepts no keyword arguments.  MetaDrive's
        # top-down renderer does, so call the underlying MetaDriveEnv directly.
        frame = env.unwrapped.render(
            mode="top_down",
            window=False,
            screen_size=(1200, 1200),
        )

        if frame is None:
            raise RuntimeError("MetaDrive did not return a top-down render frame.")

        figure, axes = plt.subplots(figsize=(10, 10))
        try:
            axes.imshow(frame)
            axes.axis("off")
            axes.set_title(f"MetaDrive Map Preview: {METADRIVE_CONFIG.get('map')}")
            figure.savefig(output_path, bbox_inches="tight", dpi=150)
        finally:
            plt.close(figure)
    finally:
        env.close()

    print(f"Saved map preview to {output_path}")

if __name__ == "__main__":
    main()
