from pathlib import Path

import matplotlib.pyplot as plt

from env_utils import make_metadrive_env
from config import OUTPUT_DIR, METADRIVE_CONFIG

OUTPUT_DIR.mkdir(exist_ok=True)

env = make_metadrive_env()
obs, info = env.reset()

frame = env.render(
    mode="top_down",
    window=False,
    screen_size=(1200, 1200),
)

output_path = OUTPUT_DIR / "map_preview.png"

plt.figure(figsize=(10, 10))
plt.imshow(frame)
plt.axis("off")
plt.title(f"MetaDrive Map Preview: {METADRIVE_CONFIG.get('map')}")
plt.savefig(output_path, bbox_inches="tight", dpi=150)
plt.close()

print(f"Saved map preview to {output_path}")

env.close()
