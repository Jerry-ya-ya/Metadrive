# MetaDrive PPO Project

This project is a cleaned-up version of the teacher's MetaDrive demo.

It trains a PPO agent on MetaDrive using:

```text
MetaDriveEnv
PPO
MlpPolicy
State / LiDAR observation
Top-down recording
Checkpoint saving
TensorBoard logging
```

## Difference from CarRacing

| Project | Observation | Policy | Action |
|---|---|---|---|
| CarRacing | 96x96 RGB image | CnnPolicy | steering, gas, brake |
| MetaDrive | vehicle state + LiDAR | MlpPolicy | steering, throttle/brake |

This project does **not** use `CnnPolicy` by default because MetaDrive's default observation is a numerical state vector, not a camera image.

## Install

```bash
pip install -r requirements.txt
```

If you use Windows and installation fails, try:

```bash
pip install swig
pip install -r requirements.txt
```

## Check GPU

```bash
python check_cuda.py
```

MetaDrive with `MlpPolicy` may still be CPU-heavy because the simulator runs on CPU. GPU helps the neural network part, but the environment simulation may remain the bottleneck.

## Preview Map

```bash
python preview_map.py
```

This saves a top-down map preview to:

```text
outputs/map_preview.png
```

## Train

Smoke test:

```bash
python train.py --timesteps 10000
```

More serious training:

```bash
python train.py --timesteps 50000
```

## Continue Training

```bash
python continue_train.py --timesteps 50000
```

## Evaluate

```bash
python evaluate.py --episodes 5
```

## Record Video

```bash
python record_video.py --steps 1000
```

The output video will be saved to:

```text
videos/metadrive_driving_video.mp4
```

## TensorBoard

```bash
tensorboard --logdir logs
```

Then open:

```text
http://localhost:6006
```

## Default Environment Config

The default config is close to the teacher's example:

```python
map = "XSSORC"
traffic_density = 0.0
use_render = False
```

You can edit it in:

```text
config.py
```

## Suggested Experiments

After confirming the project runs:

```text
1. Train 10k steps to verify the environment
2. Train 500k steps on traffic_density=0.0
3. Change traffic_density to 0.1
4. Change traffic_density to 0.2
5. Compare route_completion and crash rate
```
