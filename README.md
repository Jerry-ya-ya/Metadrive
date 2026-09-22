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
python -m env_check.preview_map
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
python -m analyze.model_evaluate --episodes 5 --output outputs/model_evaluation.txt
```

The evaluation summary is written to `outputs/model_evaluation.txt`.

## Record Video

```bash
python -m record.1st_person --steps 1000 --seed 85
```

The default video and readable report are saved to:

```text
videos/metadrive_driving_1stp_video.mp4
videos/metadrive_driving_1stp_video.txt
```

## Run the Model Test Suite

Use the PowerShell trigger script to run the evaluation first and the
first-person recording second. Each test remains an independent Python command;
the trigger stops immediately if either command fails.

```powershell
powershell -ExecutionPolicy Bypass -File .\run_model_tests.ps1 -TestName "sc_seed_85" -ModelPath "models\ppo_metadrive.zip" -Episodes 5 -MaxSteps 1000 -RecordSteps 1000 -Seed 85
```

`-TestName` is required and becomes the result folder name. Each name must be
new so an earlier test result is not overwritten. The command creates
`model_backup` automatically when needed and writes:

```text
model_backup/
└── sc_seed_85/
    ├── evaluation.txt
    ├── recording.txt
    ├── first_person.mp4
    └── run_summary.txt
```

The reports are produced by their corresponding Python scripts, while
`run_summary.txt` confirms that the complete sequence finished successfully.

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
