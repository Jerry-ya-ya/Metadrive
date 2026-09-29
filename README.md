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

## Interactive CLI

Start the project menu with:

```bash
python cli.py
```

The CLI can:

- start a new training run with a selected learning rate;
- discover model ZIP files under `models`, `checkpoints`, `models_backup`, and
  `model_backup`, then continue training one;
- browse tools grouped under `analyze`, `env_check`, `record`, and `statistic`;
- inspect each tool's `argparse` options and prompt for them automatically.

After either a new training run or a continued training run finishes
successfully, the CLI directly starts `run_model_tests.ps1` with the saved model.
The PowerShell trigger then runs model evaluation followed by first-person
recording. Selecting no at the post-training prompt skips this sequence.

Python tools are scanned recursively from those four package folders whenever
the CLI starts, so newly added scripts appear without editing `cli.py`. Files
named `__init__.py` and cache folders are skipped.

To inspect or launch the discovered tools without the interactive menu:

```bash
python cli.py --list-tools
python cli.py --run-tool analyze.model_evaluate -- --episodes 5
```

## Web Control Center (Docker)

The browser interface mirrors the CLI's main capabilities: it can start new
training, continue from a discovered model, choose the learning rate and map,
run the post-training evaluation and first-person recording sequence, and
launch tools auto-discovered from `analyze`, `env_check`, `record`, and
`statistic`.

Each successfully saved model also receives a neighboring
`<model>.metadata.json` file containing its map. Continued training leaves the
map field blank by default and reuses that saved value. Checkpoints additionally
carry the map inside the model ZIP; legacy models without either value fall back
to the current map in `config.py`. Entering a map during continued training
overrides and saves the new value. Evaluation and recording automatically use
the same resolved map.

Build and start the container with:

```bash
docker compose up --build
```

The default image installs the CPU PyTorch wheel to keep the image portable and
avoid bundling several gigabytes of CUDA libraries. To use a compatible custom
PyTorch wheel index, set `TORCH_INDEX_URL` before building the image.

Then open:

```text
http://localhost:4000
```

The FastAPI backend and frontend are served together on port `4000`. Compose
binds the port to `127.0.0.1`, so the control center is only reachable from the
same computer. Only one training or tool job runs at a time; its current stage
and combined terminal output are available in the Jobs page. A running job can
also be cancelled there.

Project result folders are mounted into the container, so models, checkpoints,
logs, reports, and videos remain on the host after the container stops. Stop it
with:

```bash
docker compose down
```

For local development without Docker, install the requirements and run:

```bash
uvicorn webapp.backend:app --host 0.0.0.0 --port 4000
```

This control center starts local Python processes and is intended for a trusted
machine or private network; it does not provide user authentication.

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
python train.py --timesteps 50000 --test-name sc_50k --record-seed 85
```

Choose a map explicitly with, for example:

```bash
python train.py --timesteps 50000 --map XSSORC
python continue_train.py --model-path models/ppo_metadrive.zip --map SC
```

Omit `--map` from `continue_train.py` to reuse the model's previous map.

After saving the trained model, `train.py` automatically starts
`run_model_tests.ps1`. It evaluates the saved model and records one complete
first-person test run under `model_backup/<test-name>/`. If `--test-name` is
omitted, a timestamped name such as `training_20260925_153000` is generated.

Post-training options include:

```text
--test-episodes 5
--test-max-steps 1000
--record-steps 1000
--record-seed 0
--record-fps 30
--record-screen-size 672
```

Use `--skip-post-test` only when training should save the model without running
the evaluation and recording sequence.

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
