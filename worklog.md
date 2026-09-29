# Metadrive

## 2026/08/14

- Connect project to remote Git repository.

## 2026/08/18

- Add learning rate config and performance benchmark, and reduce scenarios to one.

## 2026/08/26

- Add no-steering test and model observation, and update wrapper.

## 2026/08/30

- Add image observation preprocessing and update training and video diagnostics.

- Ignore Python cache files and directories.

## 2026/09/03

- Organize MetaDrive utilities and add multi-seed curve evaluation.

## 2026/09/07

- Add curve seed statistics and make utility scripts runnable from subdirectories.

## 2026/09/11

- Add seeded SC-curve recording with steering stability diagnostics.

## 2026/09/22

- Add sequential model test automation with file reports and example recordings.

## 2026/09/26

- Add an interactive CLI for configurable training and auto-discovered project tools.

- Trigger model evaluation and recording after CLI training runs.

## 2026/09/28

- Add a containerized web control center for training, model evaluation, recording, and auto-discovered CLI tools on port 4000.

## 2026/09/29

- Add persistent MetaDrive map settings to training and evaluation with a responsive, collapsible web interface.

- Reduce MetaDrive training memory use and add persistent retries for failed jobs.

- Enable CUDA GPU training for the Docker web app with device validation and status reporting.

- Show live PPO training progress and stabilize MetaDrive offscreen rendering in Docker.
