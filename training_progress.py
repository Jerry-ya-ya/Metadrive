"""Emit line-oriented PPO progress so the web job log can show live activity."""

import time

from stable_baselines3.common.callbacks import BaseCallback, CallbackList


class TrainingProgressCallback(BaseCallback):
    def __init__(self, total_timesteps, report_every=5):
        super().__init__()
        self.total_timesteps = max(1, int(total_timesteps))
        self.report_every = max(1, int(report_every))
        self.start_timesteps = 0
        self.last_reported = 0
        self.started_at = None

    def _on_training_start(self):
        self.start_timesteps = self.num_timesteps
        self.started_at = time.monotonic()
        print(
            f"Training started: collecting {self.total_timesteps:,} environment steps.",
            flush=True,
        )

    def _on_step(self):
        completed = max(0, self.num_timesteps - self.start_timesteps)
        if completed - self.last_reported < self.report_every and completed < self.total_timesteps:
            return True

        elapsed = max(time.monotonic() - self.started_at, 1e-6)
        rate = completed / elapsed
        percent = min(100.0, completed * 100 / self.total_timesteps)
        remaining = max(0, self.total_timesteps - completed)
        eta_minutes = int((remaining / rate if rate > 0 else 0) // 60)
        print(
            f"Training progress: {completed:,}/{self.total_timesteps:,} steps "
            f"({percent:.2f}%, {rate:.2f} steps/s, ETA {eta_minutes} min).",
            flush=True,
        )
        self.last_reported = completed
        return True


def training_callbacks(checkpoint_callback, total_timesteps):
    return CallbackList(
        [checkpoint_callback, TrainingProgressCallback(total_timesteps)]
    )
