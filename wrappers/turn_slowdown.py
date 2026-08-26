import gymnasium as gym
import numpy as np

class TurnSlowDownWrapper(gym.ActionWrapper):
    def __init__(
        self,
        env,
        max_throttle=0.6,
        turn_slow_factor=0.8
    ):
        super().__init__(env)

        self.max_throttle = max_throttle
        self.turn_slow_factor = turn_slow_factor

    def reset(self, *, seed=None, options=None):
            return self.env.reset(seed=seed)

    def action(self, action):
        action = np.array(
            action,
            dtype=np.float32
        ).copy()

        steering = float(action[0])
        throttle = float(action[1])

        # Global throttle limit
        throttle = np.clip(
            throttle,
            -1.0,
            self.max_throttle
        )

        # Reduce maximum throttle as steering increases
        throttle_limit = (
            self.max_throttle
            * (
                1.0
                - self.turn_slow_factor
                * abs(steering)
            )
        )

        action[1] = min(
            throttle,
            throttle_limit
        )

        return action