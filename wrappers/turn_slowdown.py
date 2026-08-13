import gymnasium as gym
import numpy as np

class TurnSlowDownWrapper(gym.ActionWrapper):
    def __init__(self, env, max_throttle=0.6, turn_slow_factor=0.8):
        super().__init__(env)
        self.max_throttle = max_throttle
        self.turn_slow_factor = turn_slow_factor

    def action(self, action):
        action = np.array(action, dtype=np.float32)

        steering = action[0]
        throttle = action[1]

        throttle = np.clip(throttle, -1.0, self.max_throttle)

        throttle_limit = self.max_throttle * (
            1 - self.turn_slow_factor * abs(steering)
        )

        action[1] = min(throttle, throttle_limit)
        return action

    def render(self, *args, **kwargs):
        return self.env.render(*args, **kwargs)

    def close(self):
        return self.env.close()