import gymnasium as gym
import numpy as np

class SteeringBoostWrapper(gym.ActionWrapper):
    def __init__(self, env, steering_scale=1.8):
        super().__init__(env)
        self.steering_scale = steering_scale

    def action(self, action):
        action = np.array(action, dtype=np.float32)

        action[0] = np.clip(
            action[0] * self.steering_scale,
            -1.0,
            1.0
        )

        return action

    def reset(self, **kwargs):
        self.last_steering = 0.0
        kwargs.pop("options", None)
        return self.env.reset(**kwargs)

    def render(self, *args, **kwargs):
        return self.env.render(*args, **kwargs)

    def close(self):
        return self.env.close()