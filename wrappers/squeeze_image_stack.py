import gymnasium as gym
import numpy as np
from gymnasium import spaces


class ImagePreprocessWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

        old_space = env.observation_space

        image_space = old_space["image"]
        state_space = old_space["state"]

        # MetaDrive:
        # (84, 84, 3, 1)
        #
        # After squeeze:
        # (84, 84, 3)
        #
        # After transpose:
        # (3, 84, 84)

        h, w, c, _ = image_space.shape

        self.observation_space = spaces.Dict({
            "image": spaces.Box(
                low=0.0,
                high=1.0,
                shape=(c, h, w),
                dtype=np.float32,
            ),
            "state": state_space,
        })

    def observation(self, obs):
        obs = dict(obs)

        image = obs["image"]

        # (84, 84, 3, 1)
        # ->
        # (84, 84, 3)
        image = np.squeeze(image, axis=-1)

        # HWC -> CHW
        # (84, 84, 3)
        # ->
        # (3, 84, 84)
        image = np.transpose(
            image,
            (2, 0, 1)
        )

        obs["image"] = image.astype(np.float32)

        return obs