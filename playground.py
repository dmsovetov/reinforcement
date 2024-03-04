from typing import Tuple

from gymnasium import Env
from gymnasium.spaces import Discrete

from environments import QuantizationObservationTransformer


class WindyGridworldEnv(Env):
    def __init__(self, width: int = 10, height: int = 7):
        self.width = width
        self.height = height
        self.state = self.state_from_point(0, 3)
        self.observation_space = Discrete(width * height)
        self.action_space = Discrete(4)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = self.state_from_point(0, 3)
        return self.state, dict()

    def step(self, action):
        x, y = self.point_from_state(self.state)
        wind_value = 0

        if x in [3, 4, 5, 8]:
            wind_value = 1
        if x in [6, 7]:
            wind_value = 2

        if action == 0:
            y -= 1
        if action == 1:
            y += 1
        if action == 2:
            x -= 1
        if action == 3:
            x += 1

        y -= wind_value

        if x == 7 and y == 3:
            done = True
        else:
            done = False

        if x < 0 or y < 0 or x >= self.width or y >= self.height:
            x = max(0, min(self.width - 1, x))
            y = max(0, min(self.height - 1, y))

        self.state = self.state_from_point(x, y)

        return self.state, -1.0, done, False, dict()

    def point_from_state(self, s: int) -> Tuple[int, int]:
        return s % self.width, s // self.width

    def state_from_point(self, x: int, y: int) -> int:
        return y * self.width + x


class RandomWalkEnv(Env):
    def __init__(self, size: int):
        self.size = size
        self.position = 0
        self.observation_space = Discrete(2 * size + 1, start=-size)
        self.action_space = Discrete(2)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.position = 0
        return self.position, dict()

    def step(self, action):
        if action == 0:
            self.position -= 1
        if action == 1:
            self.position += 1

        reward = 0.0
        terminated = abs(self.position) >= self.size

        if terminated:
            reward = 0.0 if self.position < 0 else 1.0

        return self.position, reward, terminated, False, dict()


class MountainCarQuantization(QuantizationObservationTransformer):
    def __init__(self, env, bins: int = 25):
        super(MountainCarQuantization, self).__init__(env, bins)
