"""Optional Gymnasium adapter for the bounded movement exploration task."""

import gymnasium as gym
import numpy as np

from .agents import DIRECTIONS, context_action
from .env import BrogueEnv, RewardConfig
from .skill import LinearFrontierAgent


class GymExplorationEnv(gym.Env):
    metadata = {"render_modes": ["ansi"]}

    def __init__(self, binary, max_events=120, max_seconds=8,
                 max_no_progress=30, memory_mb=512, render_mode=None):
        super().__init__()
        if render_mode not in (None, "ansi"):
            raise ValueError("unsupported render mode")
        self.render_mode = render_mode
        self.engine = BrogueEnv(binary, max_events=max_events, max_seconds=max_seconds,
                                max_no_progress=max_no_progress, memory_mb=memory_mb,
                                reward=RewardConfig.named("exploration_v1"))
        self.encoder = LinearFrontierAgent()
        self.action_space = gym.spaces.Discrete(len(DIRECTIONS))
        self.observation_space = gym.spaces.Box(0.0, 1.0, (len(DIRECTIONS) * 16,), np.float32)
        self.current = None

    def _encode(self, observation):
        if observation["context"] != "gameplay":
            # Terminal observations may be on a postgame prompt. The final
            # gameplay observation remains the last policy state.
            return self.current.copy()
        features = self.encoder.features(observation)
        self.current = np.array([value for action in DIRECTIONS
                                 for value in features[action]], dtype=np.float32)
        return self.current.copy()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        game_seed = int(seed) if seed is not None else int(self.np_random.integers(1, 2**63))
        if game_seed < 1:
            game_seed = 1
        self.encoder = LinearFrontierAgent()
        observation, info = self.engine.reset(seed=game_seed, options=options)
        return self._encode(observation), info

    def step(self, action):
        if not self.action_space.contains(action):
            raise ValueError("action outside movement space")
        old_turn = self.engine.observation["turn"]
        old_events = self.engine.events
        observation, reward, terminated, truncated, info = self.engine.step(tuple(DIRECTIONS)[int(action)])
        while not (terminated or truncated) and observation["context"] != "gameplay":
            observation, additional, terminated, truncated, info = self.engine.step(context_action(observation))
            reward += additional
        # Resource samples depend on scheduling and are excluded from the
        # deterministic Gym info contract; raw BrogueEnv retains them.
        info = {key: info[key] for key in ("reason", "events", "turn", "depth",
                                           "max_depth", "new_cells")}
        info.update({"events_in_step": self.engine.events - old_events,
                     "turns_in_step": info["turn"] - old_turn})
        return self._encode(observation), reward, terminated, truncated, info

    def render(self):
        if self.render_mode != "ansi" or self.engine.observation is None:
            return None
        observation = self.engine.observation
        cells = {(c[0], c[1]): c for c in observation["cells"]}
        return "\n".join("".join("@" if (x, y) == (observation["x"], observation["y"])
                                 else "?" if (x, y) not in cells else
                                 "#" if cells[(x, y)][4] else "."
                                 for x in range(observation["width"]))
                         for y in range(observation["height"]))

    def close(self):
        self.engine.close()
