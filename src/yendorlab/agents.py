"""Simple, explicit policies for Brogue protocol v2."""

import random
from collections import deque


DIRECTIONS = {"h": (-1, 0), "j": (0, 1), "k": (0, -1), "l": (1, 0),
              "y": (-1, -1), "u": (1, -1), "b": (-1, 1), "n": (1, 1)}
GAME_ACTIONS = tuple(DIRECTIONS) + ("z",)


def context_action(observation, preferred_item=None):
    context = observation["context"]
    allowed = observation["admissible"]
    if context == "gameplay":
        raise ValueError("gameplay action belongs to the policy")
    if context in {"death_acknowledgment", "acknowledgment"}:
        return " "
    if context == "text_input":
        return "ESC"
    if context == "confirmation":
        return "N"
    if context == "target_selection":
        return "ESC"
    if context == "inventory":
        if preferred_item and preferred_item in allowed:
            return preferred_item
        return "ESC"
    raise ValueError(f"unknown context: {context}")


class RandomAgent:
    name = "uniform_random_v2"

    def __init__(self, seed: int):
        self.rng = random.Random(seed)

    def act(self, observation):
        if observation["context"] != "gameplay":
            return context_action(observation)
        return self.rng.choice(GAME_ACTIONS)


class HeuristicAgent:
    name = "frontier_v1"

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)
        self.visits = {}
        self.preferred_item = None
        self.last_position = None
        self.last_action = None
        self.last_turn = None
        self.failed_edges = set()

    def act(self, observation):
        if observation["context"] != "gameplay":
            answer = context_action(observation, self.preferred_item)
            if observation["context"] == "inventory":
                self.preferred_item = None
            return answer

        position = (observation["x"], observation["y"])
        if (self.last_position == position and self.last_action in DIRECTIONS
                and self.last_turn == observation["turn"]):
            self.failed_edges.add((position, self.last_action))
        self.visits[position] = self.visits.get(position, 0) + 1
        if observation["hp"] < observation["max_hp"] // 2:
            for item in observation["inventory"]:
                if "healing" in item["name"].lower() or "potion of life" in item["name"].lower():
                    self.preferred_item = item["letter"]
                    return "a"

        cells = {(cell[0], cell[1]): cell for cell in observation["cells"]}
        for action, (dx, dy) in DIRECTIONS.items():
            nxt = (position[0] + dx, position[1] + dy)
            if (0 <= nxt[0] < observation["width"] and 0 <= nxt[1] < observation["height"]
                    and nxt not in cells and (position, action) not in self.failed_edges):
                self.last_position, self.last_action, self.last_turn = position, action, observation["turn"]
                return action
        downstairs = [point for point, cell in cells.items() if cell[5] == 1]
        if position in downstairs:
            return ">"

        # Search known traversable cells for a route toward a frontier or staircase.
        queue = deque([(position, None)])
        seen = {position}
        fallback = None
        while queue:
            point, first = queue.popleft()
            cell = cells.get(point)
            if point != position and cell and cell[5] == 1:
                return first
            frontier = any((point[0] + dx, point[1] + dy) not in cells
                           for dx, dy in DIRECTIONS.values())
            if point != position and frontier and fallback is None:
                fallback = first
            for action, (dx, dy) in DIRECTIONS.items():
                nxt = (point[0] + dx, point[1] + dy)
                known = cells.get(nxt)
                if nxt in seen or not known or known[4] or (point, action) in self.failed_edges:
                    continue
                # Avoid a displayed creature unless an attack is unavoidable.
                if any(creature[0] == nxt[0] and creature[1] == nxt[1]
                       for creature in observation["creatures"]):
                    continue
                seen.add(nxt)
                queue.append((nxt, first or action))
        if fallback:
            self.last_position, self.last_action, self.last_turn = position, fallback, observation["turn"]
            return fallback
        choices = [(self.visits.get((position[0] + dx, position[1] + dy), 0), action)
                   for action, (dx, dy) in DIRECTIONS.items()
                   if not cells.get((position[0] + dx, position[1] + dy), [0, 0, 0, 0, 0])[4]]
        action = min(choices)[1] if choices else "z"
        self.last_position, self.last_action, self.last_turn = position, action, observation["turn"]
        return action


def encode_state(observation):
    """Compact local state v1; discards global map and inventory information."""
    x, y = observation["x"], observation["y"]
    cells = {(cell[0], cell[1]): cell for cell in observation["cells"]}
    neighbors = []
    for dx, dy in DIRECTIONS.values():
        cell = cells.get((x + dx, y + dy))
        neighbors.append("U" if cell is None else "B" if cell[4] else "O")
    hp = observation["hp"] / max(1, observation["max_hp"])
    hp_bin = 0 if hp < .34 else 1 if hp < .67 else 2
    nearby_enemy = int(any(abs(c[0] - x) <= 2 and abs(c[1] - y) <= 2
                           for c in observation["creatures"]))
    return f"{hp_bin}|{nearby_enemy}|{''.join(neighbors)}"


class QAgent:
    name = "tabular_q_v1"

    def __init__(self, q=None, seed: int = 0, epsilon: float = 0.0):
        self.q = q if q is not None else {}
        self.rng = random.Random(seed)
        self.epsilon = epsilon

    def act(self, observation):
        if observation["context"] != "gameplay":
            return context_action(observation)
        allowed = [action for action in GAME_ACTIONS if action in observation["admissible"]]
        state = encode_state(observation)
        values = self.q.get(state, [0.0] * len(GAME_ACTIONS))
        if self.rng.random() < self.epsilon:
            return self.rng.choice(allowed)
        return max(allowed, key=lambda action: (values[GAME_ACTIONS.index(action)],
                                                -GAME_ACTIONS.index(action)))

    def update(self, state, action, reward, next_state, terminal, alpha=.2, gamma=.95):
        values = self.q.setdefault(state, [0.0] * len(GAME_ACTIONS))
        index = GAME_ACTIONS.index(action)
        future = 0.0 if terminal else max(self.q.get(next_state, [0.0] * len(GAME_ACTIONS)))
        old = values[index]
        values[index] += alpha * (reward + gamma * future - old)
        return values[index] != old
