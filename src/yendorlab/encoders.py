"""Versioned per-action feature encoders over the player-visible map view.

Every encoder returns, for each of the eight moves, a sparse list of (index, value).
Weights are shared across moves, so anything that distinguishes one absolute
direction from another must be stated by an explicit feature.

rel_v3 and ego_patch_v1 contain no absolute direction. Their score for a move is
unchanged when the map and the move are rotated together, which the tests assert.
"""

from .skill import frontier_action
from .tasks import MOVES, OFFSETS, route

REL_NAMES = ("bias", "diagonal", "known_blocked", "failed_edge", "unknown_target", "enemy",
             "visits", "reverse", "target_visible", "stairs", "on_route", "local_gain",
             "momentum", "unvisited_open", "enclosure")
REL_GROUPS = {"blocked": ("known_blocked",), "failed": ("failed_edge",),
              "visits": ("visits", "reverse", "momentum", "unvisited_open"),
              "frontier": ("on_route",), "local": ("local_gain", "enclosure", "unknown_target"),
              "stairs": ("stairs",), "enemy": ("enemy",), "visible": ("target_visible",)}
RADIUS_TWO = tuple((dx, dy) for dx in range(-2, 3) for dy in range(-2, 3) if dx or dy)
ABS_NAMES = ("bias", "unknown", "blocked", "enemy", "visits", "route", "reverse", "visible")


def _rel(view, directions=False, planner=True):
    mask = route(view)[0] if planner else 0
    cells, visits, failed = view.cells, view.visits, view.failed
    x, y = view.x, view.y
    result = {}
    for index, (action, (dx, dy)) in enumerate(zip(MOVES, OFFSETS)):
        nx, ny = x + dx, y + dy
        cell = cells.get((nx, ny))
        blocked = view.move_blocked(x, y, index)
        features = [(0, 1.0)]
        if dx and dy:
            features.append((1, 1.0))
        if blocked:
            features.append((2, 1.0))
        if (x, y, index) in failed:
            features.append((3, 1.0))
        if cell is None and view.inside(nx, ny):
            features.append((4, 1.0))
        if (nx, ny) in view.enemies:
            features.append((5, 1.0))
        count = visits.get((nx, ny), 0)
        if count:
            features.append((6, min(count, 5) / 5.0))
        if (nx, ny) == view.previous:
            features.append((7, 1.0))
        if cell is not None and cell[2]:
            features.append((8, 1.0))
        if cell is not None and cell[5]:
            features.append((9, 1.0))
        if mask >> index & 1:
            features.append((10, 1.0))
        unknown = walls = 0
        for ox, oy in RADIUS_TWO:
            near = cells.get((nx + ox, ny + oy))
            if near is None:
                unknown += view.inside(nx + ox, ny + oy)
            elif near[4] and abs(ox) <= 1 and abs(oy) <= 1:
                walls += 1
        if unknown:
            features.append((11, unknown / 24.0))
        if index == view.last_action:
            features.append((12, 1.0))
        if cell is not None and not cell[4] and not count and not blocked:
            features.append((13, 1.0))
        if walls:
            features.append((14, walls / 8.0))
        if directions:
            features.append((15 + index, 1.0))
        result[action] = features
    return result


def _abs(view, directions=True):
    """The M4 features computed on this view; `route` keeps M4 direction-order ties."""
    x, y = view.x, view.y
    observation = {"x": x, "y": y, "width": view.width, "height": view.height,
                   "cells": list(view.cells.values())}
    planned = frontier_action(observation)
    result = {}
    for index, (action, (dx, dy)) in enumerate(zip(MOVES, OFFSETS)):
        target = (x + dx, y + dy)
        cell = view.cells.get(target)
        inside = view.inside(*target)
        vector = [1.0, float(inside and cell is None),
                  float(not inside or (cell is not None and bool(cell[4]))),
                  float(target in view.enemies), min(view.visits.get(target, 0), 5) / 5.0,
                  float(action == planned), float(target == view.previous),
                  float(cell is not None and bool(cell[2]))]
        features = [(i, value) for i, value in enumerate(vector) if value]
        if directions:
            features.append((8 + index, 1.0))
        result[action] = features
    return result


# Rotations of a (forward, left) frame; each move class has one canonical frame.
_ROTATIONS = {"k": 0, "l": 1, "j": 2, "h": 3, "u": 0, "n": 1, "b": 2, "y": 3}
PATCH_RADIUS = 3
_PATCH_OFFSETS = tuple((u, v) for u in range(-PATCH_RADIUS, PATCH_RADIUS + 1)
                       for v in range(-PATCH_RADIUS, PATCH_RADIUS + 1) if u or v)
PATCH_CHANNELS = 3
PATCH_CLASS_SIZE = len(_PATCH_OFFSETS) * PATCH_CHANNELS + 2
PATCH_DIMENSION = PATCH_CLASS_SIZE * 2


def _rotate(u, v, turns):
    for _ in range(turns):
        u, v = -v, u
    return u, v


_PATCH_FRAMES = {action: tuple(_rotate(u, v, _ROTATIONS[action]) for u, v in _PATCH_OFFSETS)
                 for action in MOVES}


def _patch(view):
    """Egocentric map patch in the frame of each move, with no planner input."""
    cells, visits = view.cells, view.visits
    x, y = view.x, view.y
    result = {}
    for index, (action, (dx, dy)) in enumerate(zip(MOVES, OFFSETS)):
        base = PATCH_CLASS_SIZE if dx and dy else 0
        features = [(base, 1.0)]
        if (x, y, index) in view.failed:
            features.append((base + 1, 1.0))
        for position, (u, v) in enumerate(_PATCH_FRAMES[action]):
            point = (x + u, y + v)
            cell = cells.get(point)
            offset = base + 2 + position * PATCH_CHANNELS
            if cell is None:
                features.append((offset, 1.0))
            elif cell[4]:
                features.append((offset + 1, 1.0))
            elif point in visits:
                features.append((offset + 2, 1.0))
        result[action] = features
    return result


class Encoder:
    def __init__(self, name, disabled=()):
        self.name = name
        self.disabled = tuple(sorted(disabled))
        if name in {"rel_v3", "rel_v3_dir"}:
            names = REL_NAMES + (tuple(f"dir_{m}" for m in MOVES) if name == "rel_v3_dir" else ())
            planner = "frontier" not in disabled
            self._encode = (lambda view: _rel(view, name == "rel_v3_dir", planner))
        elif name in {"abs_v2", "abs_v2_nodir"}:
            names = ABS_NAMES + (tuple(f"dir_{m}" for m in MOVES) if name == "abs_v2" else ())
            self._encode = (lambda view: _abs(view, name == "abs_v2"))
        elif name == "ego_patch_v1":
            names = tuple(f"patch_{i}" for i in range(PATCH_DIMENSION))
            self._encode = _patch
        else:
            raise ValueError(f"unknown encoder: {name}")
        self.names = names
        self.dimension = len(names)
        removed = set()
        for group in self.disabled:
            if group not in REL_GROUPS or not name.startswith("rel_v3"):
                raise ValueError(f"unknown feature group: {group}")
            removed.update(names.index(feature) for feature in REL_GROUPS[group])
        self.removed = frozenset(removed)

    def __call__(self, view):
        key = ("encoded", self.name, self.disabled)
        encoded = view.cache.get(key)
        if encoded is None:
            encoded = self._encode(view)
            if self.removed:
                encoded = {action: [pair for pair in features if pair[0] not in self.removed]
                           for action, features in encoded.items()}
            view.cache[key] = encoded
        return encoded

    def spec(self):
        return {"name": self.name, "disabled": list(self.disabled)}
