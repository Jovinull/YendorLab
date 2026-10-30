"""Brogue CE protocol v2 environment. One step equals one native input event."""

import json
import os
import select
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

try:
    import resource
except ImportError:
    resource = None


RESULTS = {"death", "quit", "victory", "supervictory"}


@dataclass(frozen=True)
class RewardConfig:
    name: str = "canonical"
    win: float = 1.0
    death: float = -1.0
    new_cell: float = 0.0
    new_depth: float = 0.0

    @classmethod
    def named(cls, name: str) -> "RewardConfig":
        if name == "canonical":
            return cls()
        if name == "exploration_v1":
            return cls(name, new_cell=0.01, new_depth=1.0)
        raise ValueError(f"unknown reward scheme: {name}")


class BrogueEnv:
    def __init__(self, binary: Path, variant: str = "classic", max_events: int = 300,
                 max_seconds: float = 30, max_no_progress: int = 50, memory_mb: int = 512,
                 reward: RewardConfig | None = None):
        if variant not in {"classic", "rapid", "bullet"}:
            raise ValueError("unsupported variant")
        if min(max_events, max_no_progress, memory_mb) <= 0 or max_seconds <= 0:
            raise ValueError("limits must be positive")
        self.binary = binary.resolve()
        self.variant = variant
        self.max_events = max_events
        self.max_seconds = max_seconds
        self.max_no_progress = max_no_progress
        self.memory_mb = memory_mb
        self.reward = reward or RewardConfig()
        self.process = None
        self.observation = None
        self.events = 0
        self.deadline = 0.0
        self.no_progress = 0
        self.known = set()
        self.max_depth = 0
        self.terminal = None
        self.peak_native_rss_kib = 0
        self.components = {"terminal": 0.0, "exploration": 0.0, "depth": 0.0}

    def _read(self):
        while time.monotonic() < self.deadline:
            ready, _, _ = select.select([self.process.stdout], [], [],
                                        min(0.25, max(0, self.deadline - time.monotonic())))
            if ready:
                line = self.process.stdout.readline()
                if not line:
                    return {"event": "engine_error", "reason": f"exit_{self.process.poll()}"}
                try:
                    return json.loads(line)
                except json.JSONDecodeError as exc:
                    return {"event": "protocol_error", "reason": str(exc)}
            if self.process.poll() is not None:
                return {"event": "engine_error", "reason": f"exit_{self.process.returncode}"}
        return {"event": "time_limit"}

    def _validate(self, message):
        if message.get("protocol") != 2 or message.get("observation_schema") != 2:
            raise ValueError("protocol or observation schema mismatch")
        if message.get("action_schema") != 2 or message.get("decision") != self.events + 1:
            raise ValueError("action schema or decision sequence mismatch")
        if not isinstance(message.get("cells"), list) or not isinstance(message.get("inventory"), list):
            raise ValueError("invalid observation arrays")
        if message.get("context") not in {"gameplay", "inventory", "confirmation",
                                           "acknowledgment", "death_acknowledgment",
                                           "text_input", "target_selection"}:
            raise ValueError("unknown input context")

    def reset(self, seed: int | None = None, options: dict | None = None):
        self.close()
        if seed is None or seed < 1:
            raise ValueError("explicit positive seed required")
        if resource is None:
            raise RuntimeError("native execution requires Linux/WSL")
        if not self.binary.is_file():
            raise FileNotFoundError(self.binary)
        command = [str(self.binary), "--yendorlab", "-n", "-s", str(seed)]
        if self.variant != "classic":
            command.extend(["--variant", f"{self.variant}_brogue"])
        memory_bytes = self.memory_mb * 1024 * 1024

        def limit_memory():
            resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))

        self.process = subprocess.Popen(command, cwd=self.binary.parent,
                                        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE, text=True, bufsize=1,
                                        preexec_fn=limit_memory)
        self.deadline = time.monotonic() + self.max_seconds
        self.events = 0
        self.no_progress = 0
        self.known = set()
        self.max_depth = 0
        self.terminal = None
        self.peak_native_rss_kib = 0
        self.components = {"terminal": 0.0, "exploration": 0.0, "depth": 0.0}
        message = self._read()
        self._validate(message)
        self.observation = message
        self._sample_rss()
        self._track(message)
        # Initial knowledge/depth are a baseline, not a step reward.
        self.components = {"terminal": 0.0, "exploration": 0.0, "depth": 0.0}
        return message, {"seed": seed, "variant": self.variant, "reward_scheme": self.reward.name}

    def _track(self, observation):
        depth = observation["depth"]
        newly_known = {(depth, cell[0], cell[1]) for cell in observation["cells"]} - self.known
        self.known.update(newly_known)
        depth_gain = max(0, depth - self.max_depth)
        self.max_depth = max(depth, self.max_depth)
        components = {"terminal": 0.0,
                      "exploration": len(newly_known) * self.reward.new_cell,
                      "depth": depth_gain * self.reward.new_depth}
        for key, value in components.items():
            self.components[key] += value
        return components

    def _sample_rss(self):
        if self.process is None:
            return
        try:
            status = Path(f"/proc/{self.process.pid}/status").read_text()
            line = next(line for line in status.splitlines() if line.startswith("VmRSS:"))
            self.peak_native_rss_kib = max(self.peak_native_rss_kib, int(line.split()[1]))
        except (OSError, StopIteration, ValueError):
            pass

    def step(self, action: str):
        if self.process is None or self.observation is None or self.terminal:
            raise RuntimeError("reset required or episode already ended")
        if action not in self.observation["admissible"]:
            raise ValueError(f"action {action!r} is not admissible in {self.observation['context']}")
        previous = self.observation
        try:
            self.process.stdin.write(action + "\n")
            self.process.stdin.flush()
        except OSError:
            message = {"event": "engine_error", "reason": "broken_pipe"}
        else:
            message = self._read()
        self.events += 1
        if self.events % 16 == 0:
            self._sample_rss()
        reward_parts = {"terminal": 0.0, "exploration": 0.0, "depth": 0.0}
        terminated = truncated = False
        reason = None
        if message.get("event") == "terminal":
            if message.get("result") not in RESULTS:
                reason = "PROTOCOL_ERROR"
                truncated = True
            else:
                reason = message["result"].upper()
                terminated = True
                reward_parts["terminal"] = (self.reward.win if reason in {"VICTORY", "SUPERVICTORY"}
                                             else self.reward.death if reason == "DEATH" else 0.0)
                self.components["terminal"] += reward_parts["terminal"]
            self.terminal = message
        elif message.get("event"):
            reason = "TRUNCATED_TIME_LIMIT" if message["event"] == "time_limit" else (
                "PROTOCOL_ERROR" if message["event"] == "protocol_error" else "ENGINE_ERROR")
            truncated = True
            self.terminal = message
        else:
            try:
                self._validate(message)
            except ValueError as exc:
                reason = "PROTOCOL_ERROR"
                truncated = True
                self.terminal = {"event": "protocol_error", "reason": str(exc)}
            else:
                reward_parts = self._track(message)
                progress = (message["turn"] > previous["turn"] or message["depth"] > previous["depth"]
                            or message["x"] != previous["x"] or message["y"] != previous["y"]
                            or message["context"] != previous["context"]
                            or reward_parts["exploration"] > 0)
                self.no_progress = 0 if progress else self.no_progress + 1
                self.observation = message
                if self.no_progress >= self.max_no_progress:
                    reason = "TRUNCATED_NO_PROGRESS"
                    truncated = True
                elif self.events >= self.max_events:
                    reason = "TRUNCATED_ACTION_LIMIT"
                    truncated = True
                if truncated:
                    self.terminal = {"event": "limit", "result": reason}
        if terminated or truncated:
            self.close()
        info = {"reason": reason, "events": self.events, "turn": self.observation["turn"],
                "depth": self.observation["depth"], "max_depth": self.max_depth,
                "reward_components": reward_parts, "reward_totals": dict(self.components),
                "terminal": self.terminal, "native_rss_peak_sampled_kib": self.peak_native_rss_kib}
        return self.observation, sum(reward_parts.values()), terminated, truncated, info

    def close(self):
        if self.process is not None:
            if self.process.poll() is None:
                self.process.terminate()
            try:
                self.process.communicate(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.communicate()
            self.process = None

    def render(self):
        return self.observation

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
