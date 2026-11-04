"""Privileged perception audit. Requires the YENDORLAB_TESTING fixture build.

The ground-truth stream exists only in that build and only on stderr. The agent
channel (stdout) is what is under test; production binaries contain neither the
scenario flag nor the truth writer (asserted by scripts/test-native-events.sh).
"""

import json
import os
import sys
from pathlib import Path

from yendorlab.agents import context_action
from yendorlab.env import BrogueEnv

BINARY = Path(".local/BrogueCE/bin/brogue")
FLAG = "--yendorlab-test-scenario"
SEEDS = range(1, 41)


def truth_lines(env):
    # The child shares this file offset, so a seek here would redirect its next write.
    descriptor = env.stderr_file.fileno()
    data = os.pread(descriptor, os.fstat(descriptor).st_size, 0)
    complete = data.split(b"\n")[:-1]  # The last line may be in flight.
    other = [line for line in complete if not line.startswith(b'{"decision"')]
    if other:
        print("non-truth stderr:", other[:3], file=sys.stderr)
    return {line["decision"]: line for line in map(json.loads, (line for line in complete
                                                                if line.startswith(b'{"decision"')))}


def check_stream(observations, truth, counters):
    """Every observation must equal the player-knowledge predicates of the truth."""
    # The truth line follows its observation, so the newest one may not be flushed yet.
    for obs in observations[:-1]:
        real = truth[obs["decision"]]
        known = {(cell[0], cell[1]) for cell in obs["cells"]}
        assert known == {tuple(cell) for cell in real["known"]}, "cell set differs from player knowledge"
        seen_monsters = sorted(tuple(m[:2]) for m in real["monsters"] if m[2])
        assert sorted((c[0], c[1]) for c in obs["creatures"]) == seen_monsters, "creature leak"
        seen_items = [item for item in real["items"] if item["seen"]]
        assert sorted((i["x"], i["y"]) for i in obs["floor_items"]) == \
            sorted((i["x"], i["y"]) for i in seen_items), "floor item leak"
        counters["unseen_monsters"] += sum(not m[2] for m in real["monsters"])
        counters["unseen_items"] += sum(not i["seen"] for i in real["items"])
        counters["observations"] += 1
        if real["hallucinating"] > 0:
            for item in obs["floor_items"]:
                twin = next(i for i in seen_items if (i["x"], i["y"]) == (item["x"], item["y"]))
                if twin["name"] in item["name"]:
                    counters["hallucination_name_matches"] += 1
                counters["hallucination_items"] += 1
        for x, y, solid in real["secrets"]:
            cell = next((c for c in obs["cells"] if c[0] == x and c[1] == y), None)
            if cell is not None:
                counters["secrets_known"] += 1
                assert cell[4] == solid, "secret cell passability differs from its appearance"


def use_item(env, name, observations):
    observation = env.observation
    letter = next(item["letter"] for item in observation["inventory"] if name in item["name"])
    observation = env.step("a")[0]
    observations.append(observation)
    assert observation["context"] == "inventory"
    observation = env.step(letter)[0]
    observations.append(observation)
    while observation["context"] != "gameplay":
        answer = "y" if observation["context"] == "confirmation" else context_action(observation)
        observation = env.step(answer)[0]
        observations.append(observation)
    return observation


def magic_flow(counters):
    for seed in (11, 12, 13):
        with BrogueEnv(BINARY, max_events=400, max_seconds=20, max_no_progress=400,
                       extra_args=(FLAG, "items")) as env:
            observation, _ = env.reset(seed=seed)
            stream = [observation]
            before = len(observation["cells"])
            mapped = use_item(env, "magic mapping", stream)
            assert len(mapped["cells"]) > before + 100, "magic mapping did not extend player knowledge"
            assert any(cell[5] == 1 for cell in mapped["cells"]), "mapped stairs missing from glyphs"
            counters["magic_map_gain"] += len(mapped["cells"]) - before
            telepathic = use_item(env, "telepathy", stream)
            counters["telepathy_creatures"] += len(telepathic["creatures"])
            use_item(env, "detect magic", stream)
            use_item(env, "darkness", stream)
            use_item(env, "hallucination", stream)
            for _ in range(30):  # Time for telepathy to lapse while hallucinating.
                observation = env.step("z" if observation["context"] == "gameplay"
                                       else context_action(observation))[0]
                stream.append(observation)
            check_stream(stream, truth_lines(env), counters)


def differential(scenario_a, scenario_b, counters):
    """Same seed, same visible world, different hidden truth: byte-equal stdout."""
    for seed in SEEDS:
        streams = []
        for scenario in (scenario_a, scenario_b):
            with BrogueEnv(BINARY, max_events=20, max_seconds=10, extra_args=(FLAG, scenario)) as env:
                observation, _ = env.reset(seed=seed)
                stream = [observation]
                stream.append(env.step("i")[0])
                stream.append(env.step("ESC")[0])
                truth = truth_lines(env)
                check_stream(stream, truth, counters)
                streams.append(stream)
        assert streams[0] == streams[1], f"hidden state changed the observation on seed {seed}"
        counters["differential_seeds"] += 1


def main():
    counters = dict.fromkeys(("observations", "unseen_monsters", "unseen_items", "secrets_known",
                              "hallucination_items", "hallucination_name_matches", "magic_map_gain",
                              "telepathy_creatures", "differential_seeds"), 0)
    magic_flow(counters)
    differential("none", "perturb_hidden", counters)
    differential("deep_none", "deep_perturb_hidden", counters)
    for name in ("unseen_monsters", "unseen_items", "secrets_known", "magic_map_gain",
                 "telepathy_creatures", "hallucination_items"):
        assert counters[name] > 0, f"vacuous audit: {name} never occurred"
    assert counters["hallucination_name_matches"] == 0, "hallucination exposed real item names"
    print("PERCEPTION_AUDIT_PASS", json.dumps(counters))


if __name__ == "__main__":
    sys.exit(main())
