# YendorLab

YendorLab is an open source, CPU focused research laboratory for agents that learn to play Brogue: Community Edition. The agent project is named **Maia**. A bounded tabular Q-learning pipeline now trains from real headless Brogue games. It supports an initial exploration task, not a full-game specialist.

## Reproduce the current demonstration

On Windows 11 with WSL2 Ubuntu, from this repository root:

```bash
bash scripts/build-wsl.sh
PYTHONPATH=src python3 -m unittest discover -s tests -v
PYTHONPATH=src python3 -m yendorlab.cli --experiment EXP-0004 --episodes 2 --max-actions 50 --max-seconds 15 --seed 123
```

Run these commands **inside WSL** at `/mnt/c/Workspace/YendorLab`. The build pins Brogue CE `v1.15.1` to commit `1ba4240b7a928ddf0ffb772717bf1d433cd63804` and applies [one source patch](native/patches/0001-yendorlab-stream.patch). Build output and SQLite history are ignored under `.local/` and `data/`. The command prints JSON; `completed` counts only natural game endings. `Ctrl+C` interrupts the runner, which records only episodes actually started and marks incomplete ones as truncations.

The M0 command above is retained for reproduction. The current operational CLI (Python 3.11+) is:

```bash
PYTHONPATH=src python3 -m yendorlab.app doctor
PYTHONPATH=src python3 -m yendorlab.app play --agent heuristic --seed 123
PYTHONPATH=src python3 -m yendorlab.app train --config configs/train-smoke.toml
PYTHONPATH=src python3 -m yendorlab.app evaluate --agent q --checkpoint data/checkpoints/maia-q-inference.json --episodes 3 --split validation --reward exploration_v1
PYTHONPATH=src python3 -m yendorlab.app benchmark --episodes 4 --max-events 150 --workers 1
PYTHONPATH=src python3 -m yendorlab.app experiments list
PYTHONPATH=src python3 -m yendorlab.app train --config configs/train-smoke.toml --episodes 6 --resume
```

Resume occurs only at an episode boundary; increase the episode target while keeping other settings fixed. Reserve `--split final` for a frozen protocol. Local SQLite history and checkpoints remain ignored in `data/`. See the [current contract](docs/architecture/initial-interface.md), [M1–M3 results](docs/results/m1-m3.md), and [roadmap](docs/roadmap.md).

## Project policy

The repository code and Brogue patch are offered under AGPL-3.0-or-later; see [LICENSE](LICENSE). Brogue CE authorship and upstream notices remain in the pinned upstream source. No Brogue CE or BrogueBot source is vendored here. Local raw data, replays, binaries, and checkpoints remain outside Git. Public research results require archived raw data and checksums before deletion.
