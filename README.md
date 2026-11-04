# YendorLab

YendorLab is an open source, CPU focused research laboratory for agents that learn to play Brogue: Community Edition. The agent is named **Maia**.

**Current state.** Maia can explore the first floors of Brogue on unseen seeds, moving only. Its deployed skill is a linear policy cloned from a hand-written planner, and it matches that planner. A separate learned policy without planner input beats random and direction controls by a wide margin and stays well below the planner. Navigation to a goal and the full game are not supported. See [M5 results](docs/results/m5.md) and the [roadmap](docs/roadmap.md).

## Setup

On Windows 11 with WSL2 Ubuntu and Python 3.11+, inside WSL at `/mnt/c/Workspace/YendorLab`:

```bash
bash scripts/build-wsl.sh
PYTHONPATH=src python3 -m unittest discover -s tests -v
bash scripts/test-native-events.sh
bash scripts/test-native-sanitizers.sh
```

The build pins Brogue CE `v1.15.1` at `1ba4240b7a928ddf0ffb772717bf1d433cd63804` and applies [one patch](native/patches/0001-yendorlab-stream.patch). Python needs only its standard library. For training and long evaluation, stage the binary on the Linux filesystem, under a path named by its hash:

```bash
L=$(bash scripts/stage-linux.sh)
```

## Using Maia

```bash
export PYTHONPATH=src
python3 -m yendorlab.app doctor
python3 scripts/m5-suite.py --binary "$L" --split diagnostic --count 64 --experiment EXP-LOCAL
python3 -m yendorlab.app experiments list
```

Selected policies, seeds and the frozen protocol are in the [M5 protocol](docs/experiments/m5-protocol.md). Checkpoints live in ignored `data/`; their hashes are in the protocol.

## Training

Campaigns are bounded, journaled and resumable. A smoke run takes seconds:

```bash
python3 -m yendorlab.app campaign start --config configs/m5/campaign-smoke.toml --directory data/smoke --binary "$L"
python3 -m yendorlab.app campaign status --directory data/smoke
```

`campaign stop` ends a run at the next episode boundary; `campaign resume` continues it and refuses a changed binary, source tree or configuration. The prepared long campaign and its commands are in [M5 results](docs/results/m5.md#14-prepared-campaign-not-launched). It has not been launched.

## Documentation

| Topic | File |
|---|---|
| Interface contract | [docs/architecture/initial-interface.md](docs/architecture/initial-interface.md) |
| Current benchmark protocol | [docs/experiments/m5-protocol.md](docs/experiments/m5-protocol.md) |
| Latest results and readiness | [docs/results/m5.md](docs/results/m5.md) |
| Decisions | [docs/decisions/](docs/decisions/) |
| Reproducing results | [docs/reproducibility.md](docs/reproducibility.md) |
| Scientific diary | [docs/research/diary.md](docs/research/diary.md) |

Earlier milestones are kept as written: [M0](docs/results/m0.md), [M1–M3](docs/results/m1-m3.md), [M4](docs/results/m4.md). The M4 commands still run, on the same seeds, for reproduction only. M5 explains which of its conclusions no longer hold.

CI runs the build, unit and integration tests, the perception audit, sanitizers and the seed catalog check. It does not run training and is not a scientific certification. WSL is needed only for local development; the Gymnasium check (`scripts/check-gym.py`) and the soak and benchmark scripts run manually.

## Project policy

The repository code and Brogue patch are offered under AGPL-3.0-or-later; see [LICENSE](LICENSE). Brogue CE authorship and upstream notices remain in the pinned upstream source. No Brogue CE or BrogueBot source is vendored here. Local raw data, replays, binaries and checkpoints remain outside Git. Public research results require archived raw data and checksums before deletion.
