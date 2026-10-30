# YendorLab

YendorLab is an open source, CPU focused research laboratory for agents that learn to play Brogue: Community Edition. The agent project is named **Maia**. This first milestone controls real Brogue CE games through a restricted, headless protocol. It does not yet train a policy or support the full game action space.

## Reproduce the current demonstration

On Windows 11 with WSL2 Ubuntu, from this repository root:

```bash
bash scripts/build-wsl.sh
PYTHONPATH=src python3 -m unittest discover -s tests -v
PYTHONPATH=src python3 -m yendorlab.cli --experiment EXP-0004 --episodes 2 --max-actions 50 --max-seconds 15 --seed 123
```

Run these commands **inside WSL** at `/mnt/c/Workspace/YendorLab`. The build pins Brogue CE `v1.15.1` to commit `1ba4240b7a928ddf0ffb772717bf1d433cd63804` and applies [one source patch](native/patches/0001-yendorlab-stream.patch). Build output and SQLite history are ignored under `.local/` and `data/`. The command prints JSON; `completed` counts only natural game endings. `Ctrl+C` interrupts the runner, which records only episodes actually started and marks incomplete ones as truncations.

The current random agent samples eight movement directions. It is a plumbing baseline, not Maia's learned policy. See [architecture](docs/architecture/initial-interface.md), [experiment report](docs/results/m0.md), and [roadmap](docs/roadmap.md).

## Project policy

The repository code and Brogue patch are offered under AGPL-3.0-or-later; see [LICENSE](LICENSE). Brogue CE authorship and upstream notices remain in the pinned upstream source. No Brogue CE or BrogueBot source is vendored here. Local raw data, replays, binaries, and checkpoints remain outside Git. Public research results require archived raw data and checksums before deletion.
