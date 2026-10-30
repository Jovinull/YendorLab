# Reproducibility (M0 history and current additions)

For the current CLI use Python 3.11+ in WSL. Run `bash scripts/build-wsl.sh`, `PYTHONPATH=src python3 -m unittest discover -s tests -v`, `bash scripts/test-native-events.sh`, then the bounded commands in [README](../README.md). The native source is fixed at `1ba4240b7a928ddf0ffb772717bf1d433cd63804`; protocol, observation and action schemas are 2, and the Q encoder is 1. The controlled native callback script temporarily builds a test binary and restores production afterward.

`configs/train-smoke.toml` provides explicit bounds. Seed derivation uses domain-separated SHA-256 of master seed, split and episode index for train, validation and final. Keep final seeds unused while tuning. Checkpoints store version, configuration, Q values, counters and validation history. Resume is at episode boundaries: an interrupted episode is discarded. SQLite stores run UUID, UTC, config, hardware, source revision when committed, native patch hash, status and result. Preserve raw artifacts externally with hashes before publishing a result.

`python3 scripts/compare-seed-catalog.py` strictly compares 6,773 catalog body lines after declared CRLF and old-header normalization. The original upstream comparator still fails on this Windows checkout; it is not reported as passing. Determinism evidence remains within the same build/platform.

For M4, see the [frozen exploration protocol](experiments/m4-exploration.md), [results and artifact hashes](results/m4.md), and [ADR 0003](decisions/0003-m4-skill.md). The two linear checkpoints have feature encoder version 2 and are incompatible with v1 tabular Q. The master seed and split generate game and policy seeds independently. Resume is at episode boundaries; each episode's policy RNG derives from its index, so a deterministic uninterrupted run and a resumed run produced identical weights in the short integration test. Exact reproduction beyond the same build and platform is unproven.

The optional [Gymnasium Env contract](https://gymnasium.farama.org/main/api/env/) was checked using Gymnasium 1.2.2 in an ignored isolated environment:

```bash
uv venv .local/gym-venv
uv pip install --python .local/gym-venv/bin/python gymnasium==1.2.2
PYTHONPATH=src .local/gym-venv/bin/python scripts/check-gym.py
```

The M4 experiment records made before the M4 commit reported the prior HEAD while the worktree contained uncommitted changes. Their checkpoint hashes and run IDs are retained, but that revision field alone does not identify exact source. New records include `worktree_dirty` and a source-tree SHA-256. Reproduce confirmatory results from a clean commit and a newly frozen seed set before publication.

The native source revision is pinned in `scripts/build-wsl.sh`; the only change is the tracked patch. Build under WSL, run `PYTHONPATH=src python3 -m unittest discover -s tests -v`, then the bounded CLI command in the README. SQLite `data/experiments.sqlite3` records run UUID, UTC time, config and results, including the native patch hash and per-child memory cap. It is ignored by Git. The runner derives game seeds as `--seed + episode index` and policy seeds as `--seed + 100000 + index`. It records per observation SHA-256 values but does not yet record the action sequence; deterministic replay is therefore limited to this versioned random policy.

For published results, record the YendorLab Git commit, native patch hash, compiler and WSL version, variant, mode, schemas, seeds, policy code/config, worker count and data checksums. Current results only establish same platform/same build repetition. Keep train, validation and final seeds separated via independent generated ranges and freeze final seeds before evaluation; this split is not implemented in the CLI yet. Do not tune on final seeds. Preserve raw evidence for any published claim outside the local retention area before deleting it.
