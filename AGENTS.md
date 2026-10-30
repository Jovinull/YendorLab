# YendorLab operating guide

1. Read [README.md](README.md), [current architecture](docs/architecture/initial-interface.md), [roadmap](docs/roadmap.md), [M0 evidence](docs/results/m0.md) and [M1–M3 evidence](docs/results/m1-m3.md) before changing the integration.
2. Build inside WSL with `bash scripts/build-wsl.sh`; run `PYTHONPATH=src python3 -m unittest discover -s tests -v` and `bash scripts/test-native-events.sh`. Never claim an unrun test passed.
3. Keep upstream pinned. Change the tracked patch and update its hash/revision in documentation. Preserve game rules and observe only player available information.
4. Record every experiment ID, config, seeds, started/completed/truncated counts, hardware, code revision, results, and limits. Keep raw SQLite and action traces in ignored `data/`; commit concise protocols and results.
5. Write an ADR for consequential architecture changes. Update the scientific diary and roadmap after a milestone. State inconclusive results plainly.
6. Limit CPU and wall time; do not launch open ended training. Keep train, validation and final seed domains separate. Test leakage, replay, termination, and resets before expanding actions or learning.
7. Final reports distinguish implementation, measurement, inference, and unresolved work. Include exact commands and Git status.
