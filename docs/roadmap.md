# Roadmap (updated 2026-10-30)

- **M4 narrow learning demonstrated:** two independent 30-episode linear policies improved raw discovered-cell counts over their zero-weight initial policy on 16 held-out seeds, with descriptive paired intervals above zero. A fixed-north control explains some, but not all, of the gain; the manual heuristic remains much stronger. See [M4 results](results/m4.md).
- **Next gate:** replicate with a clean committed revision and larger preregistered seed set, then add robust contextual item/target actions and perception audits. Profile startup from Linux filesystem and decide whether to stage the native binary there. Do not infer Bullet/Rapid/Classic campaign readiness from this task.

- **M1 partial:** protocol v2, context, terrain, inventory, canonical terminal callbacks, real death and quit tests. Live victory and full contextual action/visibility audit remain.
- **M2 partial:** bounded environment API, seed domains, reward schemes, evaluation, heuristic baseline, SQLite and CPU benchmarks. Gymnasium validator was not run; baseline quality is limited.
- **M3 initial training pipeline operational:** real Brogue Q updates, atomic checkpoints, inference loading, episode-boundary resume and held-out evaluation. No demonstrated game improvement. See [M1–M3 results](results/m1-m3.md).
- **Next:** audit target and item workflows and special perception, find a valid victory recording or live route, improve heuristic, then compare learned policies under a frozen protocol. Full-game training and specialist status remain future work.

## Historical M0 roadmap

- **M0 Foundation: partial.** Real headless input and restricted observation work; build and bounded runner work. Official upstream regression tests, full terminal coverage, formal leak tests, and benchmark comparison remain open.
- **M1 Playable environment:** design menu/context-aware actions, inventory and terrain observation with visibility audit, natural end detection tests, reset contract, and Gymnasium-compatible wrapper if measured useful.
- **M2 Baseline laboratory:** legal random and interpretable navigation agents, held-out seed protocol, robust metrics, latency and memory benchmarks, paired comparisons.
- **M3 First learning agent:** select a compact task and compare tabular/value, evolutionary and policy methods under equal CPU budgets. Do not commit to PPO before evidence.
- Later M4–M7: curriculum across Bullet/Rapid/Classic, stronger agents, public benchmark, scientific results, subject to measured progress.
