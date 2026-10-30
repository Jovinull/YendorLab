# Initial experiment protocol

All M0 runs use Brogue CE v1.15.1, Classic, Normal, WSL2, one worker, random movement agent v1, observation schema 1, action schema 1. A CLI run has one UUID and is stored in ignored SQLite. Episode seeds are sequential from `--seed`, policy seeds offset by 100000. Results are exploratory and do not form a held-out evaluation set.

| ID | Question | Bounded configuration | Status |
|---|---|---|---|
| EXP-0001 | Real control and fresh-process reset? | seeds 123–124, 2 episodes, 20 actions each, 20 s | Executed |
| EXP-0002 | Initial CPU throughput? | seeds 200–202, 3 episodes, 100 actions each, 30 s | Executed |
| EXP-0003 | Same seed/policy replay? | seed 123 twice in separate processes, 100 actions, 15 s each | Executed |
| EXP-0004 | Random plumbing baseline? | seeds 300–304, 5 episodes, 100 actions each, 30 s | Executed |
| EXP-0005 | Worker scaling? | 1 vs 2 workers | Deferred: per-step/menu semantics and natural termination need validation before worker comparison |

An episode is completed only on a native terminal event. The action cap/time cap yields truncation and cannot enter win/death denominators as a finished game. Measurements include process startup and Python pipe parsing. No confidence interval is calculated from zero completed games. See [results](../results/m0.md).
