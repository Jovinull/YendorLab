# Roadmap (updated 2026-11-03)

## Current state

**M5 complete.** Decision: exploration `CONDITIONAL_GO`, navigation `NO_GO`, full game `NO_GO`. See [M5 results](results/m5.md).

- Environment certified for the exploration scope: protocol 3, privileged perception audit, transport stress, soak, sanitizers.
- `exploration_v2` benchmark frozen, with controls, relay probe and separate seed domains. First 100 final seeds are spent.
- Deployed skill: linear behavior cloning on `rel_v3`, equal to the eight-move planner. It is a learned wrapper of a planner, not independent navigation.
- Learned line without planner: 583 cells on final seeds against 240 for random and 985 for the planner. More RL did not improve it within 400 episodes.
- Campaign infrastructure tested for crash, stop and resume. A long campaign is prepared and **not launched**; it needs approval.

## Next

1. Diagnose linear TD instability on validation seeds.
2. Define `navigation_v1`: reach known stairs, with its own controls.
3. Add HP and status to the observation and a survival signal.
4. Item and target contracts, then a separate Bullet Brogue environment.

Remaining game-contract gaps: throwing, device targeting, nested item prompts, status effects, a live victory. Full-game training stays out of scope until they close.

## Milestone history

- **M0 Foundation:** real headless input and restricted observation.
- **M1 Playable environment, partial:** protocol 2, context, terrain, inventory, canonical terminal callbacks.
- **M2 Baseline laboratory, partial:** bounded API, seed domains, rewards, heuristic, SQLite, CPU benchmarks.
- **M3 First learning pipeline:** real Q updates, checkpoints, resume. No demonstrated improvement.
- **M4 Narrow pilot:** two 30-episode linear policies improved on their zero-weight start. M5 showed that start was a fixed direction and that exposure was unequal: on a fair budget M4 policies are not distinguishable from random. See [M5 results](results/m5.md#2-audit-of-the-m4-benchmark).
- **M5 Hardening and bakeoff:** above.
