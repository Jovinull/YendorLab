# ADR 0002: Versioned event protocol and first CPU learner

Status: provisional, 2026-10-30. Extends [ADR 0001](0001-integration.md).

The M0 adapter treated each native callback as movement. Real death could hang on subsequent prompts; inventory and confirmation keys were ambiguous. We added narrow context markers at Brogue input sites, used `notifyEvent` for canonical terminal outcomes, and kept a fresh child process per episode. The Python wrapper treats one input callback as one environment step, and records the game's separate turn counter. This preserves Brogue's rules and makes menus explicit. A full high-level action API was deferred until target and item flows are audited.

For the first learning proof, tabular Q-learning with compact local features was chosen. It has no heavy dependency, makes parameter updates inspectable and fits short CPU smoke experiments. Evolutionary search would require many full episodes per parameter comparison; PPO adds a neural framework and larger rollout budget; deterministic planning alone does not learn. A hybrid planner plus learned policy remains a research option. Tabular Q is unlikely to scale to full Brogue because its state discards most map and inventory detail. It is used to validate the pipeline, not selected as the final architecture.

Gymnasium-style `reset/step/close` return semantics are implemented without a Gymnasium dependency. The actual Gymnasium validator has not been run, and its formal space contract is not claimed. SQLite plus JSON checkpoints remain sufficient for these bounded experiments. Revisit after full-game observations, targeting and action semantics are established.
