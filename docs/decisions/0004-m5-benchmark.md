# ADR 0004: Comparable exploration task, private episodes and imitation first

Status: accepted, 2026-11-03. Amends [ADR 0003](0003-m4-skill.md). Evidence: [M5 results](../results/m5.md).

**Task contract.** M4 compared policies with different actions and unequal exposure, because a no-progress limit ended stuck policies early and a heuristic counted stairs across depths. `exploration_v2_floor` fixes eight moves, one depth and a decision budget, with no early exit for stalling. Decisions, not turns, because a wall bump is a real mistake with a real cost and costs no game turn. A turn budget would make a stalling policy free. The relay task exists because the depth 1 start is nearly fixed at the bottom of the map, so depth 1 alone cannot separate spatial skill from orientation.

**Shared perception.** Every comparable policy receives the same per-episode memory. Without it, a stalled deterministic policy repeats forever, and comparing it to one with memory measures bookkeeping, not skill.

**Private episode directory.** Upstream writes recordings and scores to its cwd. The alternative, patching file handling in upstream code, widens the patch for no gain. A tmpfs directory per episode makes episodes independent and removes the dominant reset cost.

**JSON stays.** Delta encoding might save a third of per-decision cost, by estimate. At 600–2,000 decisions per second it would add a stateful protocol and new failure modes for a gain nothing currently needs. Revisit if a campaign is limited by simulation throughput.

**Recording flush.** A flush at every input hung the child: a declined prompt calls `cancelKeystroke`, which assumes the key is still buffered. Flushing only in gameplay context, between commands, keeps that invariant. The patch touches no upstream file for this.

**Encoders.** Absolute direction indicators added nothing on top of relative features, so `rel_v3` omits them. Keeping `ego_patch_v1` as a no-planner, rotation-equivariant reference costs little, though it lost to `rel_v3` without the planner.

**Architecture.** Behavior cloning of the eight-move planner on `rel_v3` is the deployed exploration skill: cheapest, best, and it matches the planner. RL refinement degraded it. Linear stays: a small MLP gave no online gain where the linear model looked underfit, and nothing here justifies a deep learning dependency, PPO, DQN or evolution. The learned no-planner line, BC then TD, is the only candidate for further training, because it is the only arm where RL added value and where success would not restate a planner.

**Consequences.** A high exploration score with `on_route` must not be reported as learned navigation. M4 results stay as historical records, but their claim of competence above random does not survive a fair budget. Any change to task, budget or encoder needs a new protocol version and a new final seed range.
