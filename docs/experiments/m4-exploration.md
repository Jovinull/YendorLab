# EXP-0018–0021: known-map exploration v1

Protocol frozen before final-seed evaluation on 2026-10-30. Scope: Brogue CE Classic/Normal, official rules, no revealed map. This is a short-horizon skill experiment, not a full-game benchmark.

## Task choice

Exploration has immediate objective feedback (new cells) and can be sampled in 120 input events. Stair navigation is more strategically relevant but requires finding a stair first and longer episodes; survival can reward resting without progress. Thus exploration is the first controlled task. A move can fail, attack, or consume a turn according to the real engine. No special map or seed is used.

## Question and design

Hypothesis: a CPU linear action-value agent trained on player-visible observations discovers more new cells on unseen seeds than its zero-weight initial policy. Primary metric: new distinct `(depth,x,y)` cells discovered after reset, averaged over episodes, with the same maximum of 120 input events. Secondary: deaths, turns, events, truncations, shaped reward, and comparison to random and the hand-coded heuristic. The primary metric is counted from the observation stream, independent of reward values. Event limits and early death can affect exposure; report outcome counts and events alongside cell counts.

Two independent 30-episode training runs use master seeds 20261031 and 20261032. SHA-256 domain-separated seeds generate train, validation and final groups. Validation uses five episodes every ten training episodes for diagnosis only. Final assessment uses 16 seeds from the `final` domain of master 20261031, shared by all evaluated policies; no tuning after looking at this set. Policies are initial zero-weight linear, learned linear from each run, random, and existing heuristic. Each agent receives its own deterministic policy RNG derived from the same master/index. One worker; each episode at most 120 events, 8 seconds, 30 no-progress callbacks, 512 MiB child address-space cap. Per training run: 180-second wall cap. Stop if protocol failures or hidden-state leakage are found; do not increase the budget to rescue a negative result.

Primary comparison: paired differences in new-cell counts per seed between each trained model and the initial linear policy. Report mean difference and a deterministic percentile paired bootstrap interval (10,000 resamples), descriptive at n=16; do not claim strong statistical significance from this pilot. A provisional success requires both independently trained models to have positive paired mean differences and no evidence of leaked state. `LEARNING_DEMONSTRATED` requires a credible uncertainty interval above zero and behavior change, with limitations stated. Random and heuristic are context, not optimization targets. If a model fails, inspect action-feature weights, state collisions, events/turns and outcome distribution before considering more episodes.

## Engineering semantics

The environment retains event-level steps. Menu inputs remain visible to the controller and do not silently consume strategic decisions. The skill policy chooses gameplay movement; context inputs follow a fixed safe router. Its feature version 2 uses only known map cells, visible creatures, a BFS direction to a frontier, and within-episode visit memory. The eight action scores share 16 linear weights. The task reward is `10 × exploration_v1.exploration + terminal`; death is -1, and initial visible cells do not earn reward. This reward trains the policy but is not the primary metric. On valid action/no-progress truncation, TD bootstrap uses the returned gameplay observation; on natural terminal there is no bootstrap; on timeout/protocol/engine failure the final transition is not updated because the post-action state is unverified.

## Threats

Sixteen seeds give limited precision. The CLI's event budget includes context inputs, not just gameplay moves. Unknown-cell movement may fail due to unseen terrain, as it should. The heuristic and linear policy may have different early-death rates. The feature map is based on the protocol v2 perception audit, which is incomplete for special effects. Repeated validation seeds can overfit checkpoint selection; this pilot makes no hyperparameter selection from final seeds. Full-game and cross-platform generalization are outside scope.
