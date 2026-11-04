# M5 protocol: `exploration_v2` and the architecture bakeoff

Frozen in stages. Git history of this file is the record of what was fixed before each result existed. Deviations are appended under [Protocol changes](#protocol-changes), never edited in place. Results live in [M5 results](../results/m5.md).

## Questions

1. **Integrity.** Is the environment trustworthy for a long exploration campaign: perception, protocol, rewards, replay, failures?
2. **Comparability.** Was the M4 comparison between Maia and the heuristic an equal one? What is the source of the M4 northward preference?
3. **Architecture.** Which CPU approach yields the best legitimate exploration policy per unit of compute: improved linear TD, behavior cloning of a planner, or imitation followed by RL?
4. **Components.** Which of direction indicators, visit memory, frontier information, shaping and the encoder actually contribute?
5. **Readiness.** Is a long campaign justified, separately for exploration, navigation and the full game?

## Task definitions

All tasks use Classic, Normal, one fresh process per episode, and the per-episode private directory of ADR 0004.

**`exploration_v2_floor`** is the primary, strictly comparable task.

- *Actions.* The eight moves. A move can walk, attack, open a door, stall, or raise a prompt; the safe router declines prompts. A policy that returns anything else ends the episode as `POLICY_ERROR`.
- *Observation.* Protocol v3 fields plus the shared `Perception` memory (visits, failed edges, previous position, last move). Policies differ only in how they use this.
- *Budget.* A fixed count of gameplay decisions, chosen under [Budget rule](#budget-rule). Prompts cost events, not decisions; events and turns are reported separately. A wall bump costs a decision and no turn, because it is a real policy mistake and a real compute cost.
- *End.* `TASK_BUDGET` at the budget. `TASK_DEPTH_EXIT` when depth changes, by stairs or by falling; the cells of the start floor already found still count and no further ones can be added. Natural `DEATH` likewise keeps what was found. There is no no-progress truncation: a stuck policy spends its whole budget, so exposure is equal and stalling is measured.
- *Primary metric.* `new_cells`: distinct cells on the start depth present in the observation stream at the end and absent at reset. It reads no reward value.
- *Why deaths stay in the mean.* Dying early is a failure of the policy. An episode that ends by death contributes its actual count.

**`exploration_v2_multilevel`** allows depth change and counts distinct `(depth, x, y)`, with the same eight moves. It is secondary: every agent has the actions to take part, but a first-floor policy need not.

**`exploration_v2_relay`** is the generalization probe. A fixed guide, the same for every compared policy, walks to depth 2 with its own decision budget; the policy then starts there with a fresh perception, baseline and budget. Depth 2 arrival positions vary across seeds, unlike the near-constant depth 1 start, so a direction preference has no structural support here. Seeds where the guide cannot reach depth 2 have no task instance and are reported as `unavailable`, identically for all policies.

**`exploration_v2_full_reference`** lets the original `frontier_v1` heuristic use stairs, rest and items. It is a reference and is never paired with a restricted policy.

## Outcome taxonomy and failure policy

| Class | Reasons | In the competence mean? |
|---|---|---|
| Task end | `TASK_BUDGET`, `TASK_DEPTH_EXIT` | yes |
| Natural end | `DEATH`, `QUIT`, `VICTORY`, `SUPERVICTORY` | yes |
| Configured limit | `TRUNCATED_ACTION_LIMIT` (event guard) | yes, and counted |
| Operational failure | `RESET_ERROR`, `ENGINE_ERROR`, `PROTOCOL_ERROR`, `POLICY_ERROR`, `TRUNCATED_TIME_LIMIT` | no, reported and retried |
| No task instance | `PREROLL_UNAVAILABLE` | no, reported, same seeds for all |

An operational failure is retried up to twice with the same game seed and policy RNG. Every attempt is recorded. In training a failed attempt commits no weights or counters. If retries are exhausted, evaluation keeps the record invalid and paired statistics report the dropped pair; training stops as `FAILED_OPERATIONAL`. A final evaluation with any unresolved failure is reported as incomplete, and a sensitivity bound assigns the failed seed a score of zero.

## Seeds

Master `20261103`, domain `yendorlab-m5-v1:{master}:{split}:{stream}:{index}`, disjoint from every M4 seed by test.

| Split | Size | Use |
|---|---|---|
| `train` | per run, stream `run{n}` | Learning and demonstrations |
| `validation` | first 40 | Selection of architecture, hyperparameters and checkpoints |
| `diagnostic` | first 64 | Budget choice, bias and failure analysis. Never used to select a policy |
| `final` | first 100 | One evaluation after the freeze in this file |

M4 final seeds (master `20261031`) are used only to reproduce and decompose M4 runs. Policy RNG comes from the `policy` stream of the same index, so all policies face the same seeds with the same random stream position. Validation estimates from repeated use of one seed set are optimistic after selection; only the final set estimates the selected policy.

## Budget rule

Fixed before seeing validation data. Run `frontier_moves_v2` on the first 32 diagnostic seeds with a 1,000-decision budget. The task budget is the smallest multiple of 50 at which its mean `new_cells` reaches 80% of its 1,000-decision mean. The rule keeps the task unsaturated, so a better policy can still show gain, and short enough to train on one laptop.

## Agents

Comparable on the primary metric, all restricted to eight moves with shared perception:

- `random_moves_v1`; each `fixed_move` direction; zero-weight linear (ties break by RNG, so it equals uniform random).
- `frontier_moves_v2`: hand-written planner. Attacks adjacent creatures, otherwise follows a shortest path to the nearest frontier.
- M4 policies A and B and the M4 zero-weight policy, evaluated unchanged.
- `direction_prior_v1`: memoryless sampling from a learned policy's own action shares. It measures what the preference alone achieves.
- New learned policies.

Reference only: original `frontier_v1` on `exploration_v2_full_reference`.

## Encoders

| Encoder | Content | Absolute direction |
|---|---|---|
| `abs_v2` | The M4 16 features | Yes, 8 indicators |
| `abs_v2_nodir` | M4 without indicators | Order of ties in the M4 route only |
| `rel_v3` | 15 relative features: blocked, failed edge, unknown, enemy, visits, reverse, visible, stairs, shortest-path move, local unknown count, momentum, unvisited, enclosure, diagonal | No |
| `rel_v3_dir` | `rel_v3` plus 8 indicators | Yes |
| `ego_patch_v1` | 7×7 local map in the frame of each move. No planner input | No, rotation-equivariant |

`rel_v3` and `ego_patch_v1` are tested for rotation equivariance: rotating the map and the move together leaves the features unchanged.

## Hypotheses, stated before running

**H1 (start-position anisotropy).** The M4 preference is structural: depth 1 starts at the bottom of the map, so north holds nearly all unexplored cells, and an action-indexed weight can exploit that with no spatial understanding. Prediction: on the relay task the direction advantage disappears.

**H2 (stalls).** M4 policies stop because equal features give equal scores in a stalled state, so a deterministic policy repeats a failed move. Prediction: failed-edge memory removes most stalling; M4 policies under a fixed-budget task lose much of their measured gain over controls.

**H3 (planner value).** Most of the heuristic's advantage comes from planning over the known map, not from stairs or items. Prediction: the eight-move planner matches or exceeds the original heuristic on the depth 1 metric.

**H4 (representation).** With `rel_v3`, TD outperforms `abs_v2`-based TD on validation seeds at equal budget. Direction indicators add no value on top of `rel_v3`.

**H5 (imitation).** Behavior cloning reaches a higher validation score than TD at a lower total compute cost, but only because the encoder hands it the planner's frontier signal; without that signal it degrades sharply.

**H6 (refinement).** RL started from imitation does not lose the imitated skill, and may improve on it where the planner is weak. Failing to improve is a likely and acceptable outcome.

## Ablations

Each ablation changes one factor from the reference configuration selected on validation, and uses the same training streams and validation seeds. Prioritized to explain M4:

| ID | Factor | Hypothesis |
|---|---|---|
| A1 | `rel_v3` vs `rel_v3_dir` | Indicators add nothing on top of relative features |
| A2 | Without `visits` | Visit memory is needed to avoid loops |
| A3 | Without `frontier` | The planner signal is the main source of performance |
| A4 | `cells_v2` vs `cells_shaped_v2` | Shaping is not needed once failed edges are visible |
| A5 | Without `failed` | Failed-edge memory removes stalling |
| A6 | `rel_v3` vs `abs_v2` | The M4 encoder cannot reach the new one under the same training |

Comparisons are reported as per-run means over three training runs. Runs are the replicates; seeds pair the evaluation. Causality is claimed only where one factor changed.

## Pilot budgets

Bounded per experiment before launch: at most 400 training episodes per TD run, three independent runs per promising method, at most six concurrent processes, no run above 30 minutes. Behavior cloning: 60 demonstration episodes, at most 20 epochs, three independent data and shuffle streams. Stop rule: do not raise a budget to rescue a negative result.

## Statistics

Per-seed paired bootstrap (10,000 resamples) and sign-flip tests for one policy against another. For a method with several training runs, report each run and a hierarchical bootstrap over runs then seeds; three runs do not support inference about all possible runs, so that interval is descriptive. Episodes from one checkpoint are never treated as independent training replicates.

Validation comparisons are exploratory and carry no significance claim. On the final set the primary family is fixed below and adjusted by Holm.

## Final evaluation

**Frozen on 2026-11-03, before any `final` seed was evaluated.** Budget 250 decisions, first 100 `final` seeds, `scripts/m5-suite.py`, the code path already rehearsed on diagnostic seeds (EXP-0033). Exclusion: only `PREROLL_UNAVAILABLE` seeds on the relay, which apply to every policy alike. Operational failures follow the failure policy above.

Selection, made on validation only:

| Role | Policy | Why | Checkpoint SHA-256 |
|---|---|---|---|
| P1 `bc_planner_r2` | Linear BC on `rel_v3` | Highest validation mean of all methods (970.5) | `afd591d742f4ca61d8f58e14a94d3a874b196f238baa4f0a7bbb081bb0979c8e` |
| P2 `warm_no_planner_r1` | BC then TD on `rel_v3` without the planner signal | Highest validation mean among policies that do not read a planner (589.0) | `da844ae6e7da42b4167cf4b973d91013635f5757726ff50a2a6ec89892fdd8f4` |

Direction priors for the control come from each policy's action shares on diagnostic seeds: `data/m5/priors.json`, SHA-256 `33f67182dab4b6b94b89ca9dff3173881a7b75c6a2ca5c3f09dc05d0c053b96e`. The best fixed direction was chosen on validation: north, `fixed_k`.

**Primary family, Holm-adjusted, ten comparisons.** For P1 and for P2, on the floor task: against random, `fixed_k` and its own direction prior. On the relay task: against random and `fixed_k`. A comparison succeeds when its paired 95% bootstrap interval lies above zero and its Holm-adjusted sign-flip p is below 0.05.

**Gate E decision rule.** Learning evidence for a policy requires all five of its comparisons to succeed. P1 carries the planner signal, so success shows a learned policy reproduces planner behavior, not independent navigation. Only P2 can support a claim about navigation without a planner, and only within depths 1 and 2.

Descriptive, not in the family: each policy against `frontier_moves_v2`, all three runs of each method, M4 policies, the multilevel task and the full-action reference. If methodology changes after seeing these results, a new protocol version and a new final set are required.

## Threats to validity

Selection on 40 validation seeds. Three training runs. The planner feature embeds hand-written knowledge, so a high score with it is not independent discovery. Depth 1 and depth 2 are a narrow slice of Brogue. Same-build determinism only. WSL scheduling affects timing, not game outcomes, unless the episode time guard fires, which would be counted as an operational failure.

## Protocol changes

- 2026-11-03, all pilots rerun. The first flush fix (`patch 3 draft`) corrupted the recording position when a prompt was declined after a flush, and the child hung. The failure policy caught it; those campaigns are archived in ignored `data/m5/invalidated-flush-bug/` and contributed no reported number. They did reveal that `rel_v3` was promising before the hyperparameter scan was planned, so that scan is not blind.
- 2026-11-03, additions decided after seeing validation data and therefore exploratory: a four-setting hyperparameter scan (EXP-0027), the tiny MLP capacity test (EXP-0028b), and a 400-episode second stage (EXP-0032). The final family above was set after all of them.
- 2026-11-03, budget fixed by the rule (EXP-0025, 32 diagnostic seeds): `frontier_moves_v2` reached 1,163.2 cells at 1,000 decisions and 961.4 at 250, the first multiple of 50 above 80%. **The budget of `exploration_v2_floor` and `exploration_v2_relay` is 250 decisions.**
