# Interface contract

Current state first; earlier contracts follow as history. Protocol **3**, observation schema **3**, action schema **2**. Upstream is Brogue CE `v1.15.1`, commit `1ba4240b7a928ddf0ffb772717bf1d433cd63804`, plus one tracked [patch](../../native/patches/0001-yendorlab-stream.patch). Decisions are in [ADR 0004](../decisions/0004-m5-benchmark.md); the task is defined in the [M5 protocol](../experiments/m5-protocol.md).

## Native channel

`--yendorlab` selects `nullConsole`. Before each native input callback the adapter writes one JSON line to stdout and reads one key from stdin. It leaves the game loop, generation, RNG and combat unchanged. The serializer uses only read paths; `getCellAppearance` runs on the cosmetic RNG. The only state the adapter touches is the input recording, which it flushes in lab mode: upstream flushes on turn ends, so about 360 turnless inputs overflowed the buffer and printed into stdout. A separate `terminal` line comes from `notifyEvent`.

Fields: `protocol`, `observation_schema`, `action_schema`, `decision`, `context`, `seed`, `depth`, `turn`, `x`, `y`, `hp`, `max_hp`, `width`, `height`, `admissible`, `cells`, `inventory`, `floor_items`, `creatures`. `decision` counts input callbacks, not strategic choices.

- `cells`: `[x, y, visible, glyph, blocked, stairs]` for cells the player has discovered, magic-mapped or currently sees. Unknown cells have no record. `glyph` is what the screen shows. `blocked` follows real flags for visible cells, remembered flags otherwise, and the displayed glyph for secret tiles. `stairs` is 1 down, -1 up, from the glyph.
- `creatures`: `[x, y, glyph]` for monsters `canSeeMonster` accepts, which includes telepathy.
- `floor_items`: items on visible cells, with the text the location description shows, so hallucination hides real names. Magically detected remote items are omitted: under-reporting, not a leak.
- `inventory`: letter, category, quantity, equipped and player-facing name.

Schema 3 changed two things from schema 2: floor item text, and the displayed glyph for cells with no stable memory (such as magic mapping). The fields read by M4 features did not change, so M4 checkpoints still load. Replay of M4-A reproduced 194.5625 mean cells and 1,401 events on the M4 final seeds.

`YENDORLAB_TESTING` builds add `--yendorlab-test-event`, `--yendorlab-test-scenario` and a privileged truth stream on **stderr**. `scripts/test-native-events.sh` asserts that the production binary contains none of this.

## Python environment

`BrogueEnv` steps one native event: `reset(seed)` returns `(observation, info)`; `step(key)` returns `(observation, reward, terminated, truncated, info)`. Each episode has a fresh process and a private directory under `/dev/shm` or `$YENDORLAB_RUNTIME_DIR`, because upstream writes recordings, high scores and run history to its cwd. The reader owns its buffer and deadline, so split lines, several lines in one read, lines above the pipe size and stalled children are all handled, and stderr goes to a file so it cannot block the child. Child address space is capped with `prlimit`. `trace_digest()` hashes every native line and action of an episode.

Outcomes are classified in [the M5 protocol](../experiments/m5-protocol.md#outcome-taxonomy-and-failure-policy). Operational failures never enter a competence mean. `info` carries `events`, `turn`, `new_cells` and `new_cells_by_depth`.

## Task layer

`tasks.run_episode` is the only loop used by M5 evaluation, training and demonstration. It routes prompts with the safe context router, enforces the action contract, and gives every policy the same `Perception` memory. Policies are data (`policies.build_policy`), so worker processes can rebuild them.

Encoders are versioned in `encoders.py`. `rel_v3` and `ego_patch_v1` carry no absolute direction and are tested for rotation equivariance. `linear_td_v2` is a shared linear scorer with normalized, clipped TD(0); checkpoints name their encoder and reject other dimensions.

The optional `GymExplorationEnv` remains a movement adapter over the M4 encoder and is not part of the M5 benchmark.

## Known gaps

Full-game play needs more than this contract: throwing, device targeting and most item workflows are unsupported; `waitForKeystrokeOrMouseClick` and nested item detail boxes report a stale context; status effects are not exposed; no live victory has occurred. Differential tests cover the listed perception cases, not every possible interaction.

# History

## M1–M4 contract, protocol 2

The C adapter emits one JSON line before every native input callback: `protocol`, `schema`, `decision`, `context`, `seed`, `turn`, `player`, `cells`, `inventory`, `creatures`, `floor_items`, and `admissible`. `decision` counts input callbacks, not strategic choices. The context is instrumented at the actual input site: `gameplay`, `inventory`, `confirmation`, `acknowledgment`, `text_input`, `death_screen`, or `target_selection`. The Python env sends one key or `ESC`/`ENTER` per callback. Unexpected input is a protocol error. A separate JSON `terminal` line comes from `notifyEvent` and distinguishes death, quit, victory and supervictory. The runner responds to death screen and postgame prompts until the canonical callback arrives, then closes the child. It does not equate process exit with game outcome.

Map `cells` are sparse six-element records `[x, y, visible, glyph, blocked, stairs]` only for cells Brogue marks discovered or visible. Unknown cells have no record. Visible glyphs come through Brogue's appearance function. The blocked flag for visible cells follows the displayed glyph for secret-looking tiles, avoiding a hidden passability leak. Previously known cells represent memory and may be stale. Creatures and floor items are emitted only at visible coordinates. Inventory names use Brogue's player-facing naming routine; item letters are stable within a prompt, but not persistent item IDs. The current representation does not cover all magical detection, hallucination and identification cases. Schema changes require a new version and checkpoint compatibility decision.

`BrogueEnv.reset(seed)` starts a fresh process; `step(action)` returns `(observation, reward, terminated, truncated, info)`. Natural terminal outcomes are `DEATH`, `QUIT`, `VICTORY`, `SUPERVICTORY`. Operational outcomes include `TRUNCATED_ACTION_LIMIT`, `TRUNCATED_TIME_LIMIT`, `TRUNCATED_NO_PROGRESS`, `ENGINE_ERROR`, `PROTOCOL_ERROR`. `info` separates input `events`, policy `decisions` and game `turn`. Limits cover time, event count, no progress and sampled child memory. An event may be a menu selection without consuming a turn. Menu operations are currently exposed as separate decisions; no compound action hides internal events.

Gameplay admissible keys are movement `h j k l y u b n`, rest `z`, stairs `< >`, inventory `i`, apply `a`, equip `e`, remove `r`, and quit `Q`. Item letters, confirmations and targeting are contextual. The list is an interface affordance, not a guarantee the attempted game action succeeds. It intentionally does not reveal unknown traps or items. Throwing, advanced device targeting, all item workflows and strategic action masks remain incomplete. The Q model uses only movement and rest in gameplay; context actions are deterministic. Full-game training is not supported.

Rewards are versioned: `canonical` assigns +1 victory and -1 death, zero otherwise; `exploration_v1` additionally assigns 0.01 per newly discovered cell and +1 per new maximum depth. Resting or revisiting a cell grants no repeat exploration reward. Reward totals are recorded separately from game metrics. `encode_state` version 1 compresses immediate local passability, visible threat and HP bucket. It discards most spatial history and inventory detail, limiting potential generalization.

### M4 exploration adapter and feature version 2

The canonical `BrogueEnv` still steps one native event. Its `info.new_cells` counts newly discovered `(depth,x,y)` cells after reset independently of reward settings. Evaluation also records action counts, stalled moves, reset time, policy time, step time, event count, gameplay decisions, turns and sampled child RSS. Reset or protocol failures become explicit error episodes in evaluation rather than vanishing from denominators.

The optional `GymExplorationEnv` is a scoped logical movement adapter. It exposes `Discrete(8)` movement actions and a `Box(128,)` float observation containing eight 16-element player-visible action feature vectors. It automatically acknowledges only safe non-gameplay prompts; `info.events_in_step` and `info.turns_in_step` retain native costs. It is unsuitable for full-game item/target decisions. Gymnasium 1.2.2 `check_env()` passed with this adapter. The raw protocol remains authoritative for auditing. Version 1 tabular checkpoints remain loadable by their original code; new linear checkpoints require feature encoder version 2 and reject incompatible schemas.

The new 16-weight linear policy uses a local map, visible creatures, known-cell frontier routing, within-episode visit counts and action direction. It cannot infer an undiscovered stair or trap. Its training task rewards `10 × newly discovered-cell reward + terminal reward`; raw `new_cells` is the primary independent metric. The detailed protocol is [M4 exploration](../experiments/m4-exploration.md).


### Integrity statement of M0–M4

The visible coordinates derive from the engine's visibility flags, not from all `pmap` cells. Neither hidden monster pointers nor unidentified item tables are read. A source review supports this limited claim. Formal differential leakage tests, magical detection cases, hallucination semantics, and observation equality to a human player's display remain pending. A per process reset is tested; multiple episodes within one process are not supported. Recording playback is an existing upstream feature, distinct from this interactive headless protocol.

## M0 contract, schema 1

Upstream: Brogue CE `v1.15.1`, commit `1ba4240b7a928ddf0ffb772717bf1d433cd63804`. The patch adds `--yendorlab` to select `nullConsole`. It reads one ASCII action from stdin for each native input callback and writes one JSON observation to stdout. It leaves the game loop, procedural generation, RNG, and combat code unchanged. Each Python `run_episode` starts a fresh process, providing a hard reset and crash isolation at the cost of startup overhead.

An observation includes `schema=1`, game seed, depth, turn, player position, current and maximum HP, `ended`, and coordinates for cells satisfying Brogue's `playerCanSee` predicate. It omits terrain, entities, items, inventory, RNG state, and undiscovered cells. This narrow schema prevents the obvious hidden state leak but is insufficient to prove a complete player equivalent observation. No debug state is given to the policy. Any additional field requires a visibility review and a leakage test.

Action schema 1 accepts `h j k l y u b n` (eight movements), plus `z` rest, `<` ascend, `>` descend and space at native protocol level. The Python random baseline selects only the eight directions. Invalid input becomes `z`. Native callbacks may occur in menus or confirmations; the current protocol has no context marker or action legality mask. Thus one callback is an **input event**, not necessarily one game turn or one strategic decision. The turn field reveals when actions consume no game time. A terminal event contains result, score, depth and turn via Brogue's `notifyEvent` callback. Python status `completed` requires this event; action or time limits yield `truncated`.

The protocol is intentionally below Gymnasium for now. A future `reset/step` wrapper must define menus, action masks, rewards, and `terminated` versus `truncated` before claiming Gymnasium compatibility. Python currently uses only its standard library; NumPy, Gymnasium, PyTorch, and MLflow would add cost without solving the native contract.
