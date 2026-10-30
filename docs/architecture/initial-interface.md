# Initial interface, schema 1

Upstream: Brogue CE `v1.15.1`, commit `1ba4240b7a928ddf0ffb772717bf1d433cd63804`. The patch adds `--yendorlab` to select `nullConsole`. It reads one ASCII action from stdin for each native input callback and writes one JSON observation to stdout. It leaves the game loop, procedural generation, RNG, and combat code unchanged. Each Python `run_episode` starts a fresh process, providing a hard reset and crash isolation at the cost of startup overhead.

An observation includes `schema=1`, game seed, depth, turn, player position, current and maximum HP, `ended`, and coordinates for cells satisfying Brogue's `playerCanSee` predicate. It omits terrain, entities, items, inventory, RNG state, and undiscovered cells. This narrow schema prevents the obvious hidden state leak but is insufficient to prove a complete player equivalent observation. No debug state is given to the policy. Any additional field requires a visibility review and a leakage test.

Action schema 1 accepts `h j k l y u b n` (eight movements), plus `z` rest, `<` ascend, `>` descend and space at native protocol level. The Python random baseline selects only the eight directions. Invalid input becomes `z`. Native callbacks may occur in menus or confirmations; the current protocol has no context marker or action legality mask. Thus one callback is an **input event**, not necessarily one game turn or one strategic decision. The turn field reveals when actions consume no game time. A terminal event contains result, score, depth and turn via Brogue's `notifyEvent` callback. Python status `completed` requires this event; action or time limits yield `truncated`.

The protocol is intentionally below Gymnasium for now. A future `reset/step` wrapper must define menus, action masks, rewards, and `terminated` versus `truncated` before claiming Gymnasium compatibility. Python currently uses only its standard library; NumPy, Gymnasium, PyTorch, and MLflow would add cost without solving the native contract.

### Integrity and limitations

The visible coordinates derive from the engine's visibility flags, not from all `pmap` cells. Neither hidden monster pointers nor unidentified item tables are read. A source review supports this limited claim. Formal differential leakage tests, magical detection cases, hallucination semantics, and observation equality to a human player's display remain pending. A per process reset is tested; multiple episodes within one process are not supported. Recording playback is an existing upstream feature, distinct from this interactive headless protocol.
