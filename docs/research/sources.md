# Verified starting references

| Work | Author/year | Contribution to YendorLab | Limitation |
|---|---|---|---|
| [Brogue: Community Edition](https://github.com/tmewett/BrogueCE), [v1.15.1](https://github.com/tmewett/BrogueCE/releases/tag/v1.15.1) | Brian Walker; community edition maintained by Tom Mewett and contributors, 2026 release | Official game rules, source, seed/variant CLI and event loop | No interactive headless control API in upstream `nullConsole` |
| [BrogueBot](https://github.com/tmewett/BrogueBot), [guide](https://github.com/tmewett/BrogueBot/wiki/User-Guide) | Tom Mewett, inspected 2022 commit | Historical Lua action/observation API; visibility gating example | Replaces older Brogue source, modern CE compatibility unproved |
| [The NetHack Learning Environment](https://github.com/facebookresearch/nle) | Heinrich Küttler et al., NeurIPS 2020 | Engine integration, standardized tasks and agent evaluation | Different game; repository archived in 2024 |
| [Gymnasium documentation](https://gymnasium.farama.org/) | Farama Foundation, current online documentation | `reset`, `step`, termination/truncation conventions | Dependency and exact Brogue action model not yet validated |
| [Deep Reinforcement Learning at the Edge of the Statistical Precipice](https://arxiv.org/abs/2108.13264), [rliable](https://github.com/google-research/rliable) | Rishabh Agarwal, Max Schwarzer, Pablo Samuel Castro, Aaron Courville, Marc G. Bellemare, NeurIPS 2021 | Uncertainty-aware reporting and robust aggregate comparisons | Examples mostly multi-task deep RL; one-game protocol needs adaptation |

The source inspection focused on `RogueMain.c` (initialization and RNG seeds), `IO.c` (input loop and key commands), `main.c` (CLI), `null-platform.c` (headless console), `Recordings.c` (recording playback), and BrogueBot `Bot.c` (Lua observations). Procedural generation uses level seeds generated from the master seed; this has not yet been audited across all random calls. Existing recording playback is not an interactive simulator. Refer to [ADR 0001](../decisions/0001-integration.md) for the decision and uncertainty.
