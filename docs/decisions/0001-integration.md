# ADR 0001: Minimal direct C input adapter

Status: accepted provisionally, 2026-10-30.

Options investigated:

| Criterion | Direct C adapter | BrogueBot adaptation | External terminal automation |
|---|---|---|---|
| Rule integrity | Existing engine, two platform files patched | Requires port of older game source | Existing engine |
| Maintenance | Small patch, tied to platform API | Broad source replacement and Lua runtime | UI protocol and rendering coupling |
| CPU overhead | One process pipe per episode; measured below | Lua callback overhead unmeasured | Escape parsing and terminal I/O unmeasured |
| Episode isolation | Fresh process | Needs explicit reset audit | Fresh process |
| Information leak risk | Restricted new serializer | Large historical observation API needs audit | Terminal display naturally restricted |
| Compatibility | Compiled on v1.15.1 in WSL | Modern CE compatibility unproven | Existing terminal build possible |

BrogueBot's Lua API is valuable design evidence: it gates world cells by visibility and filters seen creatures/items. Its installation instructions replace the entire Brogue source tree, and its inspected commit is from 2022. A direct port would increase the surface needing review. Upstream CE already exposes seed and variant CLI options and event callbacks. The `nullConsole` input callback was empty, making a small platform patch viable. External terminal automation has no measured advantage for this stage and would need menu parsing.

Selected: direct adapter through a versioned patch, fresh process per episode. This selection is provisional because no apples to apples throughput benchmark of all three options has been run. Revisit after the action and observation contract is complete. The current patch is intentionally small and does not change gameplay rules.
