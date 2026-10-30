# Scientific diary

## 2026-10-30 — M0

Goal: determine whether official Brogue CE can be controlled headlessly without changing game rules. Expected an input hook might exist near the console abstraction. Found `nullConsole` but its input callback returned without an action; upstream `-vn` was playback only. BrogueBot showed a historical Lua integration, but its source replacement approach was too broad for this first revision. Built upstream with SDL and terminal disabled, then added a platform-only JSON/stdin patch. Two seeds advanced real game turns. Repeated seed/action policy produced identical restricted observations in the tested WSL build. Random runs were all truncated and stayed at depth 1, so there is no valid game outcome comparison. Next: context-aware actions, visible terrain, natural terminal tests and upstream regression validation.
