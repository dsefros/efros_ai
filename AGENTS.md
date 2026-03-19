# AGENTS.md

## Repository purpose
`efros_ai` is a Python AI platform repository with a FastAPI server, kernel/bootstrap layer, module loading, job execution, and RAG-related services. The immediate expectation for contributors and coding agents is to preserve a working local startup path while improving the codebase incrementally.

## Current state of the codebase
- The repository is currently bootable via `python run.py`.
- The codebase is functional enough for local startup, but it is still being stabilized.
- Runtime wiring depends on `run.py`, `api/server.py`, `kernel/`, `modules/support_module`, and service-layer components under `services/`.
- Tests exist under `tests/`, and `pytest.ini` points pytest at that suite.

## Startup compatibility history
Startup was previously broken because `run.py` imported `services.models.model_manager.create_default_manager` while `services/models/model_manager.py` was missing from the active tree.

That startup path has since been repaired through a minimal compatibility implementation at `services/models/model_manager.py`. Treat that module as startup-sensitive. Do not remove, rename, or broadly refactor it unless the change is necessary and fully validated.

## Non-negotiable rules for future edits
1. **Do not break `python run.py`.**
   - Any proposed change should preserve the current bootable state.
   - If you touch startup, imports, configuration, or model wiring, validate carefully.

2. **Do not confuse the repository-root `/models/` path with `services/models/`.**
   - `services/models/` contains source code.
   - Root `/models/` is a local artifact location that is intentionally ignored by Git.

3. **Keep ignored local artifacts out of Git.**
   - Preserve ignores for `.env`, `venv/`, `__pycache__/`, `.pytest_cache/`, Windows `Zone.Identifier` metadata, and root `/models/`.

4. **Prefer small, isolated PRs.**
   - Limit scope.
   - Avoid mixing documentation, refactors, and runtime behavior changes without a strong reason.

5. **Prefer additive, safe changes over broad refactors.**
   - Stabilization is more important than cleanup-driven churn.
   - Preserve compatibility-sensitive interfaces when possible.

6. **Do not invent unsupported features in docs or code.**
   - Infer behavior only from the current repository contents.
   - If something is not clearly supported by the codebase, document it cautiously or leave it out.

## Testing and validation expectations
Before proposing changes:
- Confirm the documentation or code matches the actual repository structure.
- Run focused validation for the area you changed.
- If tests are relevant and the environment supports them, run:
  - `pytest -q`
- If you change startup-adjacent code, also validate the `python run.py` path as far as the local environment reasonably allows.
- Call out any environment limitations explicitly rather than assuming success.

## Working style expectations for AI coding agents
- Read the repository before editing.
- Keep changes minimal and focused.
- Preserve current runtime behavior unless a change is required to restore correctness.
- When documenting the repository, describe only what is actually present now.
- Surface assumptions clearly in summaries and PR descriptions.
