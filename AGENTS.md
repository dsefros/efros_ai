# AGENTS.md

## Repository purpose and system shape
`efros_ai` is a Python AI platform repository with a FastAPI server, kernel/bootstrap wiring, local model registration, and Qdrant-backed retrieval-augmented generation services. The repository currently runs as a single local process started through `python run.py`, with in-memory event and job infrastructure plus a knowledge engine registered during bootstrap.

## Current bootability and runtime assumptions
- The repository is expected to remain bootable via `python run.py`.
- Runtime wiring depends on `run.py`, `api/server.py`, `kernel/`, `modules/support_module`, `services/knowledge/`, and `services/models/model_manager.py`.
- Treat `services/models/model_manager.py` as startup-sensitive because `run.py` imports `create_default_manager` from that module during bootstrap.
- Tests live under `tests/`, and `pytest.ini` points pytest at that suite.

## Configuration expectations
- Runtime settings are loaded from `.env` with `python-dotenv`.
- `.env.example` is the source-of-truth example for the supported configuration contract and should stay aligned with `configs/settings.py`.
- The active configuration contract includes API host/port, LLM backend and default model, Qdrant URL plus legacy collection names, local model paths, embedding and reranker settings, LLM runtime parameters, RAG retrieval parameters, and domain-profile settings.
- `configs/settings.py` validates types and bounds for the current environment contract and raises `SettingsError` on invalid values.
- `DEFAULT_DOMAIN_NAME` and `DOMAIN_PROFILES_JSON` participate in domain configuration loading. Do not add, remove, or rename documented config variables casually; update `.env.example`, validation, and documentation together when the contract changes.

## Domain-aware behavior
- Domain profiles are loaded from `DOMAIN_PROFILES_JSON` and materialized through `configs/domain_profiles.py`.
- Domain profiles affect knowledge and RAG routing by selecting collections, retrieval settings, reranker overrides, answering defaults, ingestion settings, and access policy behavior.
- The knowledge engine exposes domain-aware behavior through `/domains`, `/rag/search`, and `/rag/answer`.
- If domain profiles are not provided, the application falls back to a compatibility domain derived from the legacy Qdrant collection and RAG retrieval settings.
- Avoid undocumented or speculative config changes around domain behavior; this area is now part of the effective runtime contract.

## Non-negotiable rules for future edits
1. **Do not break `python run.py`.**
   - Any proposed change should preserve the current bootable state.
   - If you touch startup, imports, configuration, model wiring, or knowledge registration, validate carefully.

2. **Preserve `pytest -q` health whenever the environment supports it.**
   - Run focused validation for the area you changed.
   - Call out environment limitations explicitly rather than assuming success.

3. **Do not confuse the repository-root `/models/` path with `services/models/`.**
   - `services/models/` contains source code.
   - Root `/models/` is a local artifact location that is intentionally ignored by Git.

4. **Keep ignored local artifacts out of Git.**
   - Preserve ignores for `.env`, `venv/`, `__pycache__/`, `.pytest_cache/`, Windows `Zone.Identifier` metadata, and root `/models/`.

5. **Keep changes isolated by PR.**
   - Prefer small, focused PRs.
   - Avoid mixing documentation, runtime behavior, refactors, and dependency changes without a strong reason.

6. **Keep the config contract documented and aligned.**
   - Do not add undocumented config variables casually.
   - Keep `.env.example` aligned with the real runtime contract in `configs/settings.py` and related domain-profile loaders.

7. **Do not invent unsupported features in docs or code.**
   - Infer behavior only from the current repository contents.
   - If something is not clearly supported by the codebase, document it cautiously or leave it out.

## Working style expectations for coding agents
- Read the repository before editing.
- Keep changes minimal and focused.
- Preserve current runtime behavior unless a change is required to restore correctness.
- When documenting the repository, describe only what is actually present now.
- Surface assumptions clearly in summaries and PR descriptions.
