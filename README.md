# efros_ai

## Overview
`efros_ai` is a Python AI platform centered on a FastAPI server, kernel/bootstrap wiring, local llama.cpp-backed model registration, and a Qdrant-backed retrieval layer. The local entry point is `python run.py`, which builds the runtime, registers the knowledge engine, loads `modules/support_module`, and serves the API with Uvicorn.

## Quick start
1. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Copy the example configuration and adjust values for your machine:
   ```bash
   cp .env.example .env
   ```
4. Start the app:
   ```bash
   python run.py
   ```

The server reads `API_HOST` and `API_PORT` from `.env`, then starts a single Uvicorn process.

## Configuration contract
Runtime configuration is environment-driven through `.env`, with `.env.example` acting as the source-of-truth example for supported settings. `configs/settings.py` validates and loads the current contract.

The current configuration surface includes:
- API host and port.
- LLM backend selection and the default registered model.
- Qdrant connection settings and the legacy product/regulatory collection names.
- Local model paths for the built-in llama.cpp model registrations.
- Embedding and reranker model names.
- LLM runtime parameters such as context window, thread count, GPU layers, temperature, and max tokens.
- RAG retrieval parameters such as per-collection search depth and final reranked result count.
- Domain profile settings, including `DEFAULT_DOMAIN_NAME` and `DOMAIN_PROFILES_JSON`.

## Domain profiles
Domain profiles are JSON-configured retrieval and answering profiles loaded from `DOMAIN_PROFILES_JSON`. They define named domains with their own collection lists, retrieval parameters, optional answering overrides, ingestion metadata, and access policy settings.

When domain profiles are configured, the knowledge engine resolves a domain before searching or answering. That domain selection changes which Qdrant collections are queried, which reranker model and retrieval limits are used, and which answering defaults apply. If `DOMAIN_PROFILES_JSON` is not set, the application falls back to the compatibility behavior derived from `QDRANT_PRODUCT_COLLECTION`, `QDRANT_REGULATORY_COLLECTION`, `RAG_TOP_K_PER_COLLECTION`, and `RAG_FINAL_TOP_K`.

## API and runtime shape
The current local runtime shape is intentionally simple:
- Single-process application startup via `python run.py`.
- In-memory event bus, job queue, and worker thread wiring inside the process.
- Qdrant-backed knowledge search and answer flows.
- Domain-aware knowledge endpoints, including `/domains`, `/rag/search`, and `/rag/answer`.

## Repository notes
- `run.py` is the compatibility-sensitive bootstrap path and should remain bootable.
- `services/models/` contains model source code; the repository-root `/models/` path is only for ignored local model artifacts.
- Keep `.env`, virtual environments, caches, and the root `/models/` directory out of Git.

## Testing
Run the current test suite with:
```bash
pytest -q
```

If you change startup-adjacent documentation or behavior, validate `python run.py` as far as the local environment allows because full model and Qdrant availability still depends on machine-specific setup.
