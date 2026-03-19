# efros_ai

## Overview
`efros_ai` is a Python-based AI platform skeleton centered on a FastAPI server, a kernel/bootstrap layer, module loading, background job processing, and retrieval-augmented generation (RAG) services. The repository currently exposes API endpoints for health checks, model listing, LLM calls, agent execution, pipeline execution, and RAG search/answer flows through `python run.py`. 

## Current status
- The repository is currently **bootable** through `python run.py`.
- The codebase is still being **stabilized**; startup compatibility and safe local development are higher priorities than broad refactors.
- Recent recovery work restored the `services.models.model_manager` import path used by `run.py`, which is a critical startup dependency.
- Contributors should treat the current startup path as a compatibility baseline and avoid changes that could regress it.

## Repository and runtime context
What can be safely inferred from the current codebase:
- `run.py` configures logging, builds the kernel, starts a worker thread, registers knowledge services, loads `modules/support_module`, creates the FastAPI app, and starts Uvicorn. 
- The API application exposes `/`, `/health`, `/models`, `/llm`, `/agent/{name}`, `/pipeline/{name}`, `/rag/search`, and `/rag/answer`.
- Configuration is environment-driven via `configs/settings.py` and `python-dotenv`.
- The codebase contains tests under `tests/` and a `pytest.ini` configuration that points pytest at that directory.
- The model-management code lives under `services/models/`. Keep that separate from the repository-root `/models/` path, which is intentionally ignored for local artifacts.

## Repository layout
Key paths:
- `run.py` — application bootstrap and local entry point.
- `api/` — FastAPI app construction and request handlers.
- `kernel/` — core platform/kernel wiring.
- `modules/` — loadable modules, including `modules/support_module`.
- `services/` — events, jobs, knowledge, model, and speech services.
- `tests/` — automated test suite.
- `configs/settings.py` — environment-backed configuration values.

## Local setup
### 1. Create and activate a virtual environment
```bash
python -m venv venv
source venv/bin/activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure environment variables
Start from the example file if you need local configuration:
```bash
cp .env.example .env
```
Then fill in values appropriate for your machine.

### 4. Confirm model/backend prerequisites
The default configuration expects local model paths and a Qdrant URL to be provided through environment variables. Review `configs/settings.py` and `.env.example` before attempting full LLM or RAG flows.

## Run locally
Start the application with:
```bash
python run.py
```

By default, the app reads `API_HOST` and `API_PORT` from the environment and starts Uvicorn with those values.

## Dependencies and environment notes
- `requirements.txt` currently includes the core packages needed for the application bootstrap path, including FastAPI, Uvicorn, dotenv support, YAML parsing, and the Qdrant client.
- `sentence-transformers` is listed as an optional dependency for fuller RAG behavior.
- The code under `services/models/llama_cpp_model.py` imports `llama_cpp`, so local model execution also depends on that runtime being available in your environment.
- `.gitignore` already excludes local secrets, virtual environments, Python caches, pytest cache, Windows `Zone.Identifier` metadata, and the repository-root `/models/` directory. Keep those artifacts out of Git.

## Model manager compatibility note
`run.py` depends on `services.models.model_manager.create_default_manager`. That import path previously caused startup issues when the module was missing from the active tree, so it should be treated as a compatibility-sensitive entry point.

At present, `services/models/model_manager.py` provides:
- `ModelManager.register(name, model)`
- `ModelManager.get(name=None)`
- `ModelManager.list_models()`
- `ModelManager.generate(prompt, model_name=None)`
- `create_default_manager()`

The current implementation registers llama.cpp-backed models from environment-configured paths and sets the default model from `DEFAULT_MODEL`. Avoid unnecessary refactors here unless you are deliberately updating the startup contract.

## Tests
This repository includes a pytest configuration and a `tests/` directory, so the current documented test command is:
```bash
pytest -q
```

If your environment does not have all optional runtime dependencies or external services available, some tests or runtime paths may still require additional setup.

## Known limitations and next steps
- The project is in a stabilization phase; documentation and compatibility protections are intentionally conservative.
- Full local success for LLM/RAG paths depends on environment-specific services and model files that are not stored in the repository.
- The startup path should remain the highest-priority regression check when making changes.
- Future improvements should favor small, additive updates over broad architectural refactors until the baseline is more mature.
