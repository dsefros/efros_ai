from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from kernel.exceptions import (
    AIPlatformError,
    ValidationError,
    ModelNotFoundError,
    ExecutorNotFoundError,
    PipelineNotFoundError,
    PipelineStepError,
    KnowledgeError,
)


class Query(BaseModel):
    prompt: str = Field(..., min_length=1)
    model: str | None = None


class RagQuery(BaseModel):
    query: str = Field(..., min_length=1)
    model: str | None = None


class RagSearchQuery(BaseModel):
    query: str = Field(..., min_length=1)
    limit_per_collection: int | None = Field(default=None, ge=1, le=50)


class AgentPayload(BaseModel):
    payload: dict


class PipelinePayload(BaseModel):
    payload: dict


def _raise_http(e: Exception):
    if isinstance(e, ValidationError):
        raise HTTPException(status_code=400, detail=str(e))
    if isinstance(e, ModelNotFoundError):
        raise HTTPException(status_code=404, detail=str(e))
    if isinstance(e, ExecutorNotFoundError):
        raise HTTPException(status_code=404, detail=str(e))
    if isinstance(e, PipelineNotFoundError):
        raise HTTPException(status_code=404, detail=str(e))
    if isinstance(e, PipelineStepError):
        raise HTTPException(status_code=500, detail=str(e))
    if isinstance(e, KnowledgeError):
        raise HTTPException(status_code=500, detail=str(e))
    if isinstance(e, AIPlatformError):
        raise HTTPException(status_code=500, detail=str(e))
    raise HTTPException(status_code=500, detail=f"Unhandled error: {e}")


def create_app(kernel, model_manager):
    app = FastAPI(title="AI Platform API")

    @app.get("/")
    def root():
        return {"status": "AI platform running"}

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "knowledge": kernel.knowledge is not None,
            "models": sorted(list(model_manager.models.keys())),
            "default_model": model_manager.default_model,
            "executors": sorted(list(kernel.executors.keys())),
            "pipelines": sorted(list(kernel.pipeline_engine.pipelines.keys())),
            "tools": sorted(list(kernel.tools.tools.keys())),
        }

    @app.get("/models")
    def list_models():
        return model_manager.list_models()

    @app.post("/llm")
    def run_llm(query: Query):
        try:
            result = model_manager.generate(query.prompt, model_name=query.model)
            return {
                "response": result,
                "model": query.model or model_manager.default_model,
            }
        except Exception as e:
            _raise_http(e)

    @app.post("/agent/{name}")
    def run_agent(name: str, body: AgentPayload):
        try:
            result = kernel.run_executor(name, body.payload)
            return {"result": result}
        except Exception as e:
            _raise_http(e)

    @app.post("/pipeline/{name}")
    def run_pipeline(name: str, body: PipelinePayload):
        try:
            result = kernel.pipeline_engine.run_pipeline(name, body.payload)
            return {"result": result}
        except Exception as e:
            _raise_http(e)

    @app.post("/rag/search")
    def rag_search(payload: RagSearchQuery):
        try:
            limit = payload.limit_per_collection
            if limit is None:
                hits = kernel.knowledge.search(payload.query)
            else:
                hits = kernel.knowledge.search(payload.query, limit_per_collection=limit)

            if hasattr(kernel.knowledge, "_compact_sources"):
                hits = kernel.knowledge._compact_sources(hits)

            return {"sources": hits}
        except Exception as e:
            _raise_http(e)

    @app.post("/rag/answer")
    def rag_answer(payload: RagQuery):
        try:
            result = kernel.knowledge.answer(payload.query, model_name=payload.model)
            return result
        except Exception as e:
            _raise_http(e)

    return app
