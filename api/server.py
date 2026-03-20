from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from configs.domain_profiles import DomainProfileError
from services.knowledge.access_policy import DomainAccessPolicyEvaluator, RequestAccessContext

from kernel.exceptions import (
    AIPlatformError,
    ValidationError,
    ModelNotFoundError,
    ExecutorNotFoundError,
    PipelineNotFoundError,
    PipelineStepError,
    KnowledgeError,
    DomainNotFoundError,
    AccessDeniedError,
)


class Query(BaseModel):
    prompt: str = Field(..., min_length=1)
    model: str | None = None


class RagQuery(BaseModel):
    query: str = Field(..., min_length=1)
    model: str | None = None
    domain: str | None = Field(default=None, min_length=1)


class RagSearchQuery(BaseModel):
    query: str = Field(..., min_length=1)
    domain: str | None = Field(default=None, min_length=1)
    limit_per_collection: int | None = Field(default=None, ge=1, le=50)


class AgentPayload(BaseModel):
    payload: dict


class PipelinePayload(BaseModel):
    payload: dict


def _raise_http(e: Exception):
    if isinstance(e, ValidationError):
        raise HTTPException(status_code=400, detail=str(e))
    if isinstance(e, DomainNotFoundError):
        raise HTTPException(status_code=404, detail=str(e))
    if isinstance(e, AccessDeniedError):
        raise HTTPException(status_code=403, detail=str(e))
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
    raise HTTPException(status_code=500, detail=f'Unhandled error: {e}')



def _split_header_values(value: str | None) -> tuple[str, ...]:
    if value is None:
        return ()
    items = []
    seen = set()
    for raw in value.split(','):
        cleaned = raw.strip()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        items.append(cleaned)
    return tuple(items)


def _build_access_context(request: Request) -> RequestAccessContext:
    return RequestAccessContext.from_values(
        caller=request.headers.get('x-efros-caller'),
        roles=_split_header_values(request.headers.get('x-efros-roles')),
        groups=_split_header_values(request.headers.get('x-efros-groups')),
    )


def _resolve_domain_registry(kernel, runtime=None):
    knowledge_settings = getattr(getattr(kernel, 'knowledge', None), 'settings', None)
    knowledge_registry = getattr(knowledge_settings, 'domain_registry', None)
    if knowledge_registry is not None:
        return knowledge_registry

    kernel_registry = getattr(kernel, 'domain_registry', None)
    if kernel_registry is not None:
        return kernel_registry

    kernel_settings = getattr(kernel, 'settings', None)
    kernel_settings_registry = getattr(kernel_settings, 'domain_registry', None)
    if kernel_settings_registry is not None:
        return kernel_settings_registry

    runtime_registry = getattr(runtime, 'domain_registry', None)
    if runtime_registry is not None:
        return runtime_registry

    runtime_settings = getattr(runtime, 'settings', None)
    runtime_settings_registry = getattr(runtime_settings, 'domain_registry', None)
    if runtime_settings_registry is not None:
        return runtime_settings_registry

    return None


def create_app(kernel, model_manager, runtime=None):
    access_evaluator = DomainAccessPolicyEvaluator(_resolve_domain_registry(kernel, runtime=runtime))

    def enforce_domain_access(domain: str | None, request: Request) -> None:
        if access_evaluator.registry is None:
            return
        try:
            decision = access_evaluator.evaluate(domain, context=_build_access_context(request))
        except DomainProfileError as exc:
            raise DomainNotFoundError(str(exc)) from exc
        if not decision.allowed:
            raise AccessDeniedError(f"Access denied for domain '{decision.domain}': {decision.reason}")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if runtime is not None:
            app.state.runtime = runtime
        try:
            yield
        finally:
            if runtime is not None:
                runtime.shutdown()

    app = FastAPI(title='AI Platform API', lifespan=lifespan)

    @app.get('/')
    def root():
        return {'status': 'AI platform running'}

    @app.get('/health')
    def health():
        return {
            'status': 'ok',
            'knowledge': kernel.knowledge is not None,
            'models': sorted(list(model_manager.models.keys())),
            'default_model': model_manager.default_model,
            'executors': sorted(list(kernel.executors.keys())),
            'pipelines': sorted(list(kernel.pipeline_engine.pipelines.keys())),
            'tools': sorted(list(kernel.tools.tools.keys())),
        }

    @app.get('/models')
    def list_models():
        return model_manager.list_models()

    @app.get('/domains')
    def list_domains():
        if kernel.knowledge is None or not hasattr(kernel.knowledge, 'list_domains'):
            return {'domains': []}
        return {'domains': kernel.knowledge.list_domains()}

    @app.post('/llm')
    def run_llm(query: Query):
        try:
            result = model_manager.generate(query.prompt, model_name=query.model)
            return {
                'response': result,
                'model': query.model or model_manager.default_model,
            }
        except Exception as e:
            _raise_http(e)

    @app.post('/agent/{name}')
    def run_agent(name: str, body: AgentPayload):
        try:
            result = kernel.run_executor(name, body.payload)
            return {'result': result}
        except Exception as e:
            _raise_http(e)

    @app.post('/pipeline/{name}')
    def run_pipeline(name: str, body: PipelinePayload):
        try:
            result = kernel.pipeline_engine.run_pipeline(name, body.payload)
            return {'result': result}
        except Exception as e:
            _raise_http(e)

    @app.post('/rag/search')
    def rag_search(payload: RagSearchQuery, request: Request):
        try:
            enforce_domain_access(payload.domain, request)
            limit = payload.limit_per_collection
            hits = kernel.knowledge.search(payload.query, domain=payload.domain, limit_per_collection=limit)

            if hasattr(kernel.knowledge, '_compact_sources'):
                hits = kernel.knowledge._compact_sources(hits)

            return {'sources': hits}
        except Exception as e:
            _raise_http(e)

    @app.post('/rag/answer')
    def rag_answer(payload: RagQuery, request: Request):
        try:
            enforce_domain_access(payload.domain, request)
            result = kernel.knowledge.answer(payload.query, domain=payload.domain, model_name=payload.model)
            return result
        except Exception as e:
            _raise_http(e)

    return app
