from fastapi.testclient import TestClient

from api.server import create_app
from kernel.ai_kernel import AIKernel
from kernel.module_loader import load_module
from services.events.event_bus import EventBus
from services.jobs.job_queue import JobQueue


class IntegrationFakeModel:
    def generate(self, prompt: str):
        return f"generated::{prompt[:40]}"


class IntegrationModelManager:
    def __init__(self):
        self.models = {"mock": IntegrationFakeModel()}
        self.default_model = "mock"

    def get(self, name=None):
        return self.models[name or self.default_model]

    def generate(self, prompt: str, model_name: str | None = None):
        model = self.get(model_name)
        return model.generate(prompt)

    def list_models(self):
        return {
            "default_model": self.default_model,
            "models": sorted(list(self.models.keys())),
        }


class IntegrationKnowledge:
    def search(self, query: str, limit_per_collection: int = 5):
        items = []
        for idx in range(min(limit_per_collection, 2)):
            items.append(
                {
                    "score": 0.9 - idx * 0.1,
                    "rerank_score": 1.1 - idx * 0.1,
                    "text": f"result {idx} for {query}",
                    "metadata": {
                        "source": f"doc_{idx}.pdf",
                        "page_number": idx + 1,
                        "doc_type": "product",
                        "source_db": "rag_product",
                    },
                    "collection": "rag_product",
                }
            )
        return items

    def _compact_sources(self, items):
        compact = []
        for item in items:
            meta = item.get("metadata", {})
            compact.append(
                {
                    "score": item.get("score"),
                    "rerank_score": item.get("rerank_score"),
                    "text": item.get("text"),
                    "source": meta.get("source"),
                    "page_number": meta.get("page_number"),
                    "doc_type": meta.get("doc_type"),
                    "source_db": meta.get("source_db"),
                    "collection": item.get("collection"),
                }
            )
        return compact

    def answer(self, query: str, model_name: str | None = None):
        return {
            "answer": f"integration-answer::{query}::{model_name or 'default'}",
            "sources": self._compact_sources(self.search(query)),
        }


def build_integration_client():
    kernel = AIKernel()
    kernel.events = EventBus()
    kernel.jobs = JobQueue()

    load_module(kernel, "modules/support_module")

    model_manager = IntegrationModelManager()
    kernel.model_manager = model_manager
    kernel.knowledge = IntegrationKnowledge()

    app = create_app(kernel, model_manager)
    return TestClient(app)


def test_integration_health_models():
    client = build_integration_client()

    r1 = client.get("/health")
    assert r1.status_code == 200
    health = r1.json()
    assert health["status"] == "ok"
    assert "support_agent" in health["executors"]

    r2 = client.get("/models")
    assert r2.status_code == 200
    models = r2.json()
    assert models["default_model"] == "mock"
    assert models["models"] == ["mock"]


def test_integration_agent_pipeline_llm():
    client = build_integration_client()

    r1 = client.post("/agent/support_agent", json={"payload": {"query": "issue 123"}})
    assert r1.status_code == 200
    assert r1.json()["result"]["result"] == "found issue 123"

    r2 = client.post("/pipeline/demo", json={"payload": {"text": "hello"}})
    assert r2.status_code == 200
    payload = r2.json()["result"]
    assert payload["result"] == "HELLO!!!"

    r3 = client.post("/llm", json={"prompt": "test prompt"})
    assert r3.status_code == 200
    assert "generated::" in r3.json()["response"]


def test_integration_rag_search_and_answer():
    client = build_integration_client()

    r1 = client.post("/rag/search", json={"query": "virtual cash register", "limit_per_collection": 2})
    assert r1.status_code == 200
    data1 = r1.json()
    assert len(data1["sources"]) == 2
    assert data1["sources"][0]["source"].startswith("doc_")

    r2 = client.post("/rag/answer", json={"query": "virtual cash register", "model": "mock"})
    assert r2.status_code == 200
    data2 = r2.json()
    assert "integration-answer::virtual cash register::mock" == data2["answer"]
    assert len(data2["sources"]) == 2
