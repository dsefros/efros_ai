from fastapi.testclient import TestClient

from api.server import create_app
from kernel.ai_kernel import AIKernel
from kernel.exceptions import DomainNotFoundError
from kernel.module_loader import load_module


class FakeModelManager:
    def __init__(self):
        self.models = {'mock': object(), 'qwen2': object()}
        self.default_model = 'mock'

    def list_models(self):
        return {
            'default_model': self.default_model,
            'models': sorted(list(self.models.keys())),
        }

    def generate(self, prompt: str, model_name: str | None = None):
        model_name = model_name or self.default_model
        return f'[{model_name}] {prompt}'


class FakeKnowledge:
    def __init__(self):
        self.search_calls = []
        self.answer_calls = []

    def search(self, query: str, domain: str | None = None, limit_per_collection: int = 5):
        if domain == 'missing':
            raise DomainNotFoundError("Unknown domain 'missing'")
        self.search_calls.append({'query': query, 'domain': domain, 'limit_per_collection': limit_per_collection})
        return [
            {
                'score': 0.99,
                'rerank_score': 1.23,
                'text': f'chunk for {query}',
                'metadata': {
                    'source': 'doc.pdf',
                    'page_number': 3,
                    'doc_type': 'product',
                    'source_db': domain or 'rag_product',
                },
                'collection': domain or 'rag_product',
            }
        ]

    def _compact_sources(self, items):
        out = []
        for item in items:
            meta = item.get('metadata', {})
            out.append(
                {
                    'score': item.get('score'),
                    'rerank_score': item.get('rerank_score'),
                    'text': item.get('text'),
                    'source': meta.get('source'),
                    'page_number': meta.get('page_number'),
                    'doc_type': meta.get('doc_type'),
                    'source_db': meta.get('source_db'),
                    'collection': item.get('collection'),
                }
            )
        return out

    def answer(self, query: str, domain: str | None = None, model_name: str | None = None):
        if domain == 'missing':
            raise DomainNotFoundError("Unknown domain 'missing'")
        self.answer_calls.append({'query': query, 'domain': domain, 'model_name': model_name})
        return {
            'answer': f'answer for {query} with {model_name or "default"}',
            'sources': self._compact_sources(self.search(query, domain=domain)),
            'domain': domain or 'default',
        }

    def list_domains(self):
        return [
            {'name': 'default', 'is_default': True, 'collections': ['rag_product', 'rag_regulatory'], 'description': 'default domain'},
            {'name': 'finance', 'is_default': False, 'collections': ['finance_docs'], 'description': None},
        ]


def build_test_client():
    kernel = AIKernel()
    load_module(kernel, 'modules/support_module')
    kernel.knowledge = FakeKnowledge()
    model_manager = FakeModelManager()
    app = create_app(kernel, model_manager)
    return TestClient(app)


def test_health():
    client = build_test_client()
    resp = client.get('/health')
    assert resp.status_code == 200
    data = resp.json()
    assert data['status'] == 'ok'
    assert 'support_agent' in data['executors']
    assert 'demo' in data['pipelines']
    assert 'search' in data['tools']


def test_models():
    client = build_test_client()
    resp = client.get('/models')
    assert resp.status_code == 200
    data = resp.json()
    assert data['default_model'] == 'mock'
    assert 'mock' in data['models']


def test_domains_endpoint():
    client = build_test_client()
    resp = client.get('/domains')
    assert resp.status_code == 200
    data = resp.json()
    assert data['domains'][0]['name'] == 'default'
    assert data['domains'][1]['collections'] == ['finance_docs']


def test_llm():
    client = build_test_client()
    resp = client.post('/llm', json={'prompt': 'hello', 'model': 'qwen2'})
    assert resp.status_code == 200
    assert resp.json()['response'] == '[qwen2] hello'


def test_agent():
    client = build_test_client()
    resp = client.post('/agent/support_agent', json={'payload': {'query': 'payment error'}})
    assert resp.status_code == 200
    assert resp.json()['result']['result'] == 'found payment error'


def test_pipeline():
    client = build_test_client()
    resp = client.post('/pipeline/demo', json={'payload': {'text': 'hello'}})
    assert resp.status_code == 200
    data = resp.json()['result']
    assert data['text'] == 'HELLO'
    assert data['result'] == 'HELLO!!!'


def test_rag_search():
    client = build_test_client()
    resp = client.post('/rag/search', json={'query': 'virtual cash register', 'domain': 'finance'})
    assert resp.status_code == 200
    data = resp.json()
    assert 'sources' in data
    assert len(data['sources']) == 1
    assert data['sources'][0]['source'] == 'doc.pdf'
    assert data['sources'][0]['collection'] == 'finance'


def test_rag_search_unknown_domain_returns_404():
    client = build_test_client()
    resp = client.post('/rag/search', json={'query': 'virtual cash register', 'domain': 'missing'})
    assert resp.status_code == 404
    assert resp.json()['detail'] == "Unknown domain 'missing'"


def test_rag_answer():
    client = build_test_client()
    resp = client.post('/rag/answer', json={'query': 'virtual cash register', 'model': 'mock', 'domain': 'finance'})
    assert resp.status_code == 200
    data = resp.json()
    assert 'answer' in data
    assert 'sources' in data
    assert data['domain'] == 'finance'


def test_rag_answer_unknown_domain_returns_404():
    client = build_test_client()
    resp = client.post('/rag/answer', json={'query': 'virtual cash register', 'domain': 'missing'})
    assert resp.status_code == 404
    assert resp.json()['detail'] == "Unknown domain 'missing'"


def test_app_lifespan_retains_runtime_and_shuts_it_down():
    class RuntimeStub:
        def __init__(self):
            self.shutdown_calls = 0

        def shutdown(self):
            self.shutdown_calls += 1

    kernel = AIKernel()
    load_module(kernel, 'modules/support_module')
    kernel.knowledge = FakeKnowledge()
    model_manager = FakeModelManager()
    runtime = RuntimeStub()
    app = create_app(kernel, model_manager, runtime=runtime)

    with TestClient(app) as client:
        response = client.get('/health')
        assert response.status_code == 200
        assert client.app.state.runtime is runtime

    assert runtime.shutdown_calls == 1
