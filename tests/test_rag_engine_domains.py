from types import SimpleNamespace

import pytest

from configs.settings import Settings
from kernel.exceptions import DomainNotFoundError, ValidationError
from services.knowledge.rag_engine import KnowledgeEngine


class FakeModelManager:
    def __init__(self):
        self.calls = []

    def generate(self, prompt: str, model_name: str | None = None):
        self.calls.append({'prompt': prompt, 'model_name': model_name})
        return f'generated::{model_name or "default"}'


class FakeQdrantClient:
    def __init__(self):
        self.calls = []

    def query_points(self, *, collection_name, query, limit, with_payload):
        self.calls.append(
            {
                'collection_name': collection_name,
                'query': query,
                'limit': limit,
                'with_payload': with_payload,
            }
        )
        return [
            SimpleNamespace(
                score=0.9,
                payload={
                    'text': f'{collection_name} content',
                    'metadata': {
                        'source': f'{collection_name}.pdf',
                        'page_number': 1,
                        'source_db': collection_name,
                    },
                },
            )
        ]


@pytest.fixture
def settings_with_domains():
    return Settings.from_env(
        {
            'DEFAULT_MODEL': 'global-default',
            'RERANKER_MODEL': 'global-reranker',
            'DOMAIN_PROFILES_JSON': '{"default_domain":"finance","domains":[{"name":"finance","is_default":true,"collections":[{"name":"finance_docs"},{"name":"finance_regs"}],"retrieval":{"top_k_per_collection":2,"final_top_k":3,"reranker_model":"finance-reranker"},"answering":{"model":"finance-model","system_prompt":"Finance only."}},{"name":"support","collections":[{"name":"support_docs"}],"retrieval":{"top_k_per_collection":1,"final_top_k":1},"answering":{"model":"support-model"}}]}'
        }
    )


@pytest.fixture
def settings_without_domains():
    settings = Settings.from_env(
        {
            'QDRANT_PRODUCT_COLLECTION': 'legacy_product',
            'QDRANT_REGULATORY_COLLECTION': 'legacy_regulatory',
            'RAG_TOP_K_PER_COLLECTION': '4',
            'RAG_FINAL_TOP_K': '6',
            'RERANKER_MODEL': 'legacy-reranker',
        }
    )
    return settings.__class__(
        **{
            **settings.__dict__,
            'domain_config': None,
            'domain_registry': None,
        }
    )


def build_engine(monkeypatch, settings):
    model_manager = FakeModelManager()
    fake_qdrant = FakeQdrantClient()

    monkeypatch.setattr('services.knowledge.rag_engine.QdrantClient', lambda url: fake_qdrant)

    engine = KnowledgeEngine(model_manager=model_manager, settings=settings)
    monkeypatch.setattr(engine, '_embed', lambda text: [0.1, 0.2])
    monkeypatch.setattr(engine, '_rerank', lambda query, hits, reranker_model=None: [dict(item, reranker_model=reranker_model) for item in hits])
    return engine, model_manager, fake_qdrant


def test_domain_aware_search_routes_to_domain_collections(monkeypatch, settings_with_domains):
    engine, _, fake_qdrant = build_engine(monkeypatch, settings_with_domains)

    hits = engine.search('what is apr?', domain='finance')

    assert [call['collection_name'] for call in fake_qdrant.calls] == ['finance_docs', 'finance_regs']
    assert all(call['limit'] == 2 for call in fake_qdrant.calls)
    assert len(hits) == 2
    assert {hit['collection'] for hit in hits} == {'finance_docs', 'finance_regs'}
    assert all(hit['reranker_model'] == 'finance-reranker' for hit in hits)


def test_domain_aware_answer_uses_domain_model_and_default_fallback(monkeypatch, settings_with_domains):
    engine, model_manager, fake_qdrant = build_engine(monkeypatch, settings_with_domains)

    result = engine.answer('help me', domain='support')

    assert [call['collection_name'] for call in fake_qdrant.calls] == ['support_docs']
    assert model_manager.calls[-1]['model_name'] == 'support-model'
    assert result['domain'] == 'support'
    assert result['model'] == 'support-model'


def test_search_without_domain_uses_default_domain(monkeypatch, settings_with_domains):
    engine, _, fake_qdrant = build_engine(monkeypatch, settings_with_domains)

    engine.search('default path')

    assert [call['collection_name'] for call in fake_qdrant.calls] == ['finance_docs', 'finance_regs']


def test_unknown_domain_raises_clear_error(monkeypatch, settings_with_domains):
    engine, _, _ = build_engine(monkeypatch, settings_with_domains)

    with pytest.raises(DomainNotFoundError, match="Unknown domain 'missing'"):
        engine.search('where', domain='missing')


def test_legacy_search_fallback_works_without_domain_registry(monkeypatch, settings_without_domains):
    engine, _, fake_qdrant = build_engine(monkeypatch, settings_without_domains)

    hits = engine.search('legacy path')

    assert [call['collection_name'] for call in fake_qdrant.calls] == ['legacy_product', 'legacy_regulatory']
    assert all(call['limit'] == 4 for call in fake_qdrant.calls)
    assert len(hits) == 2
    assert all(hit['reranker_model'] == 'legacy-reranker' for hit in hits)


def test_legacy_answer_fallback_works_without_domain_registry(monkeypatch, settings_without_domains):
    engine, model_manager, _ = build_engine(monkeypatch, settings_without_domains)

    result = engine.answer('legacy answer')

    assert model_manager.calls[-1]['model_name'] is None
    assert result['answer'] == 'generated::default'
    assert 'domain' not in result
    assert 'model' not in result


def test_explicit_domain_without_registry_fails_clearly(monkeypatch, settings_without_domains):
    engine, _, fake_qdrant = build_engine(monkeypatch, settings_without_domains)

    with pytest.raises(ValidationError, match='no domain registry is configured'):
        engine.search('legacy path', domain='finance')

    assert fake_qdrant.calls == []


def test_list_domains_reports_configured_domains(monkeypatch, settings_with_domains):
    engine, _, _ = build_engine(monkeypatch, settings_with_domains)

    domains = engine.list_domains()

    assert domains == [
        {
            'name': 'finance',
            'is_default': True,
            'collections': ['finance_docs', 'finance_regs'],
            'description': None,
            'ingestion': {'enabled': False, 'strategy': 'manual', 'target_collections': ['finance_docs', 'finance_regs']},
        },
        {
            'name': 'support',
            'is_default': False,
            'collections': ['support_docs'],
            'description': None,
            'ingestion': {'enabled': False, 'strategy': 'manual', 'target_collections': ['support_docs']},
        },
    ]
