from types import SimpleNamespace

import pytest

from configs.settings import Settings
from kernel.exceptions import DomainNotFoundError, ValidationError
from services.knowledge.collection_manager import CollectionManager, CollectionPolicyError
from services.knowledge.ingestion import DomainIngestionService
from services.knowledge.rag_engine import KnowledgeEngine


class FakeQdrantCollectionsClient:
    def __init__(self, existing=None):
        self.existing = list(existing or [])
        self.created = []
        self.query_calls = []

    def get_collections(self):
        return SimpleNamespace(collections=[SimpleNamespace(name=name) for name in self.existing])

    def create_collection(self, *, collection_name, vectors_config):
        self.created.append(
            {
                'collection_name': collection_name,
                'size': vectors_config.size,
                'distance': getattr(vectors_config.distance, 'name', str(vectors_config.distance)),
            }
        )
        if collection_name not in self.existing:
            self.existing.append(collection_name)

    def query_points(self, *, collection_name, query, limit, with_payload):
        self.query_calls.append(collection_name)
        return [
            SimpleNamespace(
                score=0.5,
                payload={
                    'text': f'{collection_name} text',
                    'metadata': {'source': f'{collection_name}.pdf', 'page_number': 1, 'source_db': collection_name},
                },
            )
        ]


class FakeModelManager:
    def generate(self, prompt: str, model_name: str | None = None):
        return f'generated::{model_name or "default"}'


@pytest.fixture

def collection_settings():
    return Settings.from_env(
        {
            'DOMAIN_PROFILES_JSON': (
                '{"default_domain":"finance","domains":['
                '{"name":"finance","is_default":true,'
                '"collections":['
                '{"name":"finance_docs","role":"primary","fail_if_missing":true},'
                '{"name":"finance_archive","role":"archive","create_if_missing":true,"vector_size":768,"distance":"Cosine"}'
                '],'
                '"retrieval":{"top_k_per_collection":2,"final_top_k":2},'
                '"ingestion":{"enabled":true,"strategy":"manual","source_type":"files","target_collections":["finance_docs","finance_archive"],"default_chunk_size":800,"default_chunk_overlap":100}'
                '},'
                '{"name":"support",'
                '"collections":[{"name":"support_docs","fail_if_missing":false}],'
                '"retrieval":{"top_k_per_collection":1,"final_top_k":1},'
                '"ingestion":{"enabled":false,"strategy":"manual"}'
                '}'
                ']}'
            )
        }
    )


def build_engine(monkeypatch, settings, fake_qdrant):
    monkeypatch.setattr('services.knowledge.rag_engine.QdrantClient', lambda url: fake_qdrant)
    engine = KnowledgeEngine(model_manager=FakeModelManager(), settings=settings)
    monkeypatch.setattr(engine, '_embed', lambda text: [0.1, 0.2])
    monkeypatch.setattr(engine, '_rerank', lambda query, hits, reranker_model=None: hits)
    return engine


def test_collection_manager_fails_when_required_collection_missing(collection_settings):
    registry = collection_settings.domain_registry
    assert registry is not None
    manager = CollectionManager(FakeQdrantCollectionsClient(existing=[]))

    with pytest.raises(CollectionPolicyError, match="finance_docs"):
        manager.ensure_domain_collections(registry.get('finance'))


def test_collection_manager_auto_creates_when_enabled(collection_settings):
    registry = collection_settings.domain_registry
    assert registry is not None
    fake_qdrant = FakeQdrantCollectionsClient(existing=['finance_docs'])
    manager = CollectionManager(fake_qdrant)

    statuses = manager.ensure_domain_collections(registry.get('finance'))

    assert [(status.name, status.exists, status.created) for status in statuses] == [
        ('finance_docs', True, False),
        ('finance_archive', True, True),
    ]
    assert fake_qdrant.created == [{'collection_name': 'finance_archive', 'size': 768, 'distance': 'COSINE'}]


def test_collection_manager_allows_missing_optional_collection(collection_settings):
    registry = collection_settings.domain_registry
    assert registry is not None
    manager = CollectionManager(FakeQdrantCollectionsClient(existing=[]))

    statuses = manager.ensure_domain_collections(registry.get('support'))

    assert len(statuses) == 1
    assert statuses[0].name == 'support_docs'
    assert statuses[0].exists is False
    assert statuses[0].created is False


def test_ingestion_service_builds_domain_to_collection_plan(collection_settings):
    registry = collection_settings.domain_registry
    assert registry is not None
    service = DomainIngestionService(registry)

    plan = service.build_plan('finance', metadata={'request_id': 'abc'})

    assert plan.domain == 'finance'
    assert plan.strategy == 'manual'
    assert plan.source_type == 'files'
    assert [collection.name for collection in plan.target_collections] == ['finance_docs', 'finance_archive']
    assert plan.default_chunk_size == 800
    assert plan.default_chunk_overlap == 100
    assert plan.metadata == {'request_id': 'abc'}


def test_ingestion_profile_rejects_unknown_target_collection():
    with pytest.raises(Exception, match='target_collections must reference configured domain collections'):
        Settings.from_env(
            {
                'DOMAIN_PROFILES_JSON': (
                    '{"domains":[{"name":"finance","collections":[{"name":"finance_docs"}],'
                    '"retrieval":{"top_k_per_collection":1,"final_top_k":1},'
                    '"ingestion":{"target_collections":["missing"]}}]}'
                )
            }
        )


def test_knowledge_engine_exposes_collection_policy_and_ingestion_foundation(monkeypatch, collection_settings):
    fake_qdrant = FakeQdrantCollectionsClient(existing=['finance_docs'])
    engine = build_engine(monkeypatch, collection_settings, fake_qdrant)

    statuses = engine.ensure_domain_collections('finance')
    plan = engine.plan_ingestion('finance', metadata={'source': 'seed'})

    assert [status.name for status in statuses] == ['finance_docs', 'finance_archive']
    assert statuses[1].created is True
    assert plan['target_collections'] == ['finance_docs', 'finance_archive']
    assert plan['status'] == 'planned'
    assert plan['metadata'] == {'source': 'seed'}


def test_ingestion_without_registry_fails_clearly():
    settings = Settings.from_env({})
    settings = settings.__class__(**{**settings.__dict__, 'domain_registry': None, 'domain_config': None})
    service = DomainIngestionService(settings.domain_registry)

    with pytest.raises(ValidationError, match='no domain registry is configured'):
        service.build_plan()


def test_ingestion_unexpected_domain_resolution_error_is_not_masked():
    class BrokenRegistry:
        def get(self, name=None):
            raise RuntimeError('registry unavailable')

    service = DomainIngestionService(BrokenRegistry())

    with pytest.raises(RuntimeError, match='registry unavailable'):
        service.build_plan('finance')

def test_ingestion_unknown_domain_fails_clearly(collection_settings):
    registry = collection_settings.domain_registry
    assert registry is not None
    service = DomainIngestionService(registry)

    with pytest.raises(DomainNotFoundError, match="Unknown domain 'missing'"):
        service.build_plan('missing')
