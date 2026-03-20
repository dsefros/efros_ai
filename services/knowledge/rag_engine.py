from __future__ import annotations

from qdrant_client import QdrantClient

from configs.domain_profiles import DomainProfileError
from services.knowledge.collection_manager import CollectionManager
from services.knowledge.access_policy import DomainAccessPolicyEvaluator
from services.knowledge.ingestion import DomainIngestionService
from configs.settings import Settings, load_settings
from kernel.exceptions import DomainNotFoundError, QdrantSearchError, RerankerError, ValidationError


class KnowledgeEngine:
    def __init__(self, model_manager, qdrant_url: str | None = None, settings: Settings | None = None):
        self.settings = settings or load_settings()
        self.model_manager = model_manager
        self.qdrant = QdrantClient(url=qdrant_url or self.settings.qdrant_url)
        self.product_collection = self.settings.qdrant_product_collection
        self.regulatory_collection = self.settings.qdrant_regulatory_collection
        self._embedder = None
        self._reranker = None
        self.collection_manager = CollectionManager(self.qdrant)
        self.ingestion = DomainIngestionService(self.settings.domain_registry)
        self.access_policy = DomainAccessPolicyEvaluator(self.settings.domain_registry)

    def _get_embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer

            self._embedder = SentenceTransformer(self.settings.embedding_model)
        return self._embedder

    def _get_reranker(self, reranker_model: str | None = None):
        selected_model = reranker_model or self.settings.reranker_model
        if self._reranker is None or getattr(self._reranker, '_model_name', None) != selected_model:
            from sentence_transformers.cross_encoder import CrossEncoder

            self._reranker = CrossEncoder(selected_model)
            setattr(self._reranker, '_model_name', selected_model)
        return self._reranker

    def _embed(self, text: str):
        embedder = self._get_embedder()
        vector = embedder.encode(text)
        if hasattr(vector, 'tolist'):
            vector = vector.tolist()
        return vector

    def _resolve_domain(self, domain: str | None = None):
        registry = self.settings.domain_registry
        if registry is None:
            if domain is not None:
                raise ValidationError('Domain selection is unavailable because no domain registry is configured')
            return None
        try:
            return self.access_policy.resolve_domain(domain)
        except DomainProfileError as exc:
            raise DomainNotFoundError(str(exc)) from exc

    def list_domains(self) -> list[dict]:
        registry = self.settings.domain_registry
        if registry is None:
            return []
        return [
            {
                'name': domain.name,
                'is_default': domain.name == registry.default_domain_name,
                'collections': list(domain.collection_names),
                'description': domain.description,
                'ingestion': {
                    'enabled': domain.ingestion.enabled,
                    'strategy': domain.ingestion.strategy,
                    'target_collections': list(domain.ingestion.target_collections),
                },
                'access': {
                    'visibility': domain.access.visibility,
                },
            }
            for domain in registry.domains
        ]


    def ensure_domain_collections(self, domain: str | None = None):
        domain_profile = self._resolve_domain(domain)
        if domain_profile is None:
            raise ValidationError('Collection policy enforcement is unavailable because no domain registry is configured')
        return self.collection_manager.ensure_domain_collections(domain_profile)

    def plan_ingestion(self, domain: str | None = None, metadata: dict[str, str] | None = None):
        return self.ingestion.ingest(domain=domain, metadata=metadata)

    def _normalize_hits(self, response, collection_name: str):
        points = []

        if hasattr(response, 'points'):
            points = response.points
        elif isinstance(response, list):
            points = response

        normalized = []
        for item in points:
            payload = getattr(item, 'payload', None) or {}
            score = float(getattr(item, 'score', 0.0) or 0.0)

            text = (
                payload.get('text')
                or payload.get('chunk_text')
                or payload.get('content')
                or payload.get('document')
                or ''
            )

            metadata = payload.get('metadata', {})
            if not isinstance(metadata, dict):
                metadata = {'value': metadata}

            normalized.append(
                {
                    'score': score,
                    'text': text,
                    'metadata': metadata,
                    'collection': collection_name,
                    'payload': payload,
                }
            )

        return normalized

    def _search_collection(self, collection_name: str, query: str, limit: int):
        try:
            vector = self._embed(query)

            response = self.qdrant.query_points(
                collection_name=collection_name,
                query=vector,
                limit=limit,
                with_payload=True,
            )

            return self._normalize_hits(response, collection_name)
        except Exception as e:
            raise QdrantSearchError(f'Failed to query Qdrant collection {collection_name}: {e}') from e

    def _rerank(self, query: str, hits: list[dict], reranker_model: str | None = None):
        candidates = [h for h in hits if h.get('text', '').strip()]
        if not candidates:
            return hits

        try:
            reranker = self._get_reranker(reranker_model=reranker_model)
            pairs = [(query, item['text']) for item in candidates]
            scores = reranker.predict(pairs)

            for item, rr_score in zip(candidates, scores):
                item['rerank_score'] = float(rr_score)

            candidates.sort(
                key=lambda x: (x.get('rerank_score', -1e9), x.get('score', 0.0)),
                reverse=True,
            )

            empty = [h for h in hits if not h.get('text', '').strip()]
            return candidates + empty
        except Exception as e:
            raise RerankerError(f'Failed to rerank results: {e}') from e

    def _compact_sources(self, items: list[dict]):
        compact = []
        for item in items:
            meta = item.get('metadata', {}) or {}
            compact.append(
                {
                    'score': round(float(item.get('score', 0.0)), 6),
                    'rerank_score': round(float(item.get('rerank_score', 0.0)), 6) if 'rerank_score' in item else None,
                    'text': item.get('text', ''),
                    'source': meta.get('source'),
                    'page_number': meta.get('page_number'),
                    'doc_type': meta.get('doc_type'),
                    'source_db': meta.get('source_db'),
                    'collection': item.get('collection'),
                }
            )
        return compact

    def search(self, query: str, domain: str | None = None, limit_per_collection: int | None = None):
        if not query or not query.strip():
            raise ValidationError('Query must not be empty')

        domain_profile = self._resolve_domain(domain)
        if domain_profile is None:
            search_limit = limit_per_collection or self.settings.rag_top_k_per_collection
            combined = self._search_collection(self.product_collection, query, search_limit)
            combined.extend(self._search_collection(self.regulatory_collection, query, search_limit))
            combined.sort(key=lambda x: x['score'], reverse=True)

            reranked = self._rerank(query, combined, reranker_model=self.settings.reranker_model)
            return reranked[: self.settings.rag_final_top_k]

        search_limit = limit_per_collection or domain_profile.retrieval.top_k_per_collection
        combined = []
        for collection_name in domain_profile.collection_names:
            combined.extend(self._search_collection(collection_name, query, search_limit))
        combined.sort(key=lambda x: x['score'], reverse=True)

        reranked = self._rerank(query, combined, reranker_model=domain_profile.retrieval.reranker_model)
        return reranked[: domain_profile.retrieval.final_top_k]

    def answer(self, query: str, domain: str | None = None, model_name: str | None = None):
        domain_profile = self._resolve_domain(domain)
        context = self.search(query, domain=domain)

        if not context:
            return {'answer': 'Нет релевантных документов', 'sources': []}

        context_items = []
        for i, item in enumerate(context, start=1):
            text = item.get('text', '').strip()
            meta = item.get('metadata', {}) or {}
            source = meta.get('source', 'unknown')
            page = meta.get('page_number', '?')
            if text:
                context_items.append(f'[{i}] Источник: {source}, стр. {page}\n{text}')

        context_text = '\n\n'.join(context_items).strip()

        if not context_text:
            return {
                'answer': 'Документы найдены, но текстовых полей в payload не обнаружено. Нужно проверить структуру данных в Qdrant.',
                'sources': self._compact_sources(context),
            }

        prompt_strategy = (
            domain_profile.answering.system_prompt
            if domain_profile is not None and domain_profile.answering.system_prompt
            else (
                'Ответь только на основе контекста ниже.\n'
                'Если данных в контексте недостаточно, прямо напиши, чего именно не хватает.\n'
                'Ничего не выдумывай.'
            )
        )

        prompt = f"""
{prompt_strategy}

Сформируй ответ в такой структуре:
1. Краткое определение
2. Как это работает
3. Основные сущности и данные
4. Интеграции и взаимодействие с внешними системами
5. Ограничения / что неясно из контекста

Вопрос:
{query}

Контекст:
{context_text}
""".strip()

        selected_model = model_name or (domain_profile.answering.model if domain_profile is not None else None)
        answer = self.model_manager.generate(prompt, model_name=selected_model)

        response = {
            'answer': answer,
            'sources': self._compact_sources(context),
        }
        if domain_profile is not None:
            response['domain'] = domain_profile.name
            response['model'] = selected_model or self.settings.default_model
        return response
