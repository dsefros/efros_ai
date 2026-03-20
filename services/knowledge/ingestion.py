from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from configs.domain_profiles import CollectionProfile, DomainProfile, DomainProfileError, DomainRegistry
from kernel.exceptions import DomainNotFoundError, ValidationError


@dataclass(frozen=True)
class IngestionPlan:
    domain: str
    strategy: str
    source_type: str | None
    target_collections: tuple[CollectionProfile, ...]
    default_chunk_size: int | None
    default_chunk_overlap: int
    metadata: dict[str, str]


class IngestionService(Protocol):
    def build_plan(self, domain: str | None = None, metadata: dict[str, str] | None = None) -> IngestionPlan:
        ...


class DomainIngestionService:
    def __init__(self, domain_registry: DomainRegistry | None):
        self.domain_registry = domain_registry

    def build_plan(self, domain: str | None = None, metadata: dict[str, str] | None = None) -> IngestionPlan:
        profile = self._resolve_domain(domain)
        ingestion = profile.ingestion
        target_names = ingestion.target_collections or profile.collection_names
        target_collections = tuple(profile.get_collection(name) for name in target_names)
        return IngestionPlan(
            domain=profile.name,
            strategy=ingestion.strategy,
            source_type=ingestion.source_type,
            target_collections=target_collections,
            default_chunk_size=ingestion.default_chunk_size,
            default_chunk_overlap=ingestion.default_chunk_overlap,
            metadata=dict(metadata or {}),
        )

    def ingest(self, domain: str | None = None, metadata: dict[str, str] | None = None) -> dict:
        plan = self.build_plan(domain=domain, metadata=metadata)
        return {
            'domain': plan.domain,
            'strategy': plan.strategy,
            'source_type': plan.source_type,
            'target_collections': [collection.name for collection in plan.target_collections],
            'default_chunk_size': plan.default_chunk_size,
            'default_chunk_overlap': plan.default_chunk_overlap,
            'metadata': plan.metadata,
            'status': 'planned',
        }

    def _resolve_domain(self, domain: str | None) -> DomainProfile:
        if self.domain_registry is None:
            raise ValidationError('Ingestion is unavailable because no domain registry is configured')
        try:
            return self.domain_registry.get(domain)
        except DomainProfileError as exc:
            raise DomainNotFoundError(str(exc)) from exc
