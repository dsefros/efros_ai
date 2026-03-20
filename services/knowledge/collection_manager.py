from __future__ import annotations

from dataclasses import dataclass

from qdrant_client.models import Distance, VectorParams

from configs.domain_profiles import CollectionProfile, DomainProfile
from kernel.exceptions import KnowledgeError


class CollectionPolicyError(KnowledgeError):
    """Collection lifecycle policy failed for a domain."""


@dataclass(frozen=True)
class CollectionStatus:
    domain: str
    name: str
    exists: bool
    created: bool
    create_if_missing: bool
    fail_if_missing: bool
    role: str


class CollectionManager:
    def __init__(self, qdrant_client):
        self.qdrant = qdrant_client

    def list_existing(self) -> set[str]:
        response = self.qdrant.get_collections()
        collections = getattr(response, "collections", response)
        names: set[str] = set()
        for item in collections:
            name = getattr(item, "name", None)
            if isinstance(item, dict):
                name = item.get("name", name)
            if name:
                names.add(name)
        return names

    def ensure_domain_collections(self, domain: DomainProfile) -> list[CollectionStatus]:
        existing = self.list_existing()
        statuses: list[CollectionStatus] = []
        for collection in domain.collections:
            statuses.append(self._ensure_collection(domain, collection, existing))
        return statuses

    def _ensure_collection(
        self,
        domain: DomainProfile,
        collection: CollectionProfile,
        existing: set[str],
    ) -> CollectionStatus:
        if collection.name in existing:
            return CollectionStatus(
                domain=domain.name,
                name=collection.name,
                exists=True,
                created=False,
                create_if_missing=collection.create_if_missing,
                fail_if_missing=collection.fail_if_missing,
                role=collection.role,
            )

        if collection.create_if_missing:
            self._create_collection(collection)
            existing.add(collection.name)
            return CollectionStatus(
                domain=domain.name,
                name=collection.name,
                exists=True,
                created=True,
                create_if_missing=True,
                fail_if_missing=collection.fail_if_missing,
                role=collection.role,
            )

        if collection.fail_if_missing:
            raise CollectionPolicyError(
                f"Domain {domain.name!r} requires Qdrant collection {collection.name!r}, but it is missing and auto-create is disabled"
            )

        return CollectionStatus(
            domain=domain.name,
            name=collection.name,
            exists=False,
            created=False,
            create_if_missing=False,
            fail_if_missing=False,
            role=collection.role,
        )

    def _create_collection(self, collection: CollectionProfile) -> None:
        if collection.vector_size is None:
            raise CollectionPolicyError(
                f"Collection {collection.name!r} cannot be auto-created because vector_size is not configured"
            )
        distance_name = collection.distance or "Cosine"
        try:
            distance = getattr(Distance, distance_name.upper())
        except AttributeError as exc:
            raise CollectionPolicyError(
                f"Collection {collection.name!r} configured unsupported distance {distance_name!r}"
            ) from exc
        self.qdrant.create_collection(
            collection_name=collection.name,
            vectors_config=VectorParams(size=collection.vector_size, distance=distance),
        )
