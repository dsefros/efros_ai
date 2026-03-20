from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Mapping


_DOMAIN_NAME_RE = re.compile(r"^[a-z0-9](?:[a-z0-9_-]{0,62}[a-z0-9])?$")
_ALLOWED_DISTANCE_VALUES = {"Cosine", "Dot", "Euclid", "Manhattan"}


class DomainProfileError(ValueError):
    """Raised when domain profile configuration is invalid."""


@dataclass(frozen=True)
class CollectionProfile:
    name: str
    role: str = "knowledge"
    vector_size: int | None = None
    distance: str | None = None
    create_if_missing: bool = False
    fail_if_missing: bool = True
    description: str | None = None
    metadata: Mapping[str, str] | None = None

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any], *, path: str) -> "CollectionProfile":
        name = _required_str(data, "name", path=path)
        role = _optional_str(data, "role", default="knowledge", path=path)
        vector_size = _optional_int(data, "vector_size", path=path, minimum=1)
        distance = _optional_str(data, "distance", default=None, path=path)
        if distance is not None and distance not in _ALLOWED_DISTANCE_VALUES:
            raise DomainProfileError(
                f"{path}.distance must be one of {sorted(_ALLOWED_DISTANCE_VALUES)}, got {distance!r}"
            )
        create_if_missing = _optional_bool(data, "create_if_missing", default=False, path=path)
        fail_if_missing = _optional_bool(data, "fail_if_missing", default=True, path=path)
        if create_if_missing and vector_size is None:
            raise DomainProfileError(f"{path}.vector_size is required when create_if_missing=true")
        description = _optional_str(data, "description", default=None, path=path)
        metadata = _optional_str_mapping(data, "metadata", path=path)
        return cls(
            name=name,
            role=role,
            vector_size=vector_size,
            distance=distance,
            create_if_missing=create_if_missing,
            fail_if_missing=fail_if_missing,
            description=description,
            metadata=metadata,
        )


@dataclass(frozen=True)
class RetrievalProfile:
    top_k_per_collection: int
    final_top_k: int
    reranker_model: str | None = None

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any], *, path: str) -> "RetrievalProfile":
        top_k_per_collection = _required_int(data, "top_k_per_collection", path=path, minimum=1)
        final_top_k = _required_int(data, "final_top_k", path=path, minimum=1)
        reranker_model = _optional_str(data, "reranker_model", default=None, path=path)
        return cls(
            top_k_per_collection=top_k_per_collection,
            final_top_k=final_top_k,
            reranker_model=reranker_model,
        )


@dataclass(frozen=True)
class AnsweringProfile:
    model: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    system_prompt: str | None = None

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any], *, path: str) -> "AnsweringProfile":
        model = _optional_str(data, "model", default=None, path=path)
        temperature = _optional_float(data, "temperature", path=path, minimum=0.0)
        max_tokens = _optional_int(data, "max_tokens", path=path, minimum=1)
        system_prompt = _optional_str(data, "system_prompt", default=None, path=path)
        return cls(model=model, temperature=temperature, max_tokens=max_tokens, system_prompt=system_prompt)


@dataclass(frozen=True)
class IngestionProfile:
    enabled: bool = False
    strategy: str = "manual"
    source_type: str | None = None
    target_collections: tuple[str, ...] = ()
    default_chunk_size: int | None = None
    default_chunk_overlap: int = 0

    @classmethod
    def from_mapping(
        cls,
        data: Mapping[str, Any],
        *,
        path: str,
        domain_collections: tuple[CollectionProfile, ...] = (),
    ) -> "IngestionProfile":
        enabled = _optional_bool(data, "enabled", default=False, path=path)
        strategy = _optional_str(data, "strategy", default="manual", path=path)
        source_type = _optional_str(data, "source_type", default=None, path=path)
        target_collections = _optional_str_list(data, "target_collections", path=path)
        default_chunk_size = _optional_int(data, "default_chunk_size", path=path, minimum=1)
        default_chunk_overlap = _optional_int(data, "default_chunk_overlap", path=path, minimum=0) or 0
        if default_chunk_size is not None and default_chunk_overlap >= default_chunk_size:
            raise DomainProfileError(f"{path}.default_chunk_overlap must be lower than default_chunk_size")

        available_collections = {collection.name for collection in domain_collections}
        if target_collections:
            unknown = sorted(set(target_collections) - available_collections)
            if unknown:
                raise DomainProfileError(
                    f"{path}.target_collections must reference configured domain collections, unknown={unknown}"
                )
        else:
            target_collections = tuple(collection.name for collection in domain_collections)

        return cls(
            enabled=enabled,
            strategy=strategy,
            source_type=source_type,
            target_collections=target_collections,
            default_chunk_size=default_chunk_size,
            default_chunk_overlap=default_chunk_overlap,
        )


@dataclass(frozen=True)
class AccessProfile:
    visibility: str = "internal"
    default_action: str = "deny"
    allowed_roles: tuple[str, ...] = ()
    allowed_callers: tuple[str, ...] = ()
    allowed_groups: tuple[str, ...] = ()

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any], *, path: str) -> "AccessProfile":
        visibility = _optional_str(data, "visibility", default="internal", path=path)
        if visibility not in {"public", "internal", "private"}:
            raise DomainProfileError(f"{path}.visibility must be one of ['internal', 'private', 'public']")

        explicit_default = _optional_str(data, "default_action", default=None, path=path)
        if explicit_default is None:
            default_action = "allow" if visibility == "public" else "deny"
        else:
            default_action = explicit_default
        if default_action not in {"allow", "deny"}:
            raise DomainProfileError(f"{path}.default_action must be one of ['allow', 'deny']")

        required_roles = _optional_str_list(data, "required_roles", path=path)
        allowed_roles = _merge_unique(required_roles, _optional_str_list(data, "allowed_roles", path=path))
        allowed_callers = _optional_str_list(data, "allowed_callers", path=path)
        allowed_groups = _optional_str_list(data, "allowed_groups", path=path)
        return cls(
            visibility=visibility,
            default_action=default_action,
            allowed_roles=allowed_roles,
            allowed_callers=allowed_callers,
            allowed_groups=allowed_groups,
        )

    @property
    def required_roles(self) -> tuple[str, ...]:
        return self.allowed_roles


@dataclass(frozen=True)
class DomainProfile:
    name: str
    collections: tuple[CollectionProfile, ...]
    retrieval: RetrievalProfile
    answering: AnsweringProfile
    ingestion: IngestionProfile
    access: AccessProfile
    is_default: bool = False
    description: str | None = None

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any], *, path: str) -> "DomainProfile":
        name = _required_str(data, "name", path=path)
        _validate_domain_name(name, path=f"{path}.name")

        collections_raw = data.get("collections")
        if not isinstance(collections_raw, list) or not collections_raw:
            raise DomainProfileError(f"{path}.collections must be a non-empty list")
        collections = tuple(
            CollectionProfile.from_mapping(item, path=f"{path}.collections[{index}]")
            for index, item in enumerate(collections_raw)
        )
        collection_names = [collection.name for collection in collections]
        if len(collection_names) != len(set(collection_names)):
            raise DomainProfileError(f"{path}.collections contains duplicate collection names")

        retrieval_raw = data.get("retrieval")
        if not isinstance(retrieval_raw, Mapping):
            raise DomainProfileError(f"{path}.retrieval must be an object")

        return cls(
            name=name,
            collections=collections,
            retrieval=RetrievalProfile.from_mapping(retrieval_raw, path=f"{path}.retrieval"),
            answering=AnsweringProfile.from_mapping(_mapping_or_empty(data.get("answering"), path=f"{path}.answering"), path=f"{path}.answering"),
            ingestion=IngestionProfile.from_mapping(
                _mapping_or_empty(data.get("ingestion"), path=f"{path}.ingestion"),
                path=f"{path}.ingestion",
                domain_collections=collections,
            ),
            access=AccessProfile.from_mapping(_mapping_or_empty(data.get("access"), path=f"{path}.access"), path=f"{path}.access"),
            is_default=_optional_bool(data, "is_default", default=False, path=path),
            description=_optional_str(data, "description", default=None, path=path),
        )

    @property
    def collection_names(self) -> tuple[str, ...]:
        return tuple(collection.name for collection in self.collections)

    def get_collection(self, name: str) -> CollectionProfile:
        for collection in self.collections:
            if collection.name == name:
                return collection
        raise DomainProfileError(f"Domain {self.name!r} does not define collection {name!r}")


@dataclass(frozen=True)
class DomainConfiguration:
    default_domain: str
    domains: tuple[DomainProfile, ...]

    @classmethod
    def from_mapping(
        cls,
        data: Mapping[str, Any],
        *,
        fallback_default_domain: str,
    ) -> "DomainConfiguration":
        domains_raw = data.get("domains")
        if not isinstance(domains_raw, list) or not domains_raw:
            raise DomainProfileError("domains.domains must be a non-empty list")

        domains = tuple(
            DomainProfile.from_mapping(item, path=f"domains.domains[{index}]")
            for index, item in enumerate(domains_raw)
        )
        names = [domain.name for domain in domains]
        if len(names) != len(set(names)):
            raise DomainProfileError("domains.domains contains duplicate domain names")

        explicit_default = data.get("default_domain")
        flagged_defaults = [domain.name for domain in domains if domain.is_default]
        if explicit_default is not None and not isinstance(explicit_default, str):
            raise DomainProfileError("domains.default_domain must be a string")
        if len(flagged_defaults) > 1:
            raise DomainProfileError("Only one domain may set is_default=true")

        default_domain = explicit_default or (flagged_defaults[0] if flagged_defaults else fallback_default_domain)
        if default_domain not in names:
            raise DomainProfileError(
                f"domains.default_domain must reference one of {sorted(names)}, got {default_domain!r}"
            )
        return cls(default_domain=default_domain, domains=domains)


@dataclass(frozen=True)
class DomainRegistry:
    domains: tuple[DomainProfile, ...]
    default_domain_name: str

    def __post_init__(self) -> None:
        if not self.domains:
            raise DomainProfileError("DomainRegistry requires at least one domain")
        names = [domain.name for domain in self.domains]
        if len(names) != len(set(names)):
            raise DomainProfileError("DomainRegistry domain names must be unique")
        if self.default_domain_name not in names:
            raise DomainProfileError(
                f"DomainRegistry default domain must be one of {sorted(names)}, got {self.default_domain_name!r}"
            )

    @classmethod
    def from_configuration(cls, configuration: DomainConfiguration) -> "DomainRegistry":
        return cls(domains=configuration.domains, default_domain_name=configuration.default_domain)

    def get(self, name: str | None = None) -> DomainProfile:
        selected = name or self.default_domain_name
        for domain in self.domains:
            if domain.name == selected:
                return domain
        raise DomainProfileError(f"Unknown domain {selected!r}")

    def list_names(self) -> list[str]:
        return [domain.name for domain in self.domains]

    @property
    def default_domain(self) -> DomainProfile:
        return self.get(self.default_domain_name)


def default_domain_configuration(*, product_collection: str, regulatory_collection: str, top_k_per_collection: int, final_top_k: int) -> DomainConfiguration:
    return DomainConfiguration(
        default_domain="default",
        domains=(
            DomainProfile(
                name="default",
                collections=(
                    CollectionProfile(name=product_collection, role="product"),
                    CollectionProfile(name=regulatory_collection, role="regulatory"),
                ),
                retrieval=RetrievalProfile(
                    top_k_per_collection=top_k_per_collection,
                    final_top_k=final_top_k,
                ),
                answering=AnsweringProfile(),
                ingestion=IngestionProfile(target_collections=(product_collection, regulatory_collection)),
                access=AccessProfile(),
                is_default=True,
                description="Compatibility domain synthesized from legacy RAG settings.",
            ),
        ),
    )


def load_domain_configuration(raw_value: str | None, *, fallback_default_domain: str, product_collection: str, regulatory_collection: str, top_k_per_collection: int, final_top_k: int) -> DomainConfiguration:
    if raw_value is None or not raw_value.strip():
        return default_domain_configuration(
            product_collection=product_collection,
            regulatory_collection=regulatory_collection,
            top_k_per_collection=top_k_per_collection,
            final_top_k=final_top_k,
        )

    try:
        parsed = json.loads(raw_value)
    except json.JSONDecodeError as exc:
        raise DomainProfileError(f"DOMAIN_PROFILES_JSON must be valid JSON: {exc.msg}") from exc

    if not isinstance(parsed, Mapping):
        raise DomainProfileError("DOMAIN_PROFILES_JSON must decode to an object")

    return DomainConfiguration.from_mapping(parsed, fallback_default_domain=fallback_default_domain)


def _mapping_or_empty(value: Any, *, path: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise DomainProfileError(f"{path} must be an object")
    return value


def _validate_domain_name(value: str, *, path: str) -> None:
    if not _DOMAIN_NAME_RE.match(value):
        raise DomainProfileError(
            f"{path} must match {_DOMAIN_NAME_RE.pattern!r}, got {value!r}"
        )


def _required_str(data: Mapping[str, Any], key: str, *, path: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise DomainProfileError(f"{path}.{key} must be a non-empty string")
    return value.strip()


def _optional_str(data: Mapping[str, Any], key: str, *, default: str | None, path: str) -> str | None:
    value = data.get(key)
    if value is None:
        return default
    if not isinstance(value, str) or not value.strip():
        raise DomainProfileError(f"{path}.{key} must be a non-empty string when provided")
    return value.strip()


def _required_int(data: Mapping[str, Any], key: str, *, path: str, minimum: int | None = None) -> int:
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise DomainProfileError(f"{path}.{key} must be an integer")
    if minimum is not None and value < minimum:
        raise DomainProfileError(f"{path}.{key} must be >= {minimum}")
    return value


def _optional_int(data: Mapping[str, Any], key: str, *, path: str, minimum: int | None = None) -> int | None:
    value = data.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise DomainProfileError(f"{path}.{key} must be an integer")
    if minimum is not None and value < minimum:
        raise DomainProfileError(f"{path}.{key} must be >= {minimum}")
    return value


def _optional_float(data: Mapping[str, Any], key: str, *, path: str, minimum: float | None = None) -> float | None:
    value = data.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DomainProfileError(f"{path}.{key} must be a number")
    parsed = float(value)
    if minimum is not None and parsed < minimum:
        raise DomainProfileError(f"{path}.{key} must be >= {minimum}")
    return parsed


def _optional_bool(data: Mapping[str, Any], key: str, *, default: bool, path: str) -> bool:
    value = data.get(key)
    if value is None:
        return default
    if not isinstance(value, bool):
        raise DomainProfileError(f"{path}.{key} must be a boolean")
    return value


def _optional_str_list(data: Mapping[str, Any], key: str, *, path: str) -> tuple[str, ...]:
    value = data.get(key)
    if value is None:
        return ()
    if not isinstance(value, list):
        raise DomainProfileError(f"{path}.{key} must be a list of strings")
    values: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str) or not item.strip():
            raise DomainProfileError(f"{path}.{key}[{index}] must be a non-empty string")
        values.append(item.strip())
    return tuple(values)


def _merge_unique(*groups: tuple[str, ...]) -> tuple[str, ...]:
    merged: list[str] = []
    seen: set[str] = set()
    for group in groups:
        for item in group:
            if item in seen:
                continue
            seen.add(item)
            merged.append(item)
    return tuple(merged)


def _optional_str_mapping(data: Mapping[str, Any], key: str, *, path: str) -> Mapping[str, str] | None:
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise DomainProfileError(f"{path}.{key} must be an object of string values")
    normalized: dict[str, str] = {}
    for item_key, item_value in value.items():
        if not isinstance(item_key, str) or not item_key.strip():
            raise DomainProfileError(f"{path}.{key} keys must be non-empty strings")
        if not isinstance(item_value, str) or not item_value.strip():
            raise DomainProfileError(f"{path}.{key}[{item_key!r}] must be a non-empty string")
        normalized[item_key.strip()] = item_value.strip()
    return normalized
