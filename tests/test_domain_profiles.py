import json
import sys
import types

import pytest

sys.modules.setdefault("dotenv", types.SimpleNamespace(load_dotenv=lambda *args, **kwargs: False))

from configs.domain_profiles import DomainConfiguration, DomainProfileError, DomainRegistry, load_domain_configuration
from configs.settings import Settings, SettingsError


def test_settings_builds_compatibility_default_domain_from_legacy_rag_fields():
    settings = Settings.from_env({
        "QDRANT_PRODUCT_COLLECTION": "products_a",
        "QDRANT_REGULATORY_COLLECTION": "regulatory_b",
        "RAG_TOP_K_PER_COLLECTION": "4",
        "RAG_FINAL_TOP_K": "8",
    })

    assert settings.default_domain_name == "default"
    assert settings.domain_registry is not None
    assert settings.domain_registry.list_names() == ["default"]
    domain = settings.domain_registry.default_domain
    assert domain.collection_names == ("products_a", "regulatory_b")
    assert domain.retrieval.top_k_per_collection == 4
    assert domain.retrieval.final_top_k == 8


def test_settings_loads_explicit_domain_profiles_from_json():
    raw = json.dumps({
        "default_domain": "finance",
        "domains": [
            {
                "name": "finance",
                "is_default": True,
                "collections": [
                    {"name": "finance_docs", "role": "primary"},
                    {"name": "finance_rules", "role": "regulatory"},
                ],
                "retrieval": {"top_k_per_collection": 3, "final_top_k": 6},
                "answering": {"model": "ministral", "temperature": 0.1, "max_tokens": 512},
                "ingestion": {"enabled": False, "strategy": "manual"},
                "access": {"visibility": "internal", "required_roles": ["analyst"]},
            },
            {
                "name": "support",
                "collections": [{"name": "support_docs"}],
                "retrieval": {"top_k_per_collection": 2, "final_top_k": 2},
            },
        ],
    })

    settings = Settings.from_env({"DOMAIN_PROFILES_JSON": raw})

    assert settings.default_domain_name == "finance"
    assert settings.domain_registry is not None
    assert settings.domain_registry.list_names() == ["finance", "support"]
    finance = settings.domain_registry.get("finance")
    assert finance.answering.model == "ministral"
    assert finance.ingestion.strategy == "manual"
    assert finance.access.required_roles == ("analyst",)


def test_domain_registry_returns_default_when_name_omitted():
    config = DomainConfiguration.from_mapping(
        {
            "domains": [
                {
                    "name": "alpha",
                    "is_default": True,
                    "collections": [{"name": "alpha_docs"}],
                    "retrieval": {"top_k_per_collection": 1, "final_top_k": 1},
                }
            ]
        },
        fallback_default_domain="default",
    )

    registry = DomainRegistry.from_configuration(config)

    assert registry.default_domain_name == "alpha"
    assert registry.get().name == "alpha"
    assert registry.get("alpha").collection_names == ("alpha_docs",)


def test_retrieval_profile_allows_final_top_k_lower_than_per_collection():
    config = DomainConfiguration.from_mapping(
        {
            "domains": [
                {
                    "name": "alpha",
                    "collections": [{"name": "alpha_docs"}],
                    "retrieval": {"top_k_per_collection": 5, "final_top_k": 2},
                }
            ]
        },
        fallback_default_domain="alpha",
    )

    domain = config.domains[0]

    assert domain.retrieval.top_k_per_collection == 5
    assert domain.retrieval.final_top_k == 2


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"domains": []}, "domains.domains must be a non-empty list"),
        ({"domains": [{"name": "Bad Name", "collections": [{"name": "docs"}], "retrieval": {"top_k_per_collection": 1, "final_top_k": 1}}]}, "must match"),
        ({"domains": [{"name": "alpha", "collections": [], "retrieval": {"top_k_per_collection": 1, "final_top_k": 1}}]}, "collections must be a non-empty list"),
        ({"default_domain": "missing", "domains": [{"name": "alpha", "collections": [{"name": "docs"}], "retrieval": {"top_k_per_collection": 1, "final_top_k": 1}}]}, "must reference one of"),
    ],
)
def test_invalid_domain_definitions_raise_validation_errors(payload, message):
    with pytest.raises(DomainProfileError, match=message):
        DomainConfiguration.from_mapping(payload, fallback_default_domain="default")


def test_settings_wraps_invalid_domain_profile_errors():
    with pytest.raises(SettingsError, match="DOMAIN_PROFILES_JSON must be valid JSON"):
        Settings.from_env({"DOMAIN_PROFILES_JSON": "{"})


def test_load_domain_configuration_rejects_non_object_json():
    with pytest.raises(DomainProfileError, match="must decode to an object"):
        load_domain_configuration(
            "[]",
            fallback_default_domain="default",
            product_collection="products",
            regulatory_collection="regulatory",
            top_k_per_collection=5,
            final_top_k=8,
        )
