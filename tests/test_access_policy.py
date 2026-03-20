import json
import sys
import types

import pytest

sys.modules.setdefault("dotenv", types.SimpleNamespace(load_dotenv=lambda *args, **kwargs: False))

from configs.settings import Settings
from configs.domain_profiles import DomainProfileError
from services.knowledge.access_policy import DomainAccessPolicyEvaluator, RequestAccessContext


def build_settings():
    raw = json.dumps(
        {
            "default_domain": "public-docs",
            "domains": [
                {
                    "name": "public-docs",
                    "is_default": True,
                    "collections": [{"name": "public_docs"}],
                    "retrieval": {"top_k_per_collection": 1, "final_top_k": 1},
                    "access": {"visibility": "public"},
                },
                {
                    "name": "finance",
                    "collections": [{"name": "finance_docs"}],
                    "retrieval": {"top_k_per_collection": 1, "final_top_k": 1},
                    "access": {
                        "visibility": "private",
                        "allowed_roles": ["analyst"],
                        "allowed_callers": ["svc-finance"],
                        "allowed_groups": ["risk"],
                    },
                },
                {
                    "name": "ops",
                    "collections": [{"name": "ops_docs"}],
                    "retrieval": {"top_k_per_collection": 1, "final_top_k": 1},
                    "access": {
                        "visibility": "internal",
                        "default_action": "allow",
                    },
                },
            ],
        }
    )
    return Settings.from_env({"DOMAIN_PROFILES_JSON": raw})


def test_access_policy_allows_matching_role():
    evaluator = DomainAccessPolicyEvaluator(build_settings().domain_registry)

    decision = evaluator.evaluate("finance", RequestAccessContext.from_values(roles=["analyst"]))

    assert decision.allowed is True
    assert decision.matched_by == "role"


def test_access_policy_denies_missing_identity_for_private_domain():
    evaluator = DomainAccessPolicyEvaluator(build_settings().domain_registry)

    decision = evaluator.evaluate("finance", RequestAccessContext())

    assert decision.allowed is False
    assert "access denied" in decision.reason


def test_access_policy_unknown_domain_raises_clear_error():
    evaluator = DomainAccessPolicyEvaluator(build_settings().domain_registry)

    with pytest.raises(DomainProfileError, match="Unknown domain 'missing'"):
        evaluator.evaluate("missing", RequestAccessContext())


def test_access_policy_honors_default_allow_behavior():
    evaluator = DomainAccessPolicyEvaluator(build_settings().domain_registry)

    decision = evaluator.evaluate("ops", RequestAccessContext())

    assert decision.allowed is True
    assert decision.matched_by == "default"
