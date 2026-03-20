from __future__ import annotations

from dataclasses import dataclass

from configs.domain_profiles import DomainProfile, DomainProfileError, DomainRegistry


@dataclass(frozen=True)
class RequestAccessContext:
    caller: str | None = None
    roles: tuple[str, ...] = ()
    groups: tuple[str, ...] = ()

    @classmethod
    def from_values(
        cls,
        *,
        caller: str | None = None,
        roles: tuple[str, ...] | list[str] | None = None,
        groups: tuple[str, ...] | list[str] | None = None,
    ) -> "RequestAccessContext":
        return cls(
            caller=_normalize_optional(caller),
            roles=_normalize_many(roles),
            groups=_normalize_many(groups),
        )


@dataclass(frozen=True)
class AccessDecision:
    allowed: bool
    domain: str
    reason: str
    matched_by: str | None = None


class DomainAccessPolicyEvaluator:
    def __init__(self, registry: DomainRegistry | None):
        self.registry = registry

    def resolve_domain(self, domain: str | None = None) -> DomainProfile | None:
        if self.registry is None:
            if domain is not None:
                raise DomainProfileError("Domain selection is unavailable because no domain registry is configured")
            return None
        return self.registry.get(domain)

    def evaluate(self, domain: str | None, context: RequestAccessContext | None = None) -> AccessDecision:
        profile = self.resolve_domain(domain)
        if profile is None:
            return AccessDecision(allowed=True, domain="default", reason="domain registry not configured", matched_by="legacy")
        return self.evaluate_profile(profile, context=context)

    def evaluate_profile(self, profile: DomainProfile, context: RequestAccessContext | None = None) -> AccessDecision:
        context = context or RequestAccessContext()
        access = profile.access

        if context.caller and context.caller in access.allowed_callers:
            return AccessDecision(True, profile.name, "caller explicitly allowed", matched_by="caller")
        if access.allowed_roles and set(context.roles).intersection(access.allowed_roles):
            return AccessDecision(True, profile.name, "role explicitly allowed", matched_by="role")
        if access.allowed_groups and set(context.groups).intersection(access.allowed_groups):
            return AccessDecision(True, profile.name, "group explicitly allowed", matched_by="group")

        if access.default_action == "allow":
            return AccessDecision(True, profile.name, f"default allow for {access.visibility} domain", matched_by="default")

        restrictions = []
        if access.allowed_callers:
            restrictions.append("callers")
        if access.allowed_roles:
            restrictions.append("roles")
        if access.allowed_groups:
            restrictions.append("groups")

        if restrictions:
            reason = f"access denied: no matching {', '.join(restrictions)} policy"
        else:
            reason = f"access denied by default {access.default_action} policy for {access.visibility} domain"
        return AccessDecision(False, profile.name, reason)


def _normalize_optional(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip()
    return value or None


def _normalize_many(values: tuple[str, ...] | list[str] | None) -> tuple[str, ...]:
    if not values:
        return ()
    normalized: list[str] = []
    seen: set[str] = set()
    for item in values:
        if not isinstance(item, str):
            continue
        cleaned = item.strip()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        normalized.append(cleaned)
    return tuple(normalized)
