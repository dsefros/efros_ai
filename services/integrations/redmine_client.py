from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Protocol
from urllib import error, parse, request

if TYPE_CHECKING:
    from configs.settings import RedmineSettings, Settings

REDMINE_SERVICE_NAME = "redmine"
_VALID_SPECIAL_STATUS_FILTERS = {"*", "open", "closed"}


class RedmineError(RuntimeError):
    """Base error for Redmine integration failures."""


class RedmineConfigurationError(RedmineError):
    """Raised when Redmine settings are missing or invalid for runtime usage."""


class RedmineRequestError(RedmineError):
    """Raised when the Redmine API request fails."""


class RedmineAuthenticationError(RedmineRequestError):
    """Raised when the Redmine API rejects authentication."""


class RedmineNotFoundError(RedmineRequestError):
    """Raised when a Redmine issue is not found."""


@dataclass(frozen=True)
class RedmineIssueRelation:
    id: int | None
    name: str | None


@dataclass(frozen=True)
class RedmineIssue:
    id: int
    subject: str
    description: str
    status: RedmineIssueRelation
    project: RedmineIssueRelation
    author: RedmineIssueRelation
    assigned_to: RedmineIssueRelation
    tracker: RedmineIssueRelation
    priority: RedmineIssueRelation
    url: str


class RedmineTransport(Protocol):
    def request(
        self,
        method: str,
        path: str,
        *,
        params: Mapping[str, str] | None = None,
        json_body: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]: ...


class UrllibRedmineTransport:
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key

    def request(
        self,
        method: str,
        path: str,
        *,
        params: Mapping[str, str] | None = None,
        json_body: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        url = f"{self.base_url}{path}"
        if params:
            url = f"{url}?{parse.urlencode(params)}"
        payload = None
        headers = {
            "X-Redmine-API-Key": self.api_key,
            "Accept": "application/json",
        }
        if json_body is not None:
            payload = json.dumps(json_body).encode("utf-8")
            headers["Content-Type"] = "application/json"
        req = request.Request(url=url, data=payload, headers=headers, method=method.upper())
        try:
            with request.urlopen(req) as response:
                body = response.read().decode("utf-8")
        except error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="ignore")
            raise _map_http_error(exc.code, body) from exc
        except error.URLError as exc:
            raise RedmineRequestError(f"Unable to reach Redmine: {exc.reason}") from exc

        if not body:
            return {}
        return json.loads(body)


class RedmineClient:
    def __init__(self, *, base_url: str, api_key: str, transport: RedmineTransport | None = None):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self._transport = transport or UrllibRedmineTransport(self.base_url, api_key)

    def get_issue(self, issue_id: int) -> RedmineIssue:
        payload = self._transport.request("GET", f"/issues/{int(issue_id)}.json", params={"include": "attachments"})
        issue = payload.get("issue")
        if not isinstance(issue, Mapping):
            raise RedmineRequestError("Redmine response did not include an issue payload")
        return _normalize_issue(issue, self.base_url)

    def list_issues(
        self,
        *,
        status_ids: Iterable[str | int] | None = None,
        project_ids: Iterable[int] | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[RedmineIssue]:
        params = build_issue_query_params(
            status_ids=status_ids,
            project_ids=project_ids,
            limit=limit,
            offset=offset,
        )
        payload = self._transport.request("GET", "/issues.json", params=params)
        issues = payload.get("issues", [])
        if not isinstance(issues, list):
            raise RedmineRequestError("Redmine response did not include an issues list")
        return [_normalize_issue(issue, self.base_url) for issue in issues if isinstance(issue, Mapping)]

    def add_private_note(self, issue_id: int, note: str) -> None:
        cleaned_note = note.strip()
        if not cleaned_note:
            raise RedmineConfigurationError("Redmine private note cannot be empty")
        self._transport.request(
            "PUT",
            f"/issues/{int(issue_id)}.json",
            json_body={"issue": {"notes": cleaned_note, "private_notes": True}},
        )


@dataclass(frozen=True)
class ConfiguredRedmineService:
    client: RedmineClient
    settings: "RedmineSettings"

    def list_configured_issues(self, *, limit: int = 100, offset: int = 0) -> list[RedmineIssue]:
        ordered_status_filters: list[str] = []
        seen_status_filters: set[str] = set()
        for value in ((self.settings.target_status_id,) if self.settings.target_status_id is not None else ()) + self.settings.status_ids:
            if value not in seen_status_filters:
                ordered_status_filters.append(value)
                seen_status_filters.add(value)
        return self.client.list_issues(
            status_ids=tuple(ordered_status_filters) or None,
            project_ids=self.settings.project_ids or None,
            limit=limit,
            offset=offset,
        )


def build_redmine_client(
    settings: "RedmineSettings | Settings",
    *,
    transport: RedmineTransport | None = None,
) -> RedmineClient:
    redmine_settings = settings.support_integrations.redmine if hasattr(settings, "support_integrations") else settings
    if not redmine_settings.base_url:
        raise RedmineConfigurationError("Redmine base URL is not configured")
    if not redmine_settings.api_key:
        raise RedmineConfigurationError("Redmine API key is not configured")
    return RedmineClient(
        base_url=redmine_settings.base_url,
        api_key=redmine_settings.api_key,
        transport=transport,
    )


def build_redmine_service(
    settings: "Settings",
    *,
    transport: RedmineTransport | None = None,
) -> ConfiguredRedmineService | None:
    redmine_settings = settings.support_integrations.redmine
    if not redmine_settings.enabled:
        return None
    client = build_redmine_client(redmine_settings, transport=transport)
    return ConfiguredRedmineService(client=client, settings=redmine_settings)


def build_issue_query_params(
    *,
    status_ids: Iterable[str | int] | None = None,
    project_ids: Iterable[int] | None = None,
    limit: int = 100,
    offset: int = 0,
) -> dict[str, str]:
    if limit < 1:
        raise RedmineConfigurationError("Redmine issue list limit must be >= 1")
    if offset < 0:
        raise RedmineConfigurationError("Redmine issue list offset must be >= 0")

    params = {"limit": str(limit), "offset": str(offset)}
    normalized_status_ids = normalize_redmine_status_filters(status_ids or ())
    normalized_project_ids = normalize_redmine_project_ids(project_ids or ())
    if normalized_status_ids:
        params["status_id"] = "|".join(normalized_status_ids)
    if normalized_project_ids:
        params["project_id"] = "|".join(str(project_id) for project_id in normalized_project_ids)
    return params


def normalize_redmine_status_filters(values: Iterable[str | int]) -> tuple[str, ...]:
    normalized: list[str] = []
    for value in values:
        item = str(value).strip()
        if not item:
            raise RedmineConfigurationError("Redmine status filters cannot contain empty values")
        if item in _VALID_SPECIAL_STATUS_FILTERS:
            normalized.append(item)
            continue
        if not item.isdigit():
            raise RedmineConfigurationError(
                "Redmine status filters must be numeric ids or one of '*', 'open', 'closed'"
            )
        normalized.append(str(int(item)))

    if not normalized:
        return ()

    special_values = [value for value in normalized if value in _VALID_SPECIAL_STATUS_FILTERS]
    if special_values and len(normalized) > 1:
        raise RedmineConfigurationError("Redmine special status filters cannot be combined with other values")
    return tuple(normalized)


def normalize_redmine_project_ids(values: Iterable[str | int]) -> tuple[int, ...]:
    normalized: list[int] = []
    for value in values:
        item = str(value).strip()
        if not item:
            raise RedmineConfigurationError("Redmine project filters cannot contain empty values")
        if not item.isdigit():
            raise RedmineConfigurationError("Redmine project filters must contain numeric project ids")
        parsed = int(item)
        if parsed <= 0:
            raise RedmineConfigurationError("Redmine project filters must be positive numeric project ids")
        normalized.append(parsed)
    return tuple(normalized)


def _normalize_issue(issue: Mapping[str, Any], base_url: str) -> RedmineIssue:
    issue_id = int(issue["id"])
    return RedmineIssue(
        id=issue_id,
        subject=str(issue.get("subject") or ""),
        description=str(issue.get("description") or ""),
        status=_normalize_relation(issue.get("status")),
        project=_normalize_relation(issue.get("project")),
        author=_normalize_relation(issue.get("author")),
        assigned_to=_normalize_relation(issue.get("assigned_to")),
        tracker=_normalize_relation(issue.get("tracker")),
        priority=_normalize_relation(issue.get("priority")),
        url=f"{base_url.rstrip('/')}/issues/{issue_id}",
    )


def _normalize_relation(value: Any) -> RedmineIssueRelation:
    if not isinstance(value, Mapping):
        return RedmineIssueRelation(id=None, name=None)
    relation_id = value.get("id")
    return RedmineIssueRelation(
        id=int(relation_id) if relation_id is not None else None,
        name=str(value.get("name")) if value.get("name") is not None else None,
    )


def _map_http_error(status_code: int, body: str) -> RedmineRequestError:
    message = body.strip() or f"HTTP {status_code}"
    if status_code in {401, 403}:
        return RedmineAuthenticationError(message)
    if status_code == 404:
        return RedmineNotFoundError(message)
    return RedmineRequestError(message)
