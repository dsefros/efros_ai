import sys
import types
from dataclasses import replace

import pytest

sys.modules.setdefault("dotenv", types.SimpleNamespace(load_dotenv=lambda *args, **kwargs: False))

from configs.settings import RedmineSettings, Settings, SettingsError
from kernel.ai_kernel import AIKernel
from kernel.module_loader import load_module
from services.integrations import (
    REDMINE_SERVICE_NAME,
    ConfiguredRedmineService,
    RedmineAuthenticationError,
    RedmineClient,
    RedmineConfigurationError,
    RedmineIssue,
    RedmineNotFoundError,
    RedmineRequestError,
    build_issue_query_params,
    build_redmine_client,
    build_redmine_service,
    normalize_redmine_project_ids,
    normalize_redmine_status_filters,
)

class FakeTransport:
    def __init__(self, responses=None, errors=None):
        self.responses = responses or {}
        self.errors = errors or {}
        self.calls = []

    def request(self, method, path, *, params=None, json_body=None):
        key = (method, path)
        self.calls.append({"method": method, "path": path, "params": params, "json_body": json_body})
        if key in self.errors:
            raise self.errors[key]
        return self.responses.get(key, {})

@pytest.fixture
def redmine_settings():
    return RedmineSettings(
        enabled=True,
        base_url="https://redmine.example.com",
        api_key="token",
        target_status_id="150",
        status_ids=("1", "150"),
        project_ids=(10, 20),
    )

def test_normalize_redmine_status_filters_accepts_numeric_and_special_values():
    assert normalize_redmine_status_filters(["001", 150]) == ("1", "150")
    assert normalize_redmine_status_filters(["open"]) == ("open",)
    assert normalize_redmine_status_filters(["*"]) == ("*",)

@pytest.mark.parametrize("value", [["triaged"], ["open", "1"], [""]])
def test_normalize_redmine_status_filters_rejects_invalid_values(value):
    with pytest.raises(RedmineConfigurationError):
        normalize_redmine_status_filters(value)

@pytest.mark.parametrize("value", [["ops"], ["0"], [""]])
def test_normalize_redmine_project_ids_rejects_non_numeric_values(value):
    with pytest.raises(RedmineConfigurationError):
        normalize_redmine_project_ids(value)

def test_build_issue_query_params_uses_numeric_filters_and_bounds():
    params = build_issue_query_params(status_ids=["1", "150"], project_ids=[10, "20"], limit=25, offset=5)

    assert params == {
        "status_id": "1|150",
        "project_id": "10|20",
        "limit": "25",
        "offset": "5",
    }

@pytest.mark.parametrize("limit,offset", [(0, 0), (10, -1)])
def test_build_issue_query_params_validates_bounds(limit, offset):
    with pytest.raises(RedmineConfigurationError):
        build_issue_query_params(limit=limit, offset=offset)

def test_get_issue_normalizes_issue_payload():
    transport = FakeTransport(
        responses={
            ("GET", "/issues/42.json"): {
                "issue": {
                    "id": 42,
                    "subject": "Login broken",
                    "description": "Cannot sign in",
                    "status": {"id": 150, "name": "In Progress"},
                    "project": {"id": 10, "name": "Support"},
                    "author": {"id": 7, "name": "Alice"},
                    "assigned_to": {"id": 8, "name": "Bob"},
                    "tracker": {"id": 3, "name": "Bug"},
                    "priority": {"id": 4, "name": "High"},
                }
            }
        }
    )
    client = RedmineClient(base_url="https://redmine.example.com", api_key="token", transport=transport)

    issue = client.get_issue(42)

    assert issue == RedmineIssue(
        id=42,
        subject="Login broken",
        description="Cannot sign in",
        status=issue.status,
        project=issue.project,
        author=issue.author,
        assigned_to=issue.assigned_to,
        tracker=issue.tracker,
        priority=issue.priority,
        url="https://redmine.example.com/issues/42",
    )
    assert issue.status.id == 150
    assert issue.project.id == 10
    assert transport.calls[0]["params"] == {"include": "attachments"}

def test_list_issues_builds_expected_request():
    transport = FakeTransport(
        responses={
            ("GET", "/issues.json"): {
                "issues": [
                    {"id": 1, "subject": "One", "status": {"id": 1, "name": "New"}},
                    {"id": 2, "subject": "Two", "status": {"id": 2, "name": "Assigned"}},
                ]
            }
        }
    )
    client = RedmineClient(base_url="https://redmine.example.com", api_key="token", transport=transport)

    issues = client.list_issues(status_ids=[1, 2], project_ids=[10], limit=50, offset=10)

    assert [issue.id for issue in issues] == [1, 2]
    assert transport.calls[0]["params"] == {
        "status_id": "1|2",
        "project_id": "10",
        "limit": "50",
        "offset": "10",
    }

def test_add_private_note_sends_private_note_payload():
    transport = FakeTransport(responses={("PUT", "/issues/5.json"): {}})
    client = RedmineClient(base_url="https://redmine.example.com", api_key="token", transport=transport)

    client.add_private_note(5, "  analysis complete  ")

    assert transport.calls == [
        {
            "method": "PUT",
            "path": "/issues/5.json",
            "params": None,
            "json_body": {"issue": {"notes": "analysis complete", "private_notes": True}},
        }
    ]

def test_add_private_note_rejects_empty_note():
    client = RedmineClient(base_url="https://redmine.example.com", api_key="token", transport=FakeTransport())

    with pytest.raises(RedmineConfigurationError, match="cannot be empty"):
        client.add_private_note(5, "   ")

def test_typed_failure_paths_are_preserved():
    auth_transport = FakeTransport(errors={("GET", "/issues/1.json"): RedmineAuthenticationError("nope")})
    client = RedmineClient(base_url="https://redmine.example.com", api_key="token", transport=auth_transport)
    with pytest.raises(RedmineAuthenticationError):
        client.get_issue(1)

    missing_transport = FakeTransport(errors={("GET", "/issues/2.json"): RedmineNotFoundError("missing")})
    client = RedmineClient(base_url="https://redmine.example.com", api_key="token", transport=missing_transport)
    with pytest.raises(RedmineNotFoundError):
        client.get_issue(2)

    bad_payload_transport = FakeTransport(responses={("GET", "/issues.json"): {"issues": "oops"}})
    client = RedmineClient(base_url="https://redmine.example.com", api_key="token", transport=bad_payload_transport)
    with pytest.raises(RedmineRequestError):
        client.list_issues()

def test_build_redmine_client_requires_base_url_and_api_key(redmine_settings):
    with pytest.raises(RedmineConfigurationError, match="base URL"):
        build_redmine_client(replace(redmine_settings, base_url=None))

    with pytest.raises(RedmineConfigurationError, match="API key"):
        build_redmine_client(replace(redmine_settings, api_key=None))

def test_configured_redmine_service_prefers_target_status_and_projects(redmine_settings):
    transport = FakeTransport(responses={("GET", "/issues.json"): {"issues": []}})
    service = ConfiguredRedmineService(
        client=RedmineClient(base_url="https://redmine.example.com", api_key="token", transport=transport),
        settings=redmine_settings,
    )

    service.list_configured_issues(limit=10, offset=2)

    assert transport.calls[0]["params"] == {
        "status_id": "150|1",
        "project_id": "10|20",
        "limit": "10",
        "offset": "2",
    }

def test_configured_redmine_service_deduplicates_statuses_while_preserving_order(redmine_settings):
    transport = FakeTransport(responses={("GET", "/issues.json"): {"issues": []}})
    service = ConfiguredRedmineService(
        client=RedmineClient(base_url="https://redmine.example.com", api_key="token", transport=transport),
        settings=replace(redmine_settings, target_status_id="150", status_ids=("150", "1", "150", "2")),
    )

    service.list_configured_issues()

    assert transport.calls[0]["params"]["status_id"] == "150|1|2"

def test_configured_redmine_service_uses_status_list_when_target_missing(redmine_settings):
    transport = FakeTransport(responses={("GET", "/issues.json"): {"issues": []}})
    service = ConfiguredRedmineService(
        client=RedmineClient(base_url="https://redmine.example.com", api_key="token", transport=transport),
        settings=replace(redmine_settings, target_status_id=None),
    )

    service.list_configured_issues()

    assert transport.calls[0]["params"]["status_id"] == "1|150"

def test_configured_redmine_service_handles_absent_filters(redmine_settings):
    transport = FakeTransport(responses={("GET", "/issues.json"): {"issues": []}})
    service = ConfiguredRedmineService(
        client=RedmineClient(base_url="https://redmine.example.com", api_key="token", transport=transport),
        settings=replace(redmine_settings, target_status_id=None, status_ids=(), project_ids=()),
    )

    service.list_configured_issues()

    assert transport.calls[0]["params"] == {"limit": "100", "offset": "0"}

def test_build_redmine_service_returns_registered_shape(redmine_settings):
    settings = Settings(support_integrations=replace(Settings().support_integrations, redmine=redmine_settings))

    service = build_redmine_service(settings, transport=FakeTransport())

    assert service is not None
    assert service.settings.target_status_id == "150"

def test_build_redmine_service_returns_none_when_disabled(redmine_settings):
    settings = Settings(support_integrations=replace(Settings().support_integrations, redmine=replace(redmine_settings, enabled=False)))

    assert build_redmine_service(settings) is None

def test_redmine_tool_bridge_uses_bound_kernel_service(monkeypatch, redmine_settings):
    sys.modules.setdefault("yaml", types.SimpleNamespace(safe_load=lambda *args, **kwargs: {"name": "support_module", "entrypoint": "module.py"}))
    kernel = AIKernel()
    transport = FakeTransport(responses={("GET", "/issues/9.json"): {"issue": {"id": 9, "subject": "Issue 9"}}})
    kernel.register_service(
        REDMINE_SERVICE_NAME,
        ConfiguredRedmineService(
            client=RedmineClient(base_url="https://redmine.example.com", api_key="token", transport=transport),
            settings=redmine_settings,
        ),
    )

    load_module(kernel, "modules/support_module")

    result = kernel.tools.get("redmine_get_issue").execute(issue_id=9)

    assert result.id == 9
    assert transport.calls[0]["path"] == "/issues/9.json"

def test_redmine_tool_bridge_raises_when_service_missing():
    sys.modules.setdefault("yaml", types.SimpleNamespace(safe_load=lambda *args, **kwargs: {"name": "support_module", "entrypoint": "module.py"}))
    kernel = AIKernel()
    load_module(kernel, "modules/support_module")

    with pytest.raises(RedmineConfigurationError, match="not registered"):
        kernel.tools.get("redmine_get_issue").execute(issue_id=1)

def test_settings_from_env_supports_redmine_alias_precedence():
    settings = Settings.from_env(
        {
            "REDMINE_BASE_URL": "https://preferred.example.com",
            "REDMINE_URL": "https://legacy.example.com",
            "REDMINE_API_KEY": "token",
            "REDMINE_TARGET_STATUS_ID": "150",
            "REDMINE_TARGET_STATUS": "151",
            "TARGET_STATUS": "152",
            "REDMINE_STATUS_IDS": "1, 150, 5",
            "REDMINE_STATUS_LIST": "2, 3",
            "REDMINE_PROJECT_IDS": "10,20",
            "REDMINE_PROJECT_FILTERS": "7,8",
        }
    )

    redmine = settings.support_integrations.redmine
    assert redmine.base_url == "https://preferred.example.com"
    assert redmine.target_status_id == "150"
    assert redmine.status_ids == ("1", "150", "5")
    assert redmine.project_ids == (10, 20)

def test_settings_from_env_supports_legacy_redmine_aliases():
    settings = Settings.from_env(
        {
            "REDMINE_URL": "https://legacy.example.com",
            "REDMINE_API_KEY": "token",
            "TARGET_STATUS": "open",
            "REDMINE_PROJECT_IDS": "7",
        }
    )

    redmine = settings.support_integrations.redmine
    assert redmine.base_url == "https://legacy.example.com"
    assert redmine.target_status_id == "open"
    assert redmine.project_ids == (7,)

def test_settings_from_env_rejects_human_readable_redmine_status_names():
    with pytest.raises(SettingsError, match="numeric ids or one of"):
        Settings.from_env({"REDMINE_TARGET_STATUS": "triaged"})

    with pytest.raises(SettingsError, match="numeric ids or one of"):
        Settings.from_env({"REDMINE_STATUS_IDS": "1,triaged"})

def test_settings_from_env_rejects_non_numeric_project_filters():
    with pytest.raises(SettingsError, match="numeric project ids"):
        Settings.from_env({"REDMINE_PROJECT_IDS": "ops"})
