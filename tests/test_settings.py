import importlib
import sys
import types

import pytest

sys.modules.setdefault("dotenv", types.SimpleNamespace(load_dotenv=lambda *args, **kwargs: False))

from configs.settings import Settings, SettingsError, load_settings, reset_settings_cache

@pytest.fixture(autouse=True)
def clear_settings_cache():
    reset_settings_cache()
    yield
    reset_settings_cache()

def test_settings_module_import_does_not_validate_environment(monkeypatch):
    monkeypatch.setenv("API_PORT", "not-a-port")
    reset_settings_cache()

    sys.modules.pop("configs.settings", None)
    settings_module = importlib.import_module("configs.settings")

    assert hasattr(settings_module, "Settings")

def test_settings_from_env_validates_numeric_fields():
    with pytest.raises(SettingsError, match="API_PORT must be an integer"):
        Settings.from_env({"API_PORT": "not-a-port"})

    with pytest.raises(SettingsError, match="LLM_N_THREADS must be >= 1"):
        Settings.from_env({"LLM_N_THREADS": "0"})

def test_settings_from_env_validates_log_level():
    with pytest.raises(SettingsError, match="LOG_LEVEL must be one of"):
        Settings.from_env({"LOG_LEVEL": "verbose"})

def test_settings_from_env_keeps_support_integrations_disabled_by_default():
    settings = Settings.from_env({})

    assert settings.support_integrations.redmine.enabled is False
    assert settings.support_integrations.redmine.base_url is None
    assert settings.support_integrations.redmine.status_ids == ()
    assert settings.support_integrations.redmine.project_ids == ()
    assert settings.support_integrations.telegram.enabled is False
    assert settings.support_integrations.telegram.bot_token is None
    assert settings.support_integrations.history_persistence.enabled is False
    assert settings.support_integrations.history_persistence.port == 5432
    assert settings.support_integrations.history_persistence.ssl_mode == "prefer"

def test_settings_from_env_parses_support_integrations():
    settings = Settings.from_env(
        {
            "REDMINE_ENABLED": "true",
            "REDMINE_BASE_URL": "https://redmine.example.com",
            "REDMINE_API_KEY": "redmine-token",
            "REDMINE_TARGET_STATUS_ID": "150",
            "REDMINE_STATUS_IDS": "1, 150, 5",
            "REDMINE_PROJECT_IDS": "10, 20",
            "TELEGRAM_ENABLED": "true",
            "TELEGRAM_BOT_TOKEN": "12345:telegram-token",
            "TELEGRAM_DEFAULT_CHAT_ID": "-1001234567890",
            "HISTORY_PERSISTENCE_ENABLED": "true",
            "HISTORY_PERSISTENCE_HOST": "postgres.example.com",
            "HISTORY_PERSISTENCE_PORT": "5433",
            "HISTORY_PERSISTENCE_DATABASE": "support_history",
            "HISTORY_PERSISTENCE_USER": "efros",
            "HISTORY_PERSISTENCE_PASSWORD": "secret",
            "HISTORY_PERSISTENCE_SCHEMA": "support",
            "HISTORY_PERSISTENCE_SSL_MODE": "require",
        }
    )

    assert settings.support_integrations.redmine.enabled is True
    assert settings.support_integrations.redmine.base_url == "https://redmine.example.com"
    assert settings.support_integrations.redmine.api_key == "redmine-token"
    assert settings.support_integrations.redmine.target_status_id == "150"
    assert settings.support_integrations.redmine.status_ids == ("1", "150", "5")
    assert settings.support_integrations.redmine.project_ids == (10, 20)
    assert settings.support_integrations.telegram.enabled is True
    assert settings.support_integrations.telegram.bot_token == "12345:telegram-token"
    assert settings.support_integrations.telegram.default_chat_id == "-1001234567890"
    assert settings.support_integrations.history_persistence.enabled is True
    assert settings.support_integrations.history_persistence.host == "postgres.example.com"
    assert settings.support_integrations.history_persistence.port == 5433
    assert settings.support_integrations.history_persistence.database == "support_history"
    assert settings.support_integrations.history_persistence.user == "efros"
    assert settings.support_integrations.history_persistence.password == "secret"
    assert settings.support_integrations.history_persistence.schema == "support"
    assert settings.support_integrations.history_persistence.ssl_mode == "require"

def test_settings_from_env_prefers_canonical_redmine_filter_env_names():
    settings = Settings.from_env(
        {
            "REDMINE_STATUS_IDS": "150, 151",
            "REDMINE_STATUS_LIST": "1, 2",
            "REDMINE_PROJECT_IDS": "10, 20",
            "REDMINE_PROJECT_FILTERS": "7, 8",
        }
    )

    assert settings.support_integrations.redmine.status_ids == ("150", "151")
    assert settings.support_integrations.redmine.project_ids == (10, 20)

def test_settings_from_env_falls_back_to_legacy_redmine_filter_env_names():
    settings = Settings.from_env(
        {
            "REDMINE_STATUS_LIST": "1, 2",
            "REDMINE_PROJECT_FILTERS": "7, 8",
        }
    )

    assert settings.support_integrations.redmine.status_ids == ("1", "2")
    assert settings.support_integrations.redmine.project_ids == (7, 8)

def test_settings_from_env_requires_integration_fields_only_when_enabled():
    with pytest.raises(SettingsError, match="REDMINE_BASE_URL is required"):
        Settings.from_env({"REDMINE_ENABLED": "true", "REDMINE_API_KEY": "token"})

    with pytest.raises(SettingsError, match="TELEGRAM_DEFAULT_CHAT_ID is required"):
        Settings.from_env({"TELEGRAM_ENABLED": "true", "TELEGRAM_BOT_TOKEN": "token"})

    with pytest.raises(SettingsError, match="HISTORY_PERSISTENCE_DATABASE is required"):
        Settings.from_env(
            {
                "HISTORY_PERSISTENCE_ENABLED": "true",
                "HISTORY_PERSISTENCE_HOST": "postgres.example.com",
                "HISTORY_PERSISTENCE_USER": "efros",
                "HISTORY_PERSISTENCE_PASSWORD": "secret",
            }
        )

def test_settings_from_env_validates_support_integration_values():
    with pytest.raises(SettingsError, match=r"REDMINE_BASE_URL must be a valid http\(s\) URL"):
        Settings.from_env({"REDMINE_BASE_URL": "ftp://redmine.example.com"})

    with pytest.raises(SettingsError, match="numeric ids or one of"):
        Settings.from_env({"REDMINE_TARGET_STATUS": "triaged"})

    with pytest.raises(SettingsError, match="numeric ids or one of"):
        Settings.from_env({"REDMINE_STATUS_LIST": "1,triaged"})

    with pytest.raises(SettingsError, match="TELEGRAM_ENABLED must be a boolean"):
        Settings.from_env({"TELEGRAM_ENABLED": "maybe"})

    with pytest.raises(SettingsError, match="HISTORY_PERSISTENCE_SSL_MODE must be one of"):
        Settings.from_env({"HISTORY_PERSISTENCE_SSL_MODE": "sometimes"})

def test_load_settings_uses_cache_until_reset(monkeypatch):
    monkeypatch.setenv("API_PORT", "9000")
    first = load_settings()

    monkeypatch.setenv("API_PORT", "9001")
    second = load_settings()

    assert first is second
    assert second.api_port == 9000

    reset_settings_cache()
    third = load_settings()
    assert third.api_port == 9001

def test_run_module_exits_cleanly_for_invalid_settings(monkeypatch):
    sys.modules["uvicorn"] = types.SimpleNamespace(run=lambda *args, **kwargs: None)
    sys.modules["api.server"] = types.SimpleNamespace(create_app=lambda kernel, model_manager, runtime=None: None)
    sys.modules["kernel.register_knowledge"] = types.SimpleNamespace(register_knowledge=lambda kernel: None)
    sys.modules["services.models.model_manager"] = types.SimpleNamespace(create_default_manager=lambda settings=None: object())
    sys.modules["yaml"] = types.SimpleNamespace(safe_load=lambda *args, **kwargs: {})

    monkeypatch.setenv("API_PORT", "invalid")
    reset_settings_cache()
    sys.modules.pop("run", None)
    run = importlib.import_module("run")

    with pytest.raises(SystemExit, match="Configuration error: API_PORT must be an integer"):
        run.main()
