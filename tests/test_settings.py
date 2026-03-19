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
