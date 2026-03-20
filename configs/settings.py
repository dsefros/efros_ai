from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Mapping
from urllib.parse import urlparse

from configs.domain_profiles import (
    DomainConfiguration,
    DomainProfileError,
    DomainRegistry,
    load_domain_configuration,
)

from dotenv import load_dotenv
from services.integrations.redmine_client import (
    RedmineConfigurationError,
    normalize_redmine_project_ids,
    normalize_redmine_status_filters,
)


load_dotenv()


class SettingsError(ValueError):
    """Raised when application settings are missing or invalid."""


DEFAULT_DOMAIN_NAME = "default"


@dataclass(frozen=True)
class RedmineSettings:
    enabled: bool = False
    base_url: str | None = None
    api_key: str | None = None
    target_status_id: str | None = None
    status_ids: tuple[str, ...] = ()
    project_ids: tuple[int, ...] = ()


@dataclass(frozen=True)
class TelegramSettings:
    enabled: bool = False
    bot_token: str | None = None
    default_chat_id: str | None = None


@dataclass(frozen=True)
class HistoryPersistenceSettings:
    enabled: bool = False
    host: str | None = None
    port: int = 5432
    database: str | None = None
    user: str | None = None
    password: str | None = None
    schema: str = "public"
    ssl_mode: str = "prefer"


@dataclass(frozen=True)
class SupportIntegrationSettings:
    redmine: RedmineSettings = RedmineSettings()
    telegram: TelegramSettings = TelegramSettings()
    history_persistence: HistoryPersistenceSettings = HistoryPersistenceSettings()


@dataclass(frozen=True)
class Settings:
    app_env: str = "development"
    log_level: str = "INFO"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    default_model: str = "ministral"
    llm_backend: str = "local"
    qdrant_url: str = "http://localhost:6333"
    qdrant_product_collection: str = "rag_product"
    qdrant_regulatory_collection: str = "rag_regulatory"
    ministral_model_path: str = os.path.expanduser("~/models_ai/Ministral-8B-Instruct-2410-Q6_K_L.gguf")
    qwen2_model_path: str = os.path.expanduser("~/models_ai/qwen2-7b-instruct-q6_k.gguf")
    mistral7b_model_path: str = os.path.expanduser("~/models_ai/mistral-7b-instruct-v0.2.Q4_K_M.gguf")
    embedding_model: str = "BAAI/bge-m3"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    llm_n_ctx: int = 4096
    llm_n_threads: int = 8
    llm_n_gpu_layers: int = 0
    llm_temperature: float = 0.2
    llm_max_tokens: int = 900
    rag_top_k_per_collection: int = 5
    rag_final_top_k: int = 8
    support_integrations: SupportIntegrationSettings = SupportIntegrationSettings()
    domain_config: DomainConfiguration | None = None
    domain_registry: DomainRegistry | None = None
    default_domain_name: str = DEFAULT_DOMAIN_NAME

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "Settings":
        source = env or os.environ
        app_env = _get_str(source, "APP_ENV", cls.app_env)
        log_level = _get_log_level(source, "LOG_LEVEL", cls.log_level)
        api_host = _get_str(source, "API_HOST", cls.api_host)
        api_port = _get_int(source, "API_PORT", cls.api_port, minimum=1, maximum=65535)
        default_model = _get_str(source, "DEFAULT_MODEL", cls.default_model)
        llm_backend = _get_str(source, "LLM_BACKEND", cls.llm_backend)
        qdrant_url = _get_str(source, "QDRANT_URL", cls.qdrant_url)
        qdrant_product_collection = _get_str(source, "QDRANT_PRODUCT_COLLECTION", cls.qdrant_product_collection)
        qdrant_regulatory_collection = _get_str(source, "QDRANT_REGULATORY_COLLECTION", cls.qdrant_regulatory_collection)
        ministral_model_path = _get_str(source, "MINISTRAL_MODEL_PATH", cls.ministral_model_path)
        qwen2_model_path = _get_str(source, "QWEN2_MODEL_PATH", cls.qwen2_model_path)
        mistral7b_model_path = _get_str(source, "MISTRAL7B_MODEL_PATH", cls.mistral7b_model_path)
        embedding_model = _get_str(source, "EMBEDDING_MODEL", cls.embedding_model)
        reranker_model = _get_str(source, "RERANKER_MODEL", cls.reranker_model)
        llm_n_ctx = _get_int(source, "LLM_N_CTX", cls.llm_n_ctx, minimum=1)
        llm_n_threads = _get_int(source, "LLM_N_THREADS", cls.llm_n_threads, minimum=1)
        llm_n_gpu_layers = _get_int(source, "LLM_N_GPU_LAYERS", cls.llm_n_gpu_layers, minimum=0)
        llm_temperature = _get_float(source, "LLM_TEMPERATURE", cls.llm_temperature, minimum=0.0)
        llm_max_tokens = _get_int(source, "LLM_MAX_TOKENS", cls.llm_max_tokens, minimum=1)
        rag_top_k_per_collection = _get_int(source, "RAG_TOP_K_PER_COLLECTION", cls.rag_top_k_per_collection, minimum=1)
        rag_final_top_k = _get_int(source, "RAG_FINAL_TOP_K", cls.rag_final_top_k, minimum=1)
        default_domain_name = _get_str(source, "DEFAULT_DOMAIN_NAME", DEFAULT_DOMAIN_NAME)
        support_integrations = _load_support_integrations(source)

        try:
            domain_config = load_domain_configuration(
                _get_raw(source, "DOMAIN_PROFILES_JSON"),
                fallback_default_domain=default_domain_name,
                product_collection=qdrant_product_collection,
                regulatory_collection=qdrant_regulatory_collection,
                top_k_per_collection=rag_top_k_per_collection,
                final_top_k=rag_final_top_k,
            )
            domain_registry = DomainRegistry.from_configuration(domain_config)
        except DomainProfileError as exc:
            raise SettingsError(str(exc)) from exc

        return cls(
            app_env=app_env,
            log_level=log_level,
            api_host=api_host,
            api_port=api_port,
            default_model=default_model,
            llm_backend=llm_backend,
            qdrant_url=qdrant_url,
            qdrant_product_collection=qdrant_product_collection,
            qdrant_regulatory_collection=qdrant_regulatory_collection,
            ministral_model_path=ministral_model_path,
            qwen2_model_path=qwen2_model_path,
            mistral7b_model_path=mistral7b_model_path,
            embedding_model=embedding_model,
            reranker_model=reranker_model,
            llm_n_ctx=llm_n_ctx,
            llm_n_threads=llm_n_threads,
            llm_n_gpu_layers=llm_n_gpu_layers,
            llm_temperature=llm_temperature,
            llm_max_tokens=llm_max_tokens,
            rag_top_k_per_collection=rag_top_k_per_collection,
            rag_final_top_k=rag_final_top_k,
            support_integrations=support_integrations,
            domain_config=domain_config,
            domain_registry=domain_registry,
            default_domain_name=domain_registry.default_domain_name,
        )


def _load_support_integrations(env: Mapping[str, str]) -> SupportIntegrationSettings:
    redmine_enabled = _get_bool(env, "REDMINE_ENABLED", RedmineSettings.enabled)
    redmine_base_url = _get_optional_url_alias(env, "REDMINE_BASE_URL", aliases=("REDMINE_URL",))
    redmine_api_key = _get_raw(env, "REDMINE_API_KEY")
    redmine_target_status = _get_redmine_status_alias(
        env,
        "REDMINE_TARGET_STATUS_ID",
        aliases=("REDMINE_TARGET_STATUS", "TARGET_STATUS"),
    )
    redmine_status_list = _get_redmine_status_list_alias(env, "REDMINE_STATUS_IDS", aliases=("REDMINE_STATUS_LIST",))
    redmine_project_filters = _get_redmine_project_ids_alias(env, "REDMINE_PROJECT_IDS", aliases=("REDMINE_PROJECT_FILTERS",))
    if redmine_enabled:
        _require_present("REDMINE_BASE_URL", redmine_base_url)
        _require_present("REDMINE_API_KEY", redmine_api_key)

    telegram_enabled = _get_bool(env, "TELEGRAM_ENABLED", TelegramSettings.enabled)
    telegram_bot_token = _get_raw(env, "TELEGRAM_BOT_TOKEN")
    telegram_default_chat_id = _get_raw(env, "TELEGRAM_DEFAULT_CHAT_ID")
    if telegram_enabled:
        _require_present("TELEGRAM_BOT_TOKEN", telegram_bot_token)
        _require_present("TELEGRAM_DEFAULT_CHAT_ID", telegram_default_chat_id)

    history_enabled = _get_bool(env, "HISTORY_PERSISTENCE_ENABLED", HistoryPersistenceSettings.enabled)
    history_host = _get_raw(env, "HISTORY_PERSISTENCE_HOST")
    history_port = _get_int(env, "HISTORY_PERSISTENCE_PORT", HistoryPersistenceSettings.port, minimum=1, maximum=65535)
    history_database = _get_raw(env, "HISTORY_PERSISTENCE_DATABASE")
    history_user = _get_raw(env, "HISTORY_PERSISTENCE_USER")
    history_password = _get_raw(env, "HISTORY_PERSISTENCE_PASSWORD")
    history_schema = _get_str(env, "HISTORY_PERSISTENCE_SCHEMA", HistoryPersistenceSettings.schema)
    history_ssl_mode = _get_choice(
        env,
        "HISTORY_PERSISTENCE_SSL_MODE",
        HistoryPersistenceSettings.ssl_mode,
        {"disable", "allow", "prefer", "require", "verify-ca", "verify-full"},
    )
    if history_enabled:
        _require_present("HISTORY_PERSISTENCE_HOST", history_host)
        _require_present("HISTORY_PERSISTENCE_DATABASE", history_database)
        _require_present("HISTORY_PERSISTENCE_USER", history_user)
        _require_present("HISTORY_PERSISTENCE_PASSWORD", history_password)

    return SupportIntegrationSettings(
        redmine=RedmineSettings(
            enabled=redmine_enabled,
            base_url=redmine_base_url,
            api_key=redmine_api_key,
            target_status_id=redmine_target_status,
            status_ids=redmine_status_list,
            project_ids=redmine_project_filters,
        ),
        telegram=TelegramSettings(
            enabled=telegram_enabled,
            bot_token=telegram_bot_token,
            default_chat_id=telegram_default_chat_id,
        ),
        history_persistence=HistoryPersistenceSettings(
            enabled=history_enabled,
            host=history_host,
            port=history_port,
            database=history_database,
            user=history_user,
            password=history_password,
            schema=history_schema,
            ssl_mode=history_ssl_mode,
        ),
    )


def _get_raw(env: Mapping[str, str], name: str):
    value = env.get(name)
    if value is None:
        return None
    value = value.strip()
    return value if value else None


def _get_str(env: Mapping[str, str], name: str, default: str) -> str:
    value = _get_raw(env, name)
    return value if value is not None else default


def _get_bool(env: Mapping[str, str], name: str, default: bool) -> bool:
    value = _get_raw(env, name)
    if value is None:
        return default
    normalized = value.lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise SettingsError(f"{name} must be a boolean, got {value!r}")


def _get_int(
    env: Mapping[str, str],
    name: str,
    default: int,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    value = _get_raw(env, name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError as exc:
        raise SettingsError(f"{name} must be an integer, got {value!r}") from exc
    if minimum is not None and parsed < minimum:
        raise SettingsError(f"{name} must be >= {minimum}, got {parsed}")
    if maximum is not None and parsed > maximum:
        raise SettingsError(f"{name} must be <= {maximum}, got {parsed}")
    return parsed


def _get_float(
    env: Mapping[str, str],
    name: str,
    default: float,
    minimum: float | None = None,
) -> float:
    value = _get_raw(env, name)
    if value is None:
        return default
    try:
        parsed = float(value)
    except ValueError as exc:
        raise SettingsError(f"{name} must be a float, got {value!r}") from exc
    if minimum is not None and parsed < minimum:
        raise SettingsError(f"{name} must be >= {minimum}, got {parsed}")
    return parsed


def _get_log_level(env: Mapping[str, str], name: str, default: str) -> str:
    value = _get_str(env, name, default).upper()
    valid_levels = {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"}
    if value not in valid_levels:
        raise SettingsError(f"{name} must be one of {sorted(valid_levels)}, got {value!r}")
    return value


def _get_csv_list(env: Mapping[str, str], name: str) -> tuple[str, ...]:
    value = _get_raw(env, name)
    if value is None:
        return ()
    items = tuple(item.strip() for item in value.split(",") if item.strip())
    if not items:
        raise SettingsError(f"{name} must contain at least one non-empty comma-separated value when set")
    return items


def _get_choice(env: Mapping[str, str], name: str, default: str, valid_values: set[str]) -> str:
    value = _get_str(env, name, default)
    if value not in valid_values:
        raise SettingsError(f"{name} must be one of {sorted(valid_values)}, got {value!r}")
    return value



def _get_optional_url_alias(env: Mapping[str, str], name: str, aliases: tuple[str, ...] = ()) -> str | None:
    for candidate in (name, *aliases):
        value = _get_raw(env, candidate)
        if value is not None:
            parsed = urlparse(value)
            if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                raise SettingsError(f"{candidate} must be a valid http(s) URL, got {value!r}")
            return value
    return None


def _get_redmine_status_alias(env: Mapping[str, str], name: str, aliases: tuple[str, ...] = ()) -> str | None:
    for candidate in (name, *aliases):
        value = _get_raw(env, candidate)
        if value is not None:
            try:
                normalized = normalize_redmine_status_filters((value,))
            except RedmineConfigurationError as exc:
                raise SettingsError(str(exc)) from exc
            return normalized[0]
    return None


def _get_redmine_status_list_alias(env: Mapping[str, str], name: str, aliases: tuple[str, ...] = ()) -> tuple[str, ...]:
    for candidate in (name, *aliases):
        raw_values = _get_csv_list(env, candidate)
        if raw_values:
            try:
                return normalize_redmine_status_filters(raw_values)
            except RedmineConfigurationError as exc:
                raise SettingsError(str(exc)) from exc
    return ()


def _get_redmine_project_ids_alias(env: Mapping[str, str], name: str, aliases: tuple[str, ...] = ()) -> tuple[int, ...]:
    for candidate in (name, *aliases):
        raw_values = _get_csv_list(env, candidate)
        if raw_values:
            try:
                return normalize_redmine_project_ids(raw_values)
            except RedmineConfigurationError as exc:
                raise SettingsError(str(exc)) from exc
    return ()

def _get_optional_url(env: Mapping[str, str], name: str) -> str | None:
    value = _get_raw(env, name)
    if value is None:
        return None
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise SettingsError(f"{name} must be a valid http(s) URL, got {value!r}")
    return value


def _require_present(name: str, value: str | None) -> None:
    if value is None:
        raise SettingsError(f"{name} is required when the related integration is enabled")


@lru_cache(maxsize=1)
def load_settings() -> Settings:
    return Settings.from_env()


def reset_settings_cache() -> None:
    load_settings.cache_clear()
