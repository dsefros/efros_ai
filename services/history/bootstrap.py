from __future__ import annotations

from configs.settings import Settings
from services.history.postgres import PostgreSQLHistoryRepository
from services.history.repository import HistoryRepository


HISTORY_SERVICE_NAME = "history_repository"


def build_history_repository(settings: Settings, *, ensure_schema: bool = False) -> HistoryRepository | None:
    history_settings = settings.support_integrations.history_persistence
    if not history_settings.enabled:
        return None

    repository = PostgreSQLHistoryRepository(history_settings)
    if ensure_schema:
        repository.ensure_schema()
    return repository
