from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence
from uuid import UUID

from services.history.models import (
    HistoryFeedback,
    HistoryFeedbackCreate,
    HistoryRecord,
    HistoryRecordCreate,
    HistoryRecordUpdate,
)


class HistoryRepository(ABC):
    @abstractmethod
    def ensure_schema(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def create_record(self, payload: HistoryRecordCreate) -> HistoryRecord:
        raise NotImplementedError

    @abstractmethod
    def get_record(self, record_id: UUID) -> HistoryRecord | None:
        raise NotImplementedError

    @abstractmethod
    def list_records(self, limit: int = 100, *, channel: str | None = None, issue_id: str | None = None) -> Sequence[HistoryRecord]:
        raise NotImplementedError

    @abstractmethod
    def update_record(self, record_id: UUID, payload: HistoryRecordUpdate) -> HistoryRecord:
        raise NotImplementedError

    @abstractmethod
    def create_feedback(self, payload: HistoryFeedbackCreate) -> HistoryFeedback:
        raise NotImplementedError
