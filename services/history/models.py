from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from uuid import UUID


@dataclass(frozen=True)
class HistoryRecordCreate:
    source_system: str
    channel: str
    question: str
    answer: str
    status: str
    delivery_status: str
    issue_id: str | None = None
    project_name: str | None = None
    domain: str | None = None
    routing_metadata: dict[str, Any] = field(default_factory=dict)
    sources_json: list[dict[str, Any]] = field(default_factory=list)
    model_name: str | None = None
    error: str | None = None
    correlation_id: str | None = None
    run_id: str | None = None


@dataclass(frozen=True)
class HistoryRecord:
    id: UUID
    source_system: str
    channel: str
    issue_id: str | None
    project_name: str | None
    domain: str | None
    question: str
    answer: str
    routing_metadata: dict[str, Any]
    sources_json: list[dict[str, Any]]
    model_name: str | None
    status: str
    delivery_status: str
    error: str | None
    correlation_id: str | None
    run_id: str | None
    created_at: datetime


@dataclass(frozen=True)
class HistoryRecordUpdate:
    status: str | None = None
    delivery_status: str | None = None
    error: str | None = None
    answer: str | None = None
    sources_json: list[dict[str, Any]] | None = None
    model_name: str | None = None


@dataclass(frozen=True)
class HistoryFeedbackCreate:
    history_id: UUID
    rating: int | None = None
    feedback_text: str | None = None
    feedback_source: str | None = None


@dataclass(frozen=True)
class HistoryFeedback:
    id: UUID
    history_id: UUID
    rating: int | None
    feedback_text: str | None
    feedback_source: str | None
    created_at: datetime
