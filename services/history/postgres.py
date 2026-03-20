from __future__ import annotations

import json
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Callable, Iterator, Sequence
from uuid import UUID

from configs.settings import HistoryPersistenceSettings, SettingsError
from services.history.models import (
    HistoryFeedback,
    HistoryFeedbackCreate,
    HistoryRecord,
    HistoryRecordCreate,
    HistoryRecordUpdate,
)
from services.history.repository import HistoryRepository

ConnectionFactory = Callable[[], Any]


class HistoryPersistenceError(RuntimeError):
    """Raised when the history persistence backend cannot complete a request."""


_SCHEMA_STATEMENTS = (
    """
    CREATE SCHEMA IF NOT EXISTS {schema}
    """,
    """
    CREATE TABLE IF NOT EXISTS {schema}.analysis_history (
        id UUID PRIMARY KEY,
        source_system VARCHAR(100) NOT NULL,
        channel VARCHAR(100) NOT NULL,
        issue_id VARCHAR(255),
        project_name VARCHAR(255),
        domain VARCHAR(255),
        question TEXT NOT NULL,
        answer TEXT NOT NULL,
        routing_metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
        sources_json JSONB NOT NULL DEFAULT '[]'::jsonb,
        model_name VARCHAR(255),
        status VARCHAR(64) NOT NULL,
        delivery_status VARCHAR(64) NOT NULL,
        error TEXT,
        correlation_id VARCHAR(255),
        run_id VARCHAR(255),
        created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_analysis_history_issue_created_at
    ON {schema}.analysis_history (issue_id, created_at DESC)
    WHERE issue_id IS NOT NULL
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_analysis_history_channel_created_at
    ON {schema}.analysis_history (channel, created_at DESC)
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_analysis_history_created_at
    ON {schema}.analysis_history (created_at DESC)
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_analysis_history_status_delivery
    ON {schema}.analysis_history (status, delivery_status, created_at DESC)
    """,
    """
    CREATE TABLE IF NOT EXISTS {schema}.analysis_history_feedback (
        id UUID PRIMARY KEY,
        history_id UUID NOT NULL REFERENCES {schema}.analysis_history(id) ON DELETE CASCADE,
        rating INTEGER CHECK (rating BETWEEN 1 AND 5),
        feedback_text TEXT,
        feedback_source VARCHAR(100),
        created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_analysis_history_feedback_history_id
    ON {schema}.analysis_history_feedback (history_id, created_at DESC)
    """,
)


class PostgreSQLHistoryRepository(HistoryRepository):
    def __init__(self, settings: HistoryPersistenceSettings, connection_factory: ConnectionFactory | None = None):
        if not settings.enabled:
            raise SettingsError("History persistence must be enabled before creating the PostgreSQL repository")
        self._settings = settings
        self._schema = _validate_identifier(settings.schema, name="HISTORY_PERSISTENCE_SCHEMA")
        self._connection_factory = connection_factory or self._build_connection_factory(settings)

    @staticmethod
    def _build_connection_factory(settings: HistoryPersistenceSettings) -> ConnectionFactory:
        def factory():
            try:
                import psycopg
            except ModuleNotFoundError as exc:
                raise HistoryPersistenceError(
                    "PostgreSQL history persistence requires the optional 'psycopg' package"
                ) from exc

            try:
                return psycopg.connect(
                    host=settings.host,
                    port=settings.port,
                    dbname=settings.database,
                    user=settings.user,
                    password=settings.password,
                    sslmode=settings.ssl_mode,
                    autocommit=False,
                )
            except Exception as exc:  # pragma: no cover - exercised via injected factory in tests
                raise HistoryPersistenceError(f"Unable to connect to PostgreSQL history persistence: {exc}") from exc

        return factory

    @contextmanager
    def _connection(self) -> Iterator[Any]:
        conn = self._connection_factory()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def ensure_schema(self) -> None:
        try:
            with self._connection() as conn:
                with conn.cursor() as cur:
                    for statement in _SCHEMA_STATEMENTS:
                        cur.execute(statement.format(schema=self._schema))
        except Exception as exc:
            raise _wrap_history_error("Failed to ensure PostgreSQL history schema", exc) from exc

    def create_record(self, payload: HistoryRecordCreate) -> HistoryRecord:
        statement = f"""
            INSERT INTO {self._schema}.analysis_history (
                id,
                source_system,
                channel,
                issue_id,
                project_name,
                domain,
                question,
                answer,
                routing_metadata,
                sources_json,
                model_name,
                status,
                delivery_status,
                error,
                correlation_id,
                run_id
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id, source_system, channel, issue_id, project_name, domain, question, answer,
                      routing_metadata, sources_json, model_name, status, delivery_status, error,
                      correlation_id, run_id, created_at
        """
        record_id = _new_uuid()
        params = (
            record_id,
            payload.source_system,
            payload.channel,
            payload.issue_id,
            payload.project_name,
            payload.domain,
            payload.question,
            payload.answer,
            json.dumps(payload.routing_metadata or {}),
            json.dumps(payload.sources_json or []),
            payload.model_name,
            payload.status,
            payload.delivery_status,
            payload.error,
            payload.correlation_id,
            payload.run_id,
        )
        return self._fetch_single_record(statement, params, action="create history record")

    def get_record(self, record_id: UUID) -> HistoryRecord | None:
        statement = f"""
            SELECT id, source_system, channel, issue_id, project_name, domain, question, answer,
                   routing_metadata, sources_json, model_name, status, delivery_status, error,
                   correlation_id, run_id, created_at
            FROM {self._schema}.analysis_history
            WHERE id = %s
        """
        try:
            with self._connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(statement, (record_id,))
                    row = cur.fetchone()
                    return _map_history_record(row) if row else None
        except Exception as exc:
            raise _wrap_history_error("Failed to load history record", exc) from exc

    def list_records(self, limit: int = 100, *, channel: str | None = None, issue_id: str | None = None) -> Sequence[HistoryRecord]:
        conditions: list[str] = []
        params: list[Any] = []
        if channel is not None:
            conditions.append("channel = %s")
            params.append(channel)
        if issue_id is not None:
            conditions.append("issue_id = %s")
            params.append(issue_id)
        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        statement = f"""
            SELECT id, source_system, channel, issue_id, project_name, domain, question, answer,
                   routing_metadata, sources_json, model_name, status, delivery_status, error,
                   correlation_id, run_id, created_at
            FROM {self._schema}.analysis_history
            {where_clause}
            ORDER BY created_at DESC
            LIMIT %s
        """
        params.append(limit)
        try:
            with self._connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(statement, tuple(params))
                    return [_map_history_record(row) for row in cur.fetchall()]
        except Exception as exc:
            raise _wrap_history_error("Failed to list history records", exc) from exc

    def update_record(self, record_id: UUID, payload: HistoryRecordUpdate) -> HistoryRecord:
        updates: list[str] = []
        params: list[Any] = []
        for column, value in (
            ("status", payload.status),
            ("delivery_status", payload.delivery_status),
            ("error", payload.error),
            ("answer", payload.answer),
            ("sources_json", json.dumps(payload.sources_json) if payload.sources_json is not None else None),
            ("model_name", payload.model_name),
        ):
            if value is None:
                continue
            updates.append(f"{column} = %s")
            params.append(value)
        if not updates:
            raise ValueError("HistoryRecordUpdate must contain at least one field to update")
        params.append(record_id)
        statement = f"""
            UPDATE {self._schema}.analysis_history
            SET {', '.join(updates)}
            WHERE id = %s
            RETURNING id, source_system, channel, issue_id, project_name, domain, question, answer,
                      routing_metadata, sources_json, model_name, status, delivery_status, error,
                      correlation_id, run_id, created_at
        """
        return self._fetch_single_record(statement, tuple(params), action="update history record")

    def create_feedback(self, payload: HistoryFeedbackCreate) -> HistoryFeedback:
        statement = f"""
            INSERT INTO {self._schema}.analysis_history_feedback (
                id,
                history_id,
                rating,
                feedback_text,
                feedback_source
            )
            VALUES (%s, %s, %s, %s, %s)
            RETURNING id, history_id, rating, feedback_text, feedback_source, created_at
        """
        feedback_id = _new_uuid()
        try:
            with self._connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        statement,
                        (
                            feedback_id,
                            payload.history_id,
                            payload.rating,
                            payload.feedback_text,
                            payload.feedback_source,
                        ),
                    )
                    row = cur.fetchone()
                    if row is None:
                        raise HistoryPersistenceError("PostgreSQL did not return the created history feedback row")
                    return _map_feedback_record(row)
        except Exception as exc:
            raise _wrap_history_error("Failed to create history feedback", exc) from exc

    def _fetch_single_record(self, statement: str, params: tuple[Any, ...], *, action: str) -> HistoryRecord:
        try:
            with self._connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(statement, params)
                    row = cur.fetchone()
                    if row is None:
                        raise HistoryPersistenceError(f"PostgreSQL did not return a row while attempting to {action}")
                    return _map_history_record(row)
        except Exception as exc:
            raise _wrap_history_error(f"Failed to {action}", exc) from exc


def _validate_identifier(value: str, *, name: str) -> str:
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_")
    if not value or any(ch not in allowed for ch in value):
        raise SettingsError(f"{name} must contain only letters, numbers, and underscores")
    return value


def _map_history_record(row: Sequence[Any]) -> HistoryRecord:
    return HistoryRecord(
        id=_as_uuid(row[0]),
        source_system=row[1],
        channel=row[2],
        issue_id=row[3],
        project_name=row[4],
        domain=row[5],
        question=row[6],
        answer=row[7],
        routing_metadata=_load_json_value(row[8], default={}),
        sources_json=_load_json_value(row[9], default=[]),
        model_name=row[10],
        status=row[11],
        delivery_status=row[12],
        error=row[13],
        correlation_id=row[14],
        run_id=row[15],
        created_at=_as_datetime(row[16]),
    )


def _map_feedback_record(row: Sequence[Any]) -> HistoryFeedback:
    return HistoryFeedback(
        id=_as_uuid(row[0]),
        history_id=_as_uuid(row[1]),
        rating=row[2],
        feedback_text=row[3],
        feedback_source=row[4],
        created_at=_as_datetime(row[5]),
    )


def _load_json_value(value: Any, *, default: Any) -> Any:
    if value is None:
        return default
    if isinstance(value, (dict, list)):
        return value
    return json.loads(value)


def _as_uuid(value: Any) -> UUID:
    return value if isinstance(value, UUID) else UUID(str(value))


def _as_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(str(value))


def _new_uuid() -> UUID:
    from uuid import uuid4

    return uuid4()


def _wrap_history_error(message: str, exc: Exception) -> HistoryPersistenceError:
    if isinstance(exc, HistoryPersistenceError):
        return exc
    return HistoryPersistenceError(f"{message}: {exc}")
