from __future__ import annotations

import types
from datetime import datetime, timezone
from uuid import uuid4

import pytest

from configs.settings import HistoryPersistenceSettings, Settings, SettingsError
from services.history.bootstrap import build_history_repository
from services.history.models import HistoryFeedbackCreate, HistoryRecordCreate, HistoryRecordUpdate
from services.history.postgres import HistoryPersistenceError, PostgreSQLHistoryRepository


class FakeCursor:
    def __init__(self, connection):
        self.connection = connection
        self.fetchone_result = None
        self.fetchall_result = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, statement, params=None):
        self.connection.statements.append((statement, params))
        handler = self.connection.handler
        if handler is not None:
            self.fetchone_result, self.fetchall_result = handler(statement, params)
        else:
            self.fetchone_result, self.fetchall_result = None, []

    def fetchone(self):
        return self.fetchone_result

    def fetchall(self):
        return list(self.fetchall_result)


class FakeConnection:
    def __init__(self, handler=None):
        self.handler = handler
        self.statements = []
        self.commits = 0
        self.rollbacks = 0
        self.closed = False

    def cursor(self):
        return FakeCursor(self)

    def commit(self):
        self.commits += 1

    def rollback(self):
        self.rollbacks += 1

    def close(self):
        self.closed = True


class ConnectionFactory:
    def __init__(self, connections):
        self.connections = list(connections)
        self.created = []

    def __call__(self):
        if not self.connections:
            raise AssertionError("No fake connections remaining")
        conn = self.connections.pop(0)
        self.created.append(conn)
        return conn


def _settings(**overrides):
    values = {
        "enabled": True,
        "host": "localhost",
        "port": 5432,
        "database": "support_history",
        "user": "efros",
        "password": "secret",
        "schema": "support_history",
        "ssl_mode": "prefer",
    }
    values.update(overrides)
    return HistoryPersistenceSettings(**values)


def _history_row(record_id, *, status="processed", delivery_status="sent", created_at=None):
    created_at = created_at or datetime(2026, 3, 20, tzinfo=timezone.utc)
    return (
        record_id,
        "redmine",
        "support",
        "123",
        "ops",
        "banking",
        "How do I reset MFA?",
        "Use the portal reset flow.",
        '{"queue": "tier2"}',
        '[{"title": "KB-1"}]',
        "ministral",
        status,
        delivery_status,
        None,
        "corr-1",
        "run-1",
        created_at,
    )


def _feedback_row(feedback_id, history_id):
    return (
        feedback_id,
        history_id,
        5,
        "Helpful",
        "agent",
        datetime(2026, 3, 20, tzinfo=timezone.utc),
    )


def test_postgres_repository_ensure_schema_executes_expected_statements():
    conn = FakeConnection()
    factory = ConnectionFactory([conn])
    repo = PostgreSQLHistoryRepository(_settings(), connection_factory=factory)

    repo.ensure_schema()

    executed_sql = "\n".join(statement for statement, _ in conn.statements)
    assert "CREATE SCHEMA IF NOT EXISTS support_history" in executed_sql
    assert "CREATE TABLE IF NOT EXISTS support_history.analysis_history" in executed_sql
    assert "CREATE TABLE IF NOT EXISTS support_history.analysis_history_feedback" in executed_sql
    assert conn.commits == 1
    assert conn.rollbacks == 0
    assert conn.closed is True


def test_postgres_repository_create_get_list_update_and_feedback_flow():
    history_id = uuid4()
    feedback_id = uuid4()

    rows = [
        _history_row(history_id, status="queued", delivery_status="pending"),
        _history_row(history_id, status="queued", delivery_status="pending"),
        None,
        [_history_row(history_id, status="queued", delivery_status="pending")],
        _history_row(history_id, status="delivered", delivery_status="sent"),
        _feedback_row(feedback_id, history_id),
    ]

    def handler(statement, params):
        outcome = rows.pop(0)
        if isinstance(outcome, list):
            return None, outcome
        return outcome, []

    factory = ConnectionFactory([FakeConnection(handler) for _ in range(6)])
    repo = PostgreSQLHistoryRepository(_settings(), connection_factory=factory)

    created = repo.create_record(
        HistoryRecordCreate(
            source_system="redmine",
            channel="support",
            issue_id="123",
            project_name="ops",
            domain="banking",
            question="How do I reset MFA?",
            answer="Use the portal reset flow.",
            routing_metadata={"queue": "tier2"},
            sources_json=[{"title": "KB-1"}],
            model_name="ministral",
            status="queued",
            delivery_status="pending",
            correlation_id="corr-1",
            run_id="run-1",
        )
    )
    loaded = repo.get_record(history_id)
    missing = repo.get_record(uuid4())
    listed = repo.list_records(channel="support", issue_id="123", limit=25)
    updated = repo.update_record(
        history_id,
        HistoryRecordUpdate(status="delivered", delivery_status="sent", model_name="ministral"),
    )
    feedback = repo.create_feedback(
        HistoryFeedbackCreate(history_id=history_id, rating=5, feedback_text="Helpful", feedback_source="agent")
    )

    assert created.id == history_id
    assert created.routing_metadata == {"queue": "tier2"}
    assert loaded is not None and loaded.id == history_id
    assert missing is None
    assert len(listed) == 1 and listed[0].id == history_id
    assert updated.status == "delivered"
    assert updated.delivery_status == "sent"
    assert feedback.id == feedback_id
    assert feedback.history_id == history_id

    list_sql, list_params = factory.created[3].statements[0]
    assert "WHERE channel = %s AND issue_id = %s" in list_sql
    assert list_params == ("support", "123", 25)


def test_postgres_repository_raises_for_empty_update():
    repo = PostgreSQLHistoryRepository(_settings(), connection_factory=ConnectionFactory([FakeConnection()]))

    with pytest.raises(ValueError, match="at least one field"):
        repo.update_record(uuid4(), HistoryRecordUpdate())


def test_postgres_repository_wraps_connection_errors():
    def broken_connection_factory():
        raise RuntimeError("db down")

    repo = PostgreSQLHistoryRepository(_settings(), connection_factory=broken_connection_factory)

    with pytest.raises(HistoryPersistenceError, match="Failed to ensure PostgreSQL history schema"):
        repo.ensure_schema()


def test_postgres_repository_validates_schema_identifier():
    with pytest.raises(SettingsError, match="letters, numbers, and underscores"):
        PostgreSQLHistoryRepository(_settings(schema="invalid-name"), connection_factory=ConnectionFactory([FakeConnection()]))


def test_postgres_repository_requires_enabled_settings():
    with pytest.raises(SettingsError, match="must be enabled"):
        PostgreSQLHistoryRepository(_settings(enabled=False), connection_factory=ConnectionFactory([FakeConnection()]))


def test_build_history_repository_returns_none_when_disabled():
    settings = Settings.from_env({})

    assert build_history_repository(settings) is None


def test_build_history_repository_skips_schema_initialization_by_default(monkeypatch):
    class StubRepository:
        def __init__(self, history_settings):
            self.history_settings = history_settings
            self.ensure_schema_calls = 0

        def ensure_schema(self):
            self.ensure_schema_calls += 1

    created = {}

    def build_repo(history_settings):
        repo = StubRepository(history_settings)
        created["repo"] = repo
        return repo

    monkeypatch.setattr("services.history.bootstrap.PostgreSQLHistoryRepository", build_repo)

    settings = Settings.from_env(
        {
            "HISTORY_PERSISTENCE_ENABLED": "true",
            "HISTORY_PERSISTENCE_HOST": "localhost",
            "HISTORY_PERSISTENCE_DATABASE": "support_history",
            "HISTORY_PERSISTENCE_USER": "efros",
            "HISTORY_PERSISTENCE_PASSWORD": "secret",
        }
    )

    repository = build_history_repository(settings)

    assert repository is created["repo"]
    assert repository.ensure_schema_calls == 0
    assert repository.history_settings.database == "support_history"


def test_build_history_repository_can_initialize_schema_explicitly(monkeypatch):
    class StubRepository:
        def __init__(self, history_settings):
            self.history_settings = history_settings
            self.ensure_schema_calls = 0

        def ensure_schema(self):
            self.ensure_schema_calls += 1

    created = {}

    def build_repo(history_settings):
        repo = StubRepository(history_settings)
        created["repo"] = repo
        return repo

    monkeypatch.setattr("services.history.bootstrap.PostgreSQLHistoryRepository", build_repo)

    settings = Settings.from_env(
        {
            "HISTORY_PERSISTENCE_ENABLED": "true",
            "HISTORY_PERSISTENCE_HOST": "localhost",
            "HISTORY_PERSISTENCE_DATABASE": "support_history",
            "HISTORY_PERSISTENCE_USER": "efros",
            "HISTORY_PERSISTENCE_PASSWORD": "secret",
        }
    )

    repository = build_history_repository(settings, ensure_schema=True)

    assert repository is created["repo"]
    assert repository.ensure_schema_calls == 1
    assert repository.history_settings.database == "support_history"
