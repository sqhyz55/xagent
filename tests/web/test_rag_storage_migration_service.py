from __future__ import annotations

import asyncio

import pytest

from xagent.web.services.rag_storage_migration_service import RAGStorageMigrationService


@pytest.mark.asyncio
async def test_start_background_migrations_creates_task(monkeypatch: pytest.MonkeyPatch) -> None:
    service = RAGStorageMigrationService()
    called = {"run": 0}

    async def _fake_run() -> None:
        called["run"] += 1

    monkeypatch.setattr(service, "_run_migrations", _fake_run)

    task = await service.start_background_migrations()
    await task

    assert isinstance(task, asyncio.Task)
    assert called["run"] == 1


@pytest.mark.asyncio
async def test_run_migrations_skips_documents_backfill_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = RAGStorageMigrationService()
    monkeypatch.setenv("LANCEDB_AUTO_MIGRATE", "false")
    called = {"user_id": 0, "docs": 0}

    async def _fake_user_id(*, auto_migrate: bool) -> None:
        assert auto_migrate is False
        called["user_id"] += 1

    async def _fake_docs() -> None:
        called["docs"] += 1

    monkeypatch.setattr(service, "_check_and_migrate_user_id", _fake_user_id)
    monkeypatch.setattr(service, "_check_and_backfill_documents_table", _fake_docs)

    await service._run_migrations()

    assert called["user_id"] == 1
    assert called["docs"] == 0


@pytest.mark.asyncio
async def test_run_migrations_runs_documents_backfill_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = RAGStorageMigrationService()
    monkeypatch.setenv("LANCEDB_AUTO_MIGRATE", "true")
    called = {"user_id": 0, "docs": 0}

    async def _fake_user_id(*, auto_migrate: bool) -> None:
        assert auto_migrate is True
        called["user_id"] += 1

    async def _fake_docs() -> None:
        called["docs"] += 1

    monkeypatch.setattr(service, "_check_and_migrate_user_id", _fake_user_id)
    monkeypatch.setattr(service, "_check_and_backfill_documents_table", _fake_docs)

    await service._run_migrations()

    assert called["user_id"] == 1
    assert called["docs"] == 1
