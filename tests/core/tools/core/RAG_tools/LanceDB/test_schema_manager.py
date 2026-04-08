from __future__ import annotations

from pathlib import Path

import pyarrow as pa

from xagent.core.tools.core.RAG_tools.LanceDB.model_tag_utils import to_model_tag
from xagent.core.tools.core.RAG_tools.LanceDB.schema_manager import (
    check_table_needs_migration,
    ensure_chunks_table,
    ensure_documents_table,
    ensure_embeddings_table,
    ensure_parses_table,
)
from xagent.core.tools.core.RAG_tools.storage import get_vector_store_raw_connection


def test_ensure_tables(tmp_path: Path, monkeypatch) -> None:
    db_dir = tmp_path / "db"
    monkeypatch.setenv("LANCEDB_DIR", str(db_dir))
    conn = get_vector_store_raw_connection()
    ensure_documents_table(conn)
    ensure_parses_table(conn)
    ensure_chunks_table(conn)
    ensure_embeddings_table(conn, to_model_tag("BAAI/bge-large-zh-v1.5"))

    # open_table should not raise
    for name in [
        "documents",
        "parses",
        "chunks",
        "embeddings_BAAI_bge_large_zh_v1_5",
    ]:
        conn.open_table(name)


def test_check_table_needs_migration_table_not_exists(
    tmp_path: Path, monkeypatch
) -> None:
    """Test check_table_needs_migration when table doesn't exist."""
    db_dir = tmp_path / "db"
    monkeypatch.setenv("LANCEDB_DIR", str(db_dir))
    conn = get_vector_store_raw_connection()

    # Table doesn't exist, should return False
    assert check_table_needs_migration(conn, "nonexistent_table") is False


def test_check_table_needs_migration_table_without_user_id(
    tmp_path: Path, monkeypatch
) -> None:
    """Test check_table_needs_migration when table exists but missing user_id field."""
    db_dir = tmp_path / "db"
    monkeypatch.setenv("LANCEDB_DIR", str(db_dir))
    conn = get_vector_store_raw_connection()

    # Create a table without user_id field (old schema)
    old_schema = pa.schema(
        [
            pa.field("collection", pa.string()),
            pa.field("doc_id", pa.string()),
            pa.field("text", pa.string()),
        ]
    )
    conn.create_table("test_table_old", schema=old_schema)

    # Should detect that migration is needed
    assert check_table_needs_migration(conn, "test_table_old") is True


def test_check_table_needs_migration_table_with_user_id(
    tmp_path: Path, monkeypatch
) -> None:
    """Test check_table_needs_migration when table exists and has user_id field."""
    db_dir = tmp_path / "db"
    monkeypatch.setenv("LANCEDB_DIR", str(db_dir))
    conn = get_vector_store_raw_connection()

    # Create a table with user_id field (new schema)
    new_schema = pa.schema(
        [
            pa.field("collection", pa.string()),
            pa.field("doc_id", pa.string()),
            pa.field("text", pa.string()),
            pa.field("user_id", pa.int32(), nullable=True),
        ]
    )
    conn.create_table("test_table_new", schema=new_schema)

    # Should detect that no migration is needed
    assert check_table_needs_migration(conn, "test_table_new") is False


def test_check_table_needs_migration_with_ensure_tables(
    tmp_path: Path, monkeypatch
) -> None:
    """Test check_table_needs_migration with tables created by ensure_* functions."""
    db_dir = tmp_path / "db"
    monkeypatch.setenv("LANCEDB_DIR", str(db_dir))
    conn = get_vector_store_raw_connection()

    # Create tables using ensure_* functions (which create tables with user_id)
    ensure_documents_table(conn)
    ensure_chunks_table(conn)
    ensure_parses_table(conn)

    # All should have user_id, so no migration needed
    assert check_table_needs_migration(conn, "documents") is False
    assert check_table_needs_migration(conn, "chunks") is False
    assert check_table_needs_migration(conn, "parses") is False


def test_ensure_documents_table_backfills_empty_string_file_id_to_null(
    tmp_path: Path, monkeypatch
) -> None:
    """ensure_documents_table should backfill empty string file_id values to None."""
    db_dir = tmp_path / "db"
    monkeypatch.setenv("LANCEDB_DIR", str(db_dir))
    conn = get_connection_from_env()

    schema = pa.schema(
        [
            pa.field("collection", pa.string()),
            pa.field("doc_id", pa.string()),
            pa.field("file_id", pa.string()),
            pa.field("source_path", pa.string()),
            pa.field("file_type", pa.string()),
            pa.field("content_hash", pa.string()),
            pa.field("uploaded_at", pa.timestamp("us")),
            pa.field("title", pa.string()),
            pa.field("language", pa.string()),
            pa.field("user_id", pa.int64()),
        ]
    )
    conn.create_table("documents", schema=schema)
    table = conn.open_table("documents")
    # Simulate legacy data with empty string file_id (from previous PR)
    table.add(
        [
            {
                "collection": "c1",
                "doc_id": "d1",
                "file_id": "",  # Empty string from previous backfill
                "source_path": "/tmp/a.md",
                "file_type": "md",
                "content_hash": "h1",
                "uploaded_at": None,
                "title": None,
                "language": None,
                "user_id": None,
            }
        ]
    )

    ensure_documents_table(conn)

    refreshed = conn.open_table("documents")
    updated = refreshed.search().where("doc_id = 'd1'").to_list()[0]
    assert updated["file_id"] is None


def test_ensure_documents_table_backfills_user_id_from_source_path(
    tmp_path: Path, monkeypatch
) -> None:
    """ensure_documents_table should recover user_id from legacy source paths."""
    db_dir = tmp_path / "db"
    monkeypatch.setenv("LANCEDB_DIR", str(db_dir))
    conn = get_connection_from_env()

    schema = pa.schema(
        [
            pa.field("collection", pa.string()),
            pa.field("doc_id", pa.string()),
            pa.field("file_id", pa.string()),
            pa.field("source_path", pa.string()),
            pa.field("file_type", pa.string()),
            pa.field("content_hash", pa.string()),
            pa.field("uploaded_at", pa.timestamp("us")),
            pa.field("title", pa.string()),
            pa.field("language", pa.string()),
            pa.field("user_id", pa.int64()),
        ]
    )
    conn.create_table("documents", schema=schema)
    table = conn.open_table("documents")
    table.add(
        [
            {
                "collection": "xagent",
                "doc_id": "legacy-doc-1",
                "file_id": "",
                "source_path": "/home/xagent/uploads/user_58/xagent/README.md",
                "file_type": "md",
                "content_hash": "h1",
                "uploaded_at": None,
                "title": None,
                "language": None,
                "user_id": None,
            },
            {
                "collection": "xagent",
                "doc_id": "legacy-doc-2",
                "file_id": "",
                "source_path": "/legacy/path/no-user-prefix.md",
                "file_type": "md",
                "content_hash": "h2",
                "uploaded_at": None,
                "title": None,
                "language": None,
                "user_id": None,
            },
        ]
    )

    ensure_documents_table(conn)

    refreshed = conn.open_table("documents")
    rows = refreshed.search().to_list()
    row_map = {row["doc_id"]: row for row in rows}
    assert row_map["legacy-doc-1"]["user_id"] == 58
    assert row_map["legacy-doc-2"]["user_id"] is None


def test_backfill_file_id_to_null_is_idempotent(
    tmp_path: Path, monkeypatch, caplog
) -> None:
    """Backfill file_id to NULL should be idempotent and log progress."""
    import logging

    db_dir = tmp_path / "db"
    monkeypatch.setenv("LANCEDB_DIR", str(db_dir))
    conn = get_connection_from_env()

    schema = pa.schema(
        [
            pa.field("collection", pa.string()),
            pa.field("doc_id", pa.string()),
            pa.field("file_id", pa.string()),
        ]
    )
    conn.create_table("documents", schema=schema)
    table = conn.open_table("documents")

    # Add rows with empty string file_id
    table.add(
        [
            {"collection": "c1", "doc_id": "d1", "file_id": ""},
            {"collection": "c1", "doc_id": "d2", "file_id": ""},
        ]
    )

    # First backfill
    with caplog.at_level(logging.INFO):
        ensure_documents_table(conn)

    # Verify backfill happened and was logged
    assert any(
        "Backfilled empty string file_id values to NULL" in record.message
        for record in caplog.records
    )

    refreshed = conn.open_table("documents")
    rows = refreshed.search().to_list()
    for row in rows:
        assert row["file_id"] is None

    # Second backfill should be idempotent (no changes)
    caplog.clear()
    with caplog.at_level(logging.INFO):
        ensure_documents_table(conn)

    # Should not log again since no backfill needed
    assert not any(
        "Backfilled empty string file_id values to NULL" in record.message
        for record in caplog.records
    )


def test_backfill_user_id_logs_progress(tmp_path: Path, monkeypatch, caplog) -> None:
    """User ID backfill should log total rows updated."""
    import logging

    db_dir = tmp_path / "db"
    monkeypatch.setenv("LANCEDB_DIR", str(db_dir))
    conn = get_connection_from_env()

    schema = pa.schema(
        [
            pa.field("collection", pa.string()),
            pa.field("doc_id", pa.string()),
            pa.field("file_id", pa.string()),
            pa.field("source_path", pa.string()),
            pa.field("user_id", pa.int64()),
        ]
    )
    conn.create_table("documents", schema=schema)
    table = conn.open_table("documents")

    # Add rows with NULL user_id but recoverable from source_path
    table.add(
        [
            {
                "collection": "xagent",
                "doc_id": "d1",
                "file_id": "",
                "source_path": "/uploads/user_42/xagent/file.pdf",
                "user_id": None,
            },
            {
                "collection": "xagent",
                "doc_id": "d2",
                "file_id": "",
                "source_path": "/uploads/user_99/xagent/doc.pdf",
                "user_id": None,
            },
        ]
    )

    # Run backfill with logging
    with caplog.at_level(logging.INFO):
        ensure_documents_table(conn)

    # Verify progress was logged
    assert any(
        "Backfilled 2 user_id values from source_path" in record.message
        for record in caplog.records
    )

    # Verify backfill worked
    refreshed = conn.open_table("documents")
    rows = refreshed.search().to_list()
    row_map = {row["doc_id"]: row for row in rows}
    assert row_map["d1"]["user_id"] == 42
    assert row_map["d2"]["user_id"] == 99
