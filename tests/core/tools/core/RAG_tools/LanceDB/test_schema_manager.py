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
