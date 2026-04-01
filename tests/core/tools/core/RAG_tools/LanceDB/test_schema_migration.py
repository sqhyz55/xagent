from __future__ import annotations

from pathlib import Path
from threading import Thread
from unittest.mock import Mock

import pandas as pd
import pyarrow as pa
import pytest

from xagent.core.tools.core.RAG_tools.LanceDB.schema_manager import (
    _create_table,
    _ensure_schema_fields,
    _get_sql_default_for_pa_type,
    _table_exists,
    ensure_collection_config_table,
    ensure_collection_metadata_table,
    ensure_documents_table,
    ensure_embeddings_table,
    ensure_ingestion_runs_table,
    ensure_parses_table,
    ensure_prompt_templates_table,
)
from xagent.providers.vector_store.lancedb import get_connection_from_env


def test_get_sql_default_for_pa_type():
    """Test default value generation for PyArrow types."""
    assert _get_sql_default_for_pa_type(pa.string()) == "''"
    assert _get_sql_default_for_pa_type(pa.large_string()) == "''"
    assert _get_sql_default_for_pa_type(pa.int32()) == "0"
    assert _get_sql_default_for_pa_type(pa.float64()) == "0.0"
    assert _get_sql_default_for_pa_type(pa.bool_()) == "false"
    assert _get_sql_default_for_pa_type(pa.timestamp("us")) == "CAST(NULL AS TIMESTAMP)"
    # Fallback
    assert _get_sql_default_for_pa_type(pa.binary()) == "NULL"


def test_auto_migration_adds_missing_columns(tmp_path: Path, monkeypatch):
    """Test that missing columns are automatically added with correct defaults."""
    db_dir = tmp_path / "db"
    monkeypatch.setenv("LANCEDB_DIR", str(db_dir))
    conn = get_connection_from_env()

    # 1. Create a table with an OLD schema (missing 'language' and 'title')
    old_schema = pa.schema(
        [
            pa.field("collection", pa.string()),
            pa.field("doc_id", pa.string()),
            # missing fields...
        ]
    )
    conn.create_table("documents", schema=old_schema)

    # Insert some data
    conn.open_table("documents").add([{"collection": "test", "doc_id": "1"}])

    # 2. Run ensure_documents_table which should trigger migration
    ensure_documents_table(conn)

    # 3. Verify new columns exist
    table = conn.open_table("documents")
    schema = table.schema
    field_names = [f.name for f in schema]
    assert "file_id" in field_names
    assert "title" in field_names
    assert "language" in field_names
    assert "uploaded_at" in field_names

    # 4. Verify default values in existing data
    df = table.to_pandas()
    row = df.iloc[0]
    assert row["file_id"] == ""
    # String defaults should be empty string
    assert row["title"] == ""
    assert row["language"] == ""
    # Timestamp default should be NaT (None)
    assert pd.isna(row["uploaded_at"])


def test_ensure_schema_fields_idempotency(tmp_path: Path, monkeypatch):
    """Test that calling migration on an up-to-date table is safe."""
    db_dir = tmp_path / "db"
    monkeypatch.setenv("LANCEDB_DIR", str(db_dir))
    conn = get_connection_from_env()

    # Create table with FULL schema first
    ensure_collection_metadata_table(conn)
    table_before = conn.open_table("collection_metadata")
    schema_before = table_before.schema

    # Call it again
    ensure_collection_metadata_table(conn)

    table_after = conn.open_table("collection_metadata")
    schema_after = table_after.schema

    assert schema_before == schema_after

    # Also verify idempotency does not duplicate/corrupt data.
    sample_row = {
        "name": "col_a",
        "schema_version": "1",
        "embedding_model_id": "test-model",
        "embedding_dimension": 3,
        "documents": 1,
        "processed_documents": 1,
        "parses": 1,
        "chunks": 1,
        "embeddings": 1,
        "document_names": "[]",
        "collection_locked": False,
        "allow_mixed_parse_methods": False,
        "skip_config_validation": False,
        "ingestion_config": None,
        "created_at": None,
        "updated_at": None,
        "last_accessed_at": None,
        "extra_metadata": None,
    }
    table_after.add([sample_row])
    count_before = int(table_after.count_rows())

    ensure_collection_metadata_table(conn)

    table_final = conn.open_table("collection_metadata")
    count_after = int(table_final.count_rows())
    assert count_before == count_after

    rows = table_final.search().where("name = 'col_a'").to_pandas()
    assert len(rows) == 1
    assert rows.iloc[0]["embedding_model_id"] == "test-model"


def test_manual_migration_helper(tmp_path: Path, monkeypatch):
    """Test the low-level _ensure_schema_fields helper directly."""
    db_dir = tmp_path / "db"
    monkeypatch.setenv("LANCEDB_DIR", str(db_dir))
    conn = get_connection_from_env()

    # Setup simple table
    conn.create_table("test_manual", schema=pa.schema([("a", pa.int32())]))
    conn.open_table("test_manual").add([{"a": 1}])

    # Define target schema with new field
    target_schema = pa.schema(
        [("a", pa.int32()), ("b", pa.string()), ("c", pa.int32())]
    )

    # Run migration
    _ensure_schema_fields(conn, "test_manual", target_schema)
    # Check results
    df = conn.open_table("test_manual").to_pandas()
    assert "b" in df.columns
    assert "c" in df.columns
    assert df.iloc[0]["b"] == ""
    assert df.iloc[0]["c"] == 0


def test_ensure_schema_fields_type_mismatch_keeps_existing_type(
    tmp_path: Path, monkeypatch
) -> None:
    """Type mismatch should not rewrite existing column types."""
    db_dir = tmp_path / "db"
    monkeypatch.setenv("LANCEDB_DIR", str(db_dir))
    conn = get_connection_from_env()

    conn.create_table("test_type_mismatch", schema=pa.schema([("a", pa.int32())]))
    conn.open_table("test_type_mismatch").add([{"a": 7}])

    target_schema = pa.schema([("a", pa.string()), ("b", pa.string())])
    _ensure_schema_fields(conn, "test_type_mismatch", target_schema)

    table = conn.open_table("test_type_mismatch")
    schema = table.schema
    assert schema.field("a").type == pa.int32()
    assert schema.field("b").type == pa.string()

    df = table.to_pandas()
    assert int(df.iloc[0]["a"]) == 7
    assert df.iloc[0]["b"] == ""


def test_ensure_schema_fields_partial_failure_raises(
    tmp_path: Path, monkeypatch
) -> None:
    """When add_columns fails, migration should raise instead of silently masking."""
    db_dir = tmp_path / "db"
    monkeypatch.setenv("LANCEDB_DIR", str(db_dir))
    conn = get_connection_from_env()

    conn.create_table("test_partial_failure", schema=pa.schema([("a", pa.int32())]))
    table = conn.open_table("test_partial_failure")
    original_open_table = conn.open_table

    original_add_columns = table.add_columns

    def _failing_once(new_cols):  # type: ignore[no-untyped-def]
        if "b" in new_cols:
            raise RuntimeError("simulated add_columns failure")
        return original_add_columns(new_cols)

    monkeypatch.setattr(table, "add_columns", _failing_once)
    monkeypatch.setattr(
        conn,
        "open_table",
        lambda name: (
            table if name == "test_partial_failure" else original_open_table(name)
        ),
    )
    target_schema = pa.schema([("a", pa.int32()), ("b", pa.string())])

    with pytest.raises(RuntimeError, match="simulated add_columns failure"):
        _ensure_schema_fields(conn, "test_partial_failure", target_schema)


def test_table_exists_returns_false_on_open_error() -> None:
    """_table_exists should return False when open_table raises."""
    conn = Mock()
    conn.open_table.side_effect = RuntimeError("boom")
    assert _table_exists(conn, "missing_table") is False


def test_create_table_without_schema_calls_conn_create_table() -> None:
    """_create_table should call conn.create_table when schema is None."""
    conn = Mock()
    conn.open_table.side_effect = RuntimeError("not found")
    _create_table(conn, "plain_table", schema=None)
    conn.create_table.assert_called_once_with("plain_table", schema=None)


def test_create_table_existing_with_schema_triggers_migration(
    tmp_path: Path, monkeypatch
) -> None:
    """_create_table should migrate existing table when schema is provided."""
    db_dir = tmp_path / "db"
    monkeypatch.setenv("LANCEDB_DIR", str(db_dir))
    conn = get_connection_from_env()

    conn.create_table("create_table_migrate", schema=pa.schema([("a", pa.int32())]))
    target_schema = pa.schema([("a", pa.int32()), ("b", pa.string())])

    _create_table(conn, "create_table_migrate", target_schema)

    schema = conn.open_table("create_table_migrate").schema
    assert "b" in schema.names


def test_ensure_embeddings_table_with_fixed_vector_dim(
    tmp_path: Path, monkeypatch
) -> None:
    """ensure_embeddings_table should use fixed-size list when vector_dim is set."""
    db_dir = tmp_path / "db"
    monkeypatch.setenv("LANCEDB_DIR", str(db_dir))
    conn = get_connection_from_env()

    ensure_embeddings_table(conn, "test_fixed", vector_dim=8)
    schema = conn.open_table("embeddings_test_fixed").schema
    vector_type = schema.field("vector").type
    assert getattr(vector_type, "list_size", None) == 8


def test_ensure_embeddings_table_with_variable_vector_dim(
    tmp_path: Path, monkeypatch
) -> None:
    """ensure_embeddings_table should use variable list when vector_dim is None."""
    db_dir = tmp_path / "db"
    monkeypatch.setenv("LANCEDB_DIR", str(db_dir))
    conn = get_connection_from_env()

    ensure_embeddings_table(conn, "test_variable", vector_dim=None)
    schema = conn.open_table("embeddings_test_variable").schema
    vector_type = schema.field("vector").type
    assert getattr(vector_type, "list_size", -1) == -1


def test_ensure_collection_config_table_create_and_idempotent(
    tmp_path: Path, monkeypatch
) -> None:
    """ensure_collection_config_table should be creatable and idempotent."""
    db_dir = tmp_path / "db"
    monkeypatch.setenv("LANCEDB_DIR", str(db_dir))
    conn = get_connection_from_env()

    ensure_collection_config_table(conn)
    schema_before = conn.open_table("collection_config").schema
    ensure_collection_config_table(conn)
    schema_after = conn.open_table("collection_config").schema

    assert schema_before == schema_after
    assert "config_json" in schema_after.names
    assert "user_id" in schema_after.names


def test_ensure_parses_table_migrates_missing_user_id(
    tmp_path: Path, monkeypatch
) -> None:
    """ensure_parses_table should add user_id for legacy schema."""
    db_dir = tmp_path / "db"
    monkeypatch.setenv("LANCEDB_DIR", str(db_dir))
    conn = get_connection_from_env()

    old_schema = pa.schema(
        [
            pa.field("collection", pa.string()),
            pa.field("doc_id", pa.string()),
            pa.field("parse_hash", pa.string()),
            pa.field("parser", pa.string()),
            pa.field("created_at", pa.timestamp("us")),
            pa.field("params_json", pa.string()),
            pa.field("parsed_content", pa.large_string()),
        ]
    )
    conn.create_table("parses", schema=old_schema)
    conn.open_table("parses").add(
        [
            {
                "collection": "c",
                "doc_id": "d1",
                "parse_hash": "p1",
                "parser": "basic",
                "created_at": None,
                "params_json": "{}",
                "parsed_content": "x",
            }
        ]
    )

    ensure_parses_table(conn)
    schema = conn.open_table("parses").schema
    assert "user_id" in schema.names


def test_ensure_prompt_templates_table_migrates_missing_user_id(
    tmp_path: Path, monkeypatch
) -> None:
    """ensure_prompt_templates_table should add user_id for legacy schema."""
    db_dir = tmp_path / "db"
    monkeypatch.setenv("LANCEDB_DIR", str(db_dir))
    conn = get_connection_from_env()

    old_schema = pa.schema(
        [
            pa.field("collection", pa.string()),
            pa.field("id", pa.string()),
            pa.field("name", pa.string()),
            pa.field("template", pa.string()),
            pa.field("version", pa.int64()),
            pa.field("is_latest", pa.bool_()),
            pa.field("metadata", pa.string()),
            pa.field("created_at", pa.timestamp("us")),
            pa.field("updated_at", pa.timestamp("us")),
        ]
    )
    conn.create_table("prompt_templates", schema=old_schema)
    conn.open_table("prompt_templates").add(
        [
            {
                "collection": "c",
                "id": "1",
                "name": "n",
                "template": "t",
                "version": 1,
                "is_latest": True,
                "metadata": "{}",
                "created_at": None,
                "updated_at": None,
            }
        ]
    )

    ensure_prompt_templates_table(conn)
    schema = conn.open_table("prompt_templates").schema
    assert "user_id" in schema.names


def test_ensure_ingestion_runs_table_migrates_missing_user_id(
    tmp_path: Path, monkeypatch
) -> None:
    """ensure_ingestion_runs_table should add user_id for legacy schema."""
    db_dir = tmp_path / "db"
    monkeypatch.setenv("LANCEDB_DIR", str(db_dir))
    conn = get_connection_from_env()

    old_schema = pa.schema(
        [
            pa.field("collection", pa.string()),
            pa.field("doc_id", pa.string()),
            pa.field("status", pa.string()),
            pa.field("message", pa.string()),
            pa.field("parse_hash", pa.string()),
            pa.field("created_at", pa.timestamp("us")),
            pa.field("updated_at", pa.timestamp("us")),
        ]
    )
    conn.create_table("ingestion_runs", schema=old_schema)
    conn.open_table("ingestion_runs").add(
        [
            {
                "collection": "c",
                "doc_id": "d",
                "status": "ok",
                "message": "",
                "parse_hash": "p",
                "created_at": None,
                "updated_at": None,
            }
        ]
    )

    ensure_ingestion_runs_table(conn)
    schema = conn.open_table("ingestion_runs").schema
    assert "user_id" in schema.names


def test_concurrent_ensure_collection_metadata_table_is_safe(
    tmp_path: Path, monkeypatch
) -> None:
    """Concurrent ensure_collection_metadata_table calls should be safe."""
    db_dir = tmp_path / "db"
    monkeypatch.setenv("LANCEDB_DIR", str(db_dir))
    conn = get_connection_from_env()

    errors: list[Exception] = []

    def _worker() -> None:
        try:
            ensure_collection_metadata_table(conn)
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [Thread(target=_worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    schema = conn.open_table("collection_metadata").schema
    assert "ingestion_config" in schema.names
