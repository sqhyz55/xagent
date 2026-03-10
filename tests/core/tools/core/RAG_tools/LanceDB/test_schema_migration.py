from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow as pa

from xagent.core.tools.core.RAG_tools.LanceDB.schema_manager import (
    _ensure_schema_fields,
    _get_sql_default_for_pa_type,
    ensure_collection_metadata_table,
    ensure_documents_table,
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
    assert "title" in field_names
    assert "language" in field_names
    assert "uploaded_at" in field_names

    # 4. Verify default values in existing data
    df = table.to_pandas()
    row = df.iloc[0]
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
