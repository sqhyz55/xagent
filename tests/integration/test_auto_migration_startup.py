"""Integration tests for automatic LanceDB migration on application startup.

This module tests the automatic migration logic that runs during application
startup to add user_id fields to existing LanceDB tables.
"""

from __future__ import annotations

import os
import tempfile

import pyarrow as pa
import pytest

from xagent.core.tools.core.RAG_tools.LanceDB.schema_manager import (
    check_table_needs_migration,
)
from xagent.providers.vector_store.lancedb import get_connection_from_env


@pytest.fixture
def temp_lancedb_dir():
    """Create a temporary directory for LanceDB."""
    with tempfile.TemporaryDirectory() as temp_dir:
        original_env = os.environ.get("LANCEDB_DIR")
        os.environ["LANCEDB_DIR"] = temp_dir
        yield temp_dir
        if original_env is not None:
            os.environ["LANCEDB_DIR"] = original_env
        else:
            os.environ.pop("LANCEDB_DIR", None)


def test_auto_migration_detects_old_schema(temp_lancedb_dir):
    """Test that auto-migration detects tables with old schema (missing user_id)."""
    conn = get_connection_from_env()

    # Create tables with old schema (without user_id)
    old_chunks_schema = pa.schema(
        [
            pa.field("collection", pa.string()),
            pa.field("doc_id", pa.string()),
            pa.field("chunk_id", pa.string()),
            pa.field("text", pa.large_string()),
            pa.field("metadata", pa.string()),
        ]
    )
    conn.create_table("chunks", schema=old_chunks_schema)

    old_docs_schema = pa.schema(
        [
            pa.field("collection", pa.string()),
            pa.field("doc_id", pa.string()),
            pa.field("source_path", pa.string()),
        ]
    )
    conn.create_table("documents", schema=old_docs_schema)

    # Verify that migration is needed
    assert check_table_needs_migration(conn, "chunks") is True
    assert check_table_needs_migration(conn, "documents") is True


def test_auto_migration_skips_new_schema(temp_lancedb_dir):
    """Test that auto-migration skips tables with new schema (has user_id)."""
    conn = get_connection_from_env()

    # Create tables with new schema (with user_id)
    new_chunks_schema = pa.schema(
        [
            pa.field("collection", pa.string()),
            pa.field("doc_id", pa.string()),
            pa.field("chunk_id", pa.string()),
            pa.field("text", pa.large_string()),
            pa.field("metadata", pa.string()),
            pa.field("user_id", pa.int32(), nullable=True),
        ]
    )
    conn.create_table("chunks", schema=new_chunks_schema)

    # Verify that no migration is needed
    assert check_table_needs_migration(conn, "chunks") is False


def test_auto_migration_handles_missing_tables(temp_lancedb_dir):
    """Test that auto-migration handles non-existent tables gracefully."""
    conn = get_connection_from_env()

    # Check non-existent tables
    assert check_table_needs_migration(conn, "nonexistent_table") is False
    assert check_table_needs_migration(conn, "chunks") is False
    assert check_table_needs_migration(conn, "documents") is False


def test_migration_detection_for_multiple_tables(temp_lancedb_dir):
    """Test that migration detection works correctly for multiple tables."""
    conn = get_connection_from_env()

    # Create multiple tables with old schema
    old_schema = pa.schema(
        [
            pa.field("collection", pa.string()),
            pa.field("doc_id", pa.string()),
            pa.field("text", pa.string()),
        ]
    )

    conn.create_table("chunks", schema=old_schema)
    conn.create_table("documents", schema=old_schema)
    conn.create_table("parses", schema=old_schema)

    # All should need migration
    assert check_table_needs_migration(conn, "chunks") is True
    assert check_table_needs_migration(conn, "documents") is True
    assert check_table_needs_migration(conn, "parses") is True

    # Non-existent table should not need migration
    assert check_table_needs_migration(conn, "nonexistent") is False


def test_auto_migration_handles_embeddings_tables(temp_lancedb_dir):
    """Test that auto-migration detects embeddings tables correctly."""
    conn = get_connection_from_env()

    # Create embeddings table with old schema
    old_embeddings_schema = pa.schema(
        [
            pa.field("collection", pa.string()),
            pa.field("doc_id", pa.string()),
            pa.field("chunk_id", pa.string()),
            pa.field("vector", pa.list_(pa.float32())),
            pa.field("text", pa.large_string()),
        ]
    )
    conn.create_table("embeddings_test_model", schema=old_embeddings_schema)

    # Verify that migration is needed
    assert check_table_needs_migration(conn, "embeddings_test_model") is True

    # Create embeddings table with new schema
    new_embeddings_schema = pa.schema(
        [
            pa.field("collection", pa.string()),
            pa.field("doc_id", pa.string()),
            pa.field("chunk_id", pa.string()),
            pa.field("vector", pa.list_(pa.float32())),
            pa.field("text", pa.large_string()),
            pa.field("user_id", pa.int32(), nullable=True),
        ]
    )
    conn.create_table("embeddings_test_model_new", schema=new_embeddings_schema)

    # Verify that no migration is needed
    assert check_table_needs_migration(conn, "embeddings_test_model_new") is False
