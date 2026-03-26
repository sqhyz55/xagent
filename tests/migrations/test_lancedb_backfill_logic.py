"""Tests for LanceDB user_id backfill migration logic.

This module verifies that the backfill migration correctly populates user_id
fields in chunks and embeddings tables by joining with the documents table.
"""

from __future__ import annotations

import tempfile

import lancedb
import pyarrow as pa
import pytest

from xagent.core.tools.core.RAG_tools.LanceDB.schema_manager import (
    ensure_documents_table,
)
from xagent.migrations.lancedb.backfill_user_id import backfill_all


@pytest.fixture
def temp_conn():
    """Create a temporary LanceDB connection."""
    with tempfile.TemporaryDirectory() as temp_dir:
        conn = lancedb.connect(temp_dir)
        yield conn


def test_backfill_logic_success(temp_conn):
    """Test that backfill correctly updates user_id from documents to chunks/embeddings."""
    conn = temp_conn

    # 1. Setup: Create documents table with data (including user_id)
    ensure_documents_table(conn)
    docs_table = conn.open_table("documents")
    docs_data = [
        {"collection": "c1", "doc_id": "doc1", "user_id": 101, "source_path": "p1"},
        {"collection": "c1", "doc_id": "doc2", "user_id": 102, "source_path": "p2"},
    ]
    docs_table.add(docs_data)

    # 2. Setup: Create chunks table with OLD schema (no user_id)
    old_chunks_schema = pa.schema(
        [
            pa.field("collection", pa.string()),
            pa.field("doc_id", pa.string()),
            pa.field("chunk_id", pa.string()),
            pa.field("parse_hash", pa.string()),
            pa.field("text", pa.large_string()),
        ]
    )
    chunks_table = conn.create_table("chunks", schema=old_chunks_schema)
    chunks_data = [
        {
            "collection": "c1",
            "doc_id": "doc1",
            "chunk_id": "chk1",
            "parse_hash": "h1",
            "text": "t1",
        },
        {
            "collection": "c1",
            "doc_id": "doc2",
            "chunk_id": "chk2",
            "parse_hash": "h2",
            "text": "t2",
        },
        {
            "collection": "c1",
            "doc_id": "doc3",
            "chunk_id": "chk3",
            "parse_hash": "h3",
            "text": "t3",
        },  # doc3 doesn't exist in docs table
    ]
    chunks_table.add(chunks_data)

    # 3. Setup: Create embeddings table with OLD schema
    old_emb_schema = pa.schema(
        [
            pa.field("collection", pa.string()),
            pa.field("doc_id", pa.string()),
            pa.field("chunk_id", pa.string()),
            pa.field("parse_hash", pa.string()),
            pa.field("model", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), 2)),
        ]
    )
    emb_table = conn.create_table("embeddings_test_model", schema=old_emb_schema)
    emb_data = [
        {
            "collection": "c1",
            "doc_id": "doc1",
            "chunk_id": "chk1",
            "parse_hash": "h1",
            "model": "m1",
            "vector": [0.1, 0.2],
        },
    ]
    emb_table.add(emb_data)

    # 4. Simulate the first step of migration: adding the column as NULL
    # This mimics what schema_manager.py does now
    chunks_table.add_columns({"user_id": "cast(null as bigint)"})
    emb_table.add_columns({"user_id": "cast(null as bigint)"})

    # Verify they are currently NULL
    assert (
        chunks_table.search().where("user_id IS NULL").to_list()[0]["user_id"] is None
    )

    # 5. Run the backfill migration
    result = backfill_all(dry_run=False, conn=conn)

    # Re-open table to ensure we are seeing persisted state
    chunks_table = conn.open_table("chunks")

    # 6. Verifications
    assert result["chunks"]["backfilled"] == 2
    assert result["chunks"]["skipped"] == 1  # doc3 skipped
    assert result["embeddings"]["backfilled"] == 1

    # Check actual data in chunks
    updated_chunks = chunks_table.search().to_list()
    chunk_map = {c["chunk_id"]: c["user_id"] for c in updated_chunks}
    assert chunk_map["chk1"] == 101
    assert chunk_map["chk2"] == 102
    assert chunk_map["chk3"] == -2

    # Check actual data in embeddings
    emb_table = conn.open_table("embeddings_test_model")
    updated_emb = emb_table.search().to_list()
    assert updated_emb[0]["user_id"] == 101
