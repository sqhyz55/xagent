"""Tests for LanceDB-backed storage implementations."""

import asyncio
from datetime import datetime, timezone
from unittest.mock import Mock, patch

import pytest

from xagent.core.tools.core.RAG_tools.storage.lancedb_stores import (
    LanceDBMetadataStore,
    LanceDBVectorIndexStore,
)


@patch(
    "xagent.core.tools.core.RAG_tools.storage.lancedb_stores.get_connection_from_env"
)
def test_metadata_store_save_collection_config(mock_get_connection: Mock) -> None:
    """Metadata store should save collection config correctly."""
    mock_conn = Mock()
    mock_get_connection.return_value = mock_conn

    mock_table = Mock()
    mock_conn.open_table.return_value = mock_table

    store = LanceDBMetadataStore()
    asyncio.run(
        store.save_collection_config(
            collection="test_collection",
            config_json='{"parse_method": "default"}',
            user_id=1,
        )
    )

    # Verify table.delete was called to remove existing config
    mock_table.delete.assert_called_once()

    # Verify table.add was called with new config
    mock_table.add.assert_called_once()
    added_data = mock_table.add.call_args[0][0]
    assert added_data[0]["collection"] == "test_collection"
    assert added_data[0]["config_json"] == '{"parse_method": "default"}'
    assert added_data[0]["user_id"] == 1


@patch(
    "xagent.core.tools.core.RAG_tools.storage.lancedb_stores.get_connection_from_env"
)
def test_metadata_store_get_collection_config_success(
    mock_get_connection: Mock,
) -> None:
    """Metadata store should retrieve collection config correctly."""
    mock_conn = Mock()
    mock_get_connection.return_value = mock_conn

    mock_table = Mock()
    mock_conn.open_table.return_value = mock_table

    # Mock pandas DataFrame with iloc[0]["config_json"] access pattern
    mock_row = Mock()
    mock_row.__getitem__ = Mock(return_value='{"parse_method": "default"}')

    mock_result = Mock()
    mock_result.empty = False
    mock_result.iloc = [mock_row]

    mock_table.search.return_value.where.return_value.to_pandas.return_value = (
        mock_result
    )

    store = LanceDBMetadataStore()
    config = asyncio.run(
        store.get_collection_config(collection="test_collection", user_id=1)
    )

    assert config == '{"parse_method": "default"}'


@patch(
    "xagent.core.tools.core.RAG_tools.storage.lancedb_stores.get_connection_from_env"
)
def test_metadata_store_get_collection_config_not_found(
    mock_get_connection: Mock,
) -> None:
    """Metadata store should return None when config not found."""
    mock_conn = Mock()
    mock_get_connection.return_value = mock_conn

    mock_table = Mock()
    mock_conn.open_table.return_value = mock_table
    mock_result = Mock()
    mock_result.empty = True
    mock_table.search.return_value.where.return_value.to_pandas.return_value = (
        mock_result
    )

    store = LanceDBMetadataStore()
    config = asyncio.run(
        store.get_collection_config(collection="test_collection", user_id=1)
    )

    assert config is None


@patch(
    "xagent.core.tools.core.RAG_tools.storage.lancedb_stores.get_connection_from_env"
)
def test_metadata_store_get_collection_success(mock_get_connection: Mock) -> None:
    """Metadata store should deserialize collection metadata correctly."""
    mock_conn = Mock()
    mock_get_connection.return_value = mock_conn

    mock_table = Mock()
    mock_conn.open_table.return_value = mock_table
    mock_result = Mock()
    mock_result.empty = False
    mock_result.iloc = [
        Mock(
            to_dict=Mock(
                return_value={
                    "name": "test_collection",
                    "schema_version": "1.0.0",
                    "embedding_model_id": "text-embedding-v4",
                    "embedding_dimension": 1024,
                    "documents": 2,
                    "processed_documents": 2,
                    "parses": 2,
                    "chunks": 8,
                    "embeddings": 8,
                    "document_names": '["a.pdf","b.pdf"]',
                    "collection_locked": False,
                    "allow_mixed_parse_methods": False,
                    "skip_config_validation": False,
                    "created_at": datetime.now(timezone.utc).replace(tzinfo=None),
                    "updated_at": datetime.now(timezone.utc).replace(tzinfo=None),
                    "last_accessed_at": datetime.now(timezone.utc).replace(tzinfo=None),
                    "extra_metadata": "{}",
                }
            )
        )
    ]
    mock_table.search.return_value.where.return_value.to_pandas.return_value = (
        mock_result
    )

    store = LanceDBMetadataStore()
    collection = asyncio.run(store.get_collection("test_collection"))
    assert collection.name == "test_collection"
    assert collection.documents == 2
    assert collection.document_names == ["a.pdf", "b.pdf"]


@patch(
    "xagent.core.tools.core.RAG_tools.storage.lancedb_stores.UserPermissions.get_user_filter"
)
@patch("xagent.core.tools.core.RAG_tools.storage.lancedb_stores.query_to_list")
@patch(
    "xagent.core.tools.core.RAG_tools.storage.lancedb_stores.get_connection_from_env"
)
def test_vector_store_list_document_records_filters_and_maps(
    mock_get_connection: Mock,
    mock_query_to_list: Mock,
    mock_user_filter: Mock,
) -> None:
    """Vector store should apply combined filter and map to DocumentRecord."""
    mock_conn = Mock()
    mock_get_connection.return_value = mock_conn

    mock_user_filter.return_value = "user_id == 1"
    mock_table = Mock()
    mock_conn.open_table.return_value = mock_table
    mock_query_to_list.return_value = [
        {"doc_id": "doc-1", "source_path": "/tmp/a.pdf"},
        {"doc_id": "doc-2", "source_path": None},
    ]

    store = LanceDBVectorIndexStore()
    records = store.list_document_records(
        collection_name="kb1",
        user_id=1,
        is_admin=False,
        max_results=50,
    )

    assert [r.doc_id for r in records] == ["doc-1", "doc-2"]
    assert records[0].source_path == "/tmp/a.pdf"
    mock_table.search.return_value.where.assert_called_once()


@patch(
    "xagent.core.tools.core.RAG_tools.storage.lancedb_stores.get_connection_from_env"
)
def test_vector_store_rename_collection_data_updates_expected_tables(
    mock_get_connection: Mock,
) -> None:
    """Rename should update core and embeddings tables only."""
    mock_conn = Mock()
    mock_get_connection.return_value = mock_conn
    mock_conn.table_names.return_value = [
        "documents",
        "parses",
        "chunks",
        "embeddings_text_embedding_v4",
        "collection_metadata",
    ]
    mock_table = Mock()
    mock_conn.open_table.return_value = mock_table

    store = LanceDBVectorIndexStore()
    warnings = store.rename_collection_data("old_name", "new_name")

    assert warnings == []
    # 4 target tables should be updated; control-plane table excluded.
    assert mock_table.update.call_count == 4


# --- Upsert Fallback Tests ---


@patch(
    "xagent.core.tools.core.RAG_tools.storage.lancedb_stores.get_connection_from_env"
)
def test_upsert_embeddings_merge_insert_success(mock_get_connection: Mock) -> None:
    """Test successful merge_insert upsert."""
    mock_conn = Mock()
    mock_get_connection.return_value = mock_conn

    mock_table = Mock()
    mock_conn.open_table.return_value = mock_table

    # Mock merge_insert chain
    mock_merge_insert = Mock()
    mock_when_matched = Mock()
    mock_when_not_matched = Mock()
    mock_table.merge_insert.return_value = mock_merge_insert
    mock_merge_insert.when_matched_update_all.return_value = mock_when_matched
    mock_when_matched.when_not_matched_insert_all.return_value = mock_when_not_matched
    mock_when_not_matched.execute.return_value = None

    store = LanceDBVectorIndexStore()

    records = [
        {
            "collection": "test_col",
            "doc_id": "doc1",
            "chunk_id": "chunk1",
            "vector": [0.1, 0.2],
            "text": "test",
        }
    ]

    store.upsert_embeddings("text_embedding_v4", records)

    # Verify merge_insert was called
    mock_table.merge_insert.assert_called_once_with(["collection", "doc_id", "chunk_id"])
    mock_merge_insert.when_matched_update_all.assert_called_once()
    mock_when_matched.when_not_matched_insert_all.assert_called_once()
    mock_when_not_matched.execute.assert_called_once()

    # Verify add was NOT called (no fallback needed)
    mock_table.add.assert_not_called()


@patch(
    "xagent.core.tools.core.RAG_tools.storage.lancedb_stores.get_connection_from_env"
)
def test_upsert_embeddings_merge_insert_fallback_to_add(mock_get_connection: Mock) -> None:
    """Test fallback to add() when merge_insert fails with recoverable error."""
    mock_conn = Mock()
    mock_get_connection.return_value = mock_conn

    mock_table = Mock()
    mock_conn.open_table.return_value = mock_table

    # Mock merge_insert chain that fails
    mock_merge_insert = Mock()
    mock_when_matched = Mock()
    mock_when_not_matched = Mock()
    mock_table.merge_insert.return_value = mock_merge_insert
    mock_merge_insert.when_matched_update_all.return_value = mock_when_matched
    mock_when_matched.when_not_matched_insert_all.return_value = mock_when_not_matched
    # merge_insert fails with recoverable error (e.g., network issue)
    mock_when_not_matched.execute.side_effect = Exception("Temporary network error")

    # Mock add() to succeed
    mock_table.add.return_value = None

    store = LanceDBVectorIndexStore()

    records = [
        {
            "collection": "test_col",
            "doc_id": "doc1",
            "chunk_id": "chunk1",
            "vector": [0.1, 0.2],
            "text": "test",
        }
    ]

    store.upsert_embeddings("text_embedding_v4", records)

    # Verify merge_insert was attempted
    mock_table.merge_insert.assert_called_once()

    # Verify fallback to add() was used
    mock_table.add.assert_called_once()


@patch(
    "xagent.core.tools.core.RAG_tools.storage.lancedb_stores.get_connection_from_env"
)
def test_upsert_embeddings_non_recoverable_error_no_fallback(mock_get_connection: Mock) -> None:
    """Test that non-recoverable errors (schema, type mismatch) do not fallback."""
    mock_conn = Mock()
    mock_get_connection.return_value = mock_conn

    mock_table = Mock()
    mock_conn.open_table.return_value = mock_table

    # Mock merge_insert chain that fails with non-recoverable error
    mock_merge_insert = Mock()
    mock_when_matched = Mock()
    mock_when_not_matched = Mock()
    mock_table.merge_insert.return_value = mock_merge_insert
    mock_merge_insert.when_matched_update_all.return_value = mock_when_matched
    mock_when_matched.when_not_matched_insert_all.return_value = mock_when_not_matched
    # Schema error - should NOT fallback
    mock_when_not_matched.execute.side_effect = ValueError("Schema mismatch")

    store = LanceDBVectorIndexStore()

    records = [
        {
            "collection": "test_col",
            "doc_id": "doc1",
            "chunk_id": "chunk1",
            "vector": [0.1, 0.2],
            "text": "test",
        }
    ]

    # Should raise ValueError without fallback
    with pytest.raises(ValueError, match="Schema mismatch"):
        store.upsert_embeddings("text_embedding_v4", records)

    # Verify merge_insert was attempted
    mock_table.merge_insert.assert_called_once()

    # Verify add() was NOT called (no fallback for non-recoverable errors)
    mock_table.add.assert_not_called()


@patch(
    "xagent.core.tools.core.RAG_tools.storage.lancedb_stores.get_connection_from_env"
)
def test_upsert_embeddings_both_methods_fail(mock_get_connection: Mock) -> None:
    """Test that error is raised when both merge_insert and add() fail."""
    mock_conn = Mock()
    mock_get_connection.return_value = mock_conn

    mock_table = Mock()
    mock_conn.open_table.return_value = mock_table

    # Mock merge_insert chain that fails
    mock_merge_insert = Mock()
    mock_when_matched = Mock()
    mock_when_not_matched = Mock()
    mock_table.merge_insert.return_value = mock_merge_insert
    mock_merge_insert.when_matched_update_all.return_value = mock_when_matched
    mock_when_matched.when_not_matched_insert_all.return_value = mock_when_not_matched
    mock_when_not_matched.execute.side_effect = Exception("merge_insert failed")

    # Mock add() to also fail
    mock_table.add.side_effect = Exception("add() also failed")

    store = LanceDBVectorIndexStore()

    records = [
        {
            "collection": "test_col",
            "doc_id": "doc1",
            "chunk_id": "chunk1",
            "vector": [0.1, 0.2],
            "text": "test",
        }
    ]

    # Should raise when both methods fail
    with pytest.raises(Exception, match="add.*also failed"):
        store.upsert_embeddings("text_embedding_v4", records)

    # Verify both methods were attempted
    mock_table.merge_insert.assert_called_once()
    mock_table.add.assert_called_once()

