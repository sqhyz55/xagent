"""Tests for collection manager functionality."""

import asyncio
from unittest.mock import AsyncMock, Mock, patch

import pytest

from xagent.core.tools.core.RAG_tools.core.schemas import CollectionInfo
from xagent.core.tools.core.RAG_tools.management.collection_manager import (
    CollectionManager,
    get_collection_sync,
    update_collection_stats_sync,
)


@pytest.fixture
def sample_collection():
    """Create a sample CollectionInfo."""
    return CollectionInfo(
        name="test_collection",
        embedding_model_id="text-embedding-ada-002",
        embedding_dimension=1536,
        documents=5,
        processed_documents=3,
        embeddings=20,
    )


class TestCollectionManager:
    """Test CollectionManager class."""

    @pytest.fixture
    def manager(self):
        """Create a CollectionManager instance."""
        return CollectionManager()

    @pytest.mark.asyncio
    async def test_get_collection_success(self, manager):
        """Test successful collection retrieval."""
        # Mock connection and table
        mock_connection = Mock()
        mock_table = Mock()
        mock_result = Mock()

        # Set up the mock chain
        mock_connection.open_table.return_value = mock_table
        mock_table.search.return_value.where.return_value.to_pandas.return_value = (
            mock_result
        )

        # Mock data
        mock_data = {
            "name": "test_collection",
            "schema_version": "1.0.0",
            "embedding_model_id": "text-embedding-ada-002",
            "embedding_dimension": 1536,
            "documents": 5,
            "processed_documents": 3,
            "document_names": '["doc1.pdf", "doc2.md"]',
        }
        mock_result.empty = False
        mock_result.iloc = [Mock(to_dict=Mock(return_value=mock_data))]

        # Mock the _get_connection method
        with patch.object(
            manager,
            "_get_connection",
            new_callable=AsyncMock,
            return_value=mock_connection,
        ):
            result = await manager.get_collection("test_collection")

        assert result.name == "test_collection"
        assert result.embedding_model_id == "text-embedding-ada-002"
        assert result.embedding_dimension == 1536
        assert result.documents == 5
        assert result.processed_documents == 3
        assert result.document_names == ["doc1.pdf", "doc2.md"]

    @pytest.mark.asyncio
    async def test_get_collection_not_found(self, manager):
        """Test collection retrieval when not found."""
        mock_connection = Mock()
        mock_table = Mock()
        mock_result = Mock()

        # Set up the mock chain
        mock_connection.open_table.return_value = mock_table
        mock_table.search.return_value.where.return_value.to_pandas.return_value = (
            mock_result
        )

        # Mock empty result
        mock_result.empty = True

        with patch.object(
            manager,
            "_get_connection",
            new_callable=AsyncMock,
            return_value=mock_connection,
        ):
            with pytest.raises(
                ValueError, match="Collection 'test_collection' not found"
            ):
                await manager.get_collection("test_collection")

    @pytest.mark.asyncio
    async def test_save_collection_success(self, manager, sample_collection):
        """Test successful collection saving."""
        mock_connection = Mock()
        mock_table = Mock()
        mock_connection.open_table.return_value = mock_table
        mock_table.add = Mock()

        with patch.object(
            manager,
            "_get_connection",
            new_callable=AsyncMock,
            return_value=mock_connection,
        ):
            await manager.save_collection(sample_collection)

        # Verify upsert was called
        mock_table.add.assert_called_once()
        call_args = mock_table.add.call_args
        # We check only data since mode might vary or be tested separately
        assert len(call_args[0]) > 0

    @pytest.mark.asyncio
    async def test_initialize_collection_embedding_success(self, manager):
        """Test successful collection embedding initialization."""
        # Mock connection for get_collection calls
        mock_connection = Mock()
        mock_table = Mock()
        mock_result = Mock()
        mock_connection.open_table.return_value = mock_table
        mock_table.search.return_value.where.return_value.to_pandas.return_value = (
            mock_result
        )

        # Mock data for existing collection
        mock_data = {
            "name": "test_collection",
            "schema_version": "1.0.0",
            "embedding_model_id": None,
            "embedding_dimension": None,
            "documents": 0,
            "processed_documents": 0,
            "document_names": "[]",
        }
        mock_result.empty = False
        mock_result.iloc = [Mock(to_dict=Mock(return_value=mock_data))]

        # Mock embedding adapter resolution
        mock_config = Mock()
        mock_config.dimension = 1536
        mock_resolve = Mock(return_value=(mock_config, Mock()))

        with patch.object(
            manager,
            "_get_connection",
            new_callable=AsyncMock,
            return_value=mock_connection,
        ):
            with patch.object(manager, "_save_collection_with_retry") as mock_save:
                with patch(
                    "xagent.core.tools.core.RAG_tools.management.collection_manager.resolve_embedding_adapter",
                    mock_resolve,
                ):
                    result = await manager.initialize_collection_embedding(
                        "test_collection", "text-embedding-ada-002"
                    )

                assert result.name == "test_collection"
                assert result.embedding_model_id == "text-embedding-ada-002"
                assert result.embedding_dimension == 1536
                mock_save.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_collection_stats_success(self, manager):
        """Test successful collection stats update."""
        with patch.object(manager, "get_collection") as mock_get:
            existing = CollectionInfo(
                name="test_collection", documents=5, processed_documents=3
            )
            mock_get.return_value = existing

            with patch.object(manager, "_save_collection_with_retry") as mock_save:
                result = await manager.update_collection_stats(
                    "test_collection",
                    documents_delta=1,
                    processed_documents_delta=1,
                    embeddings_delta=100,
                    document_name="new_doc.pdf",
                )

                assert result.documents == 6
                assert result.processed_documents == 4
                assert result.embeddings == 100
                assert "new_doc.pdf" in result.document_names
                mock_save.assert_called_once()


class TestSyncFunctions:
    """Test synchronous wrapper functions."""

    @patch(
        "xagent.core.tools.core.RAG_tools.management.collection_manager.collection_manager"
    )
    def test_get_collection_sync(self, mock_manager):
        """Test synchronous collection retrieval."""
        mock_manager.get_collection = AsyncMock(return_value="mock_result")

        result = get_collection_sync("test_collection")

        assert result == "mock_result"
        mock_manager.get_collection.assert_called_once_with("test_collection")

    @patch(
        "xagent.core.tools.core.RAG_tools.management.collection_manager._run_in_separate_loop"
    )
    def test_update_collection_stats_sync(self, mock_run_loop):
        """Test synchronous collection stats update."""
        # Create a mock CollectionInfo to return
        mock_collection = CollectionInfo(name="test", documents=1)
        # Execute the passed coroutine to avoid "coroutine was never awaited" warnings.
        mock_run_loop.side_effect = lambda coro: asyncio.run(coro)

        with patch(
            "xagent.core.tools.core.RAG_tools.management.collection_manager.collection_manager"
        ) as mock_manager:
            mock_manager.update_collection_stats = AsyncMock(
                return_value=mock_collection
            )
            result = update_collection_stats_sync("test", documents_delta=1)

        assert result == mock_collection
        mock_run_loop.assert_called_once()


class TestCollectionInfoProperties:
    """Test CollectionInfo properties and methods."""

    def test_is_initialized_true(self):
        """Test is_initialized property when both fields are set."""
        collection = CollectionInfo(
            name="test", embedding_model_id="model-1", embedding_dimension=512
        )
        assert collection.is_initialized is True

    def test_from_storage_basic(self):
        """Test from_storage with basic data."""
        data = {
            "name": "test_collection",
            "schema_version": "1.0.0",
            "embedding_model_id": "text-embedding-ada-002",
            "embedding_dimension": 1536,
            "documents": 5,
            "processed_documents": 3,
            "document_names": '["doc1.pdf"]',
            "extra_metadata": '{"key": "value"}',
        }

        result = CollectionInfo.from_storage(data)

        assert result.name == "test_collection"
        assert result.embedding_model_id == "text-embedding-ada-002"
        assert result.embedding_dimension == 1536
        assert result.documents == 5
        assert result.processed_documents == 3
        assert result.document_names == ["doc1.pdf"]
        assert result.extra_metadata == {"key": "value"}

    def test_to_storage_basic(self, sample_collection):
        """Test to_storage serialization."""
        data = sample_collection.to_storage()

        assert data["name"] == "test_collection"
        assert data["embedding_model_id"] == "text-embedding-ada-002"
        assert data["embedding_dimension"] == 1536
        assert data["documents"] == 5
        assert data["processed_documents"] == 3
        # Complex types should be JSON strings
        assert isinstance(data["document_names"], str)
        assert isinstance(data["extra_metadata"], str)
