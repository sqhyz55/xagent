"""Tests for collection manager functionality."""

import asyncio
from unittest.mock import AsyncMock, Mock, patch

import pytest

from xagent.core.tools.core.RAG_tools.core.schemas import CollectionInfo
from xagent.core.tools.core.RAG_tools.management.collection_manager import (
    CollectionManager,
    get_collection_sync,
    resolve_effective_embedding_model_sync,
    update_collection_stats_sync,
)
from xagent.core.tools.core.RAG_tools.utils.tag_mapping import register_tag_mapping


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
    """Test CollectionManager class with real storage layer."""

    @pytest.fixture
    def manager(self):
        """Create a CollectionManager instance with real storage."""
        # The isolate_lancedb_dir fixture in conftest.py already handles directory isolation
        return CollectionManager()

    @pytest.mark.asyncio
    async def test_get_collection_success(self, manager):
        """Test successful collection retrieval from real storage."""
        expected = CollectionInfo(
            name="test_collection",
            embedding_model_id="text-embedding-ada-002",
            embedding_dimension=1536,
            documents=5,
            processed_documents=3,
            document_names=["doc1.pdf", "doc2.md"],
        )

        # Save to real storage first
        await manager.save_collection(expected)

        # Retrieve and verify
        result = await manager.get_collection("test_collection")

        assert result.name == "test_collection"
        assert result.embedding_model_id == "text-embedding-ada-002"
        assert result.embedding_dimension == 1536
        assert result.documents == 5
        assert result.processed_documents == 3
        assert sorted(result.document_names) == sorted(["doc1.pdf", "doc2.md"])

    @pytest.mark.asyncio
    async def test_get_collection_not_found(self, manager):
        """Test collection retrieval when not found in real storage."""
        with pytest.raises(ValueError, match="Collection 'non_existent' not found"):
            await manager.get_collection("non_existent")

    @pytest.mark.asyncio
    async def test_save_collection_success(self, manager, sample_collection):
        """Test successful collection saving to real storage."""
        await manager.save_collection(sample_collection)

        # Verify it was actually saved
        saved = await manager.get_collection(sample_collection.name)
        assert saved.name == sample_collection.name
        assert saved.embedding_model_id == sample_collection.embedding_model_id

    @pytest.mark.asyncio
    async def test_initialize_collection_embedding_success(self, manager):
        """Test successful collection embedding initialization with real storage."""
        # Create and save initial collection
        collection_name = "init_test"
        initial = CollectionInfo(
            name=collection_name,
            embedding_model_id=None,
            embedding_dimension=None,
        )
        await manager.save_collection(initial)

        # Mock embedding adapter resolution (keep this mock as it involves external model logic)
        mock_config = Mock()
        mock_config.id = "text-embedding-ada-002"
        mock_config.dimension = 1536
        mock_resolve = Mock(return_value=(mock_config, Mock()))

        with patch(
            "xagent.core.tools.core.RAG_tools.management.collection_manager.resolve_embedding_adapter",
            mock_resolve,
        ):
            result = await manager.initialize_collection_embedding(
                collection_name, "text-embedding-ada-002"
            )

        assert result.name == collection_name
        assert result.embedding_model_id == "text-embedding-ada-002"
        assert result.embedding_dimension == 1536

        # Verify persistence
        saved = await manager.get_collection(collection_name)
        assert saved.embedding_model_id == "text-embedding-ada-002"

    @pytest.mark.asyncio
    async def test_update_collection_stats_success(self, manager):
        """Test successful collection stats update in real storage."""
        collection_name = "stats_test"
        initial = CollectionInfo(
            name=collection_name, documents=5, processed_documents=3
        )
        await manager.save_collection(initial)

        result = await manager.update_collection_stats(
            collection_name,
            documents_delta=1,
            processed_documents_delta=1,
            embeddings_delta=100,
            document_name="new_doc.pdf",
        )

        assert result.documents == 6
        assert result.processed_documents == 4
        assert result.embeddings == 100
        assert "new_doc.pdf" in result.document_names

        # Verify persistence
        saved = await manager.get_collection(collection_name)
        assert saved.documents == 6
        assert "new_doc.pdf" in saved.document_names


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


class TestHubTagMapping:
    """Test collection-manager hub tag mapping collision handling."""

    def test_register_hub_tag_mapping_warns_on_collision(self) -> None:
        mapping = {"OPENAI_text_embedding_3_large": ("hub-id-a", 1024)}
        mock_logger = Mock()

        register_tag_mapping(
            mapping,
            "OPENAI_text_embedding_3_large",
            ("hub-id-b", 1536),
            get_identity=lambda item: item[0],
            logger=mock_logger,
        )

        assert mapping["OPENAI_text_embedding_3_large"] == ("hub-id-a", 1024)
        mock_logger.warning.assert_called_once_with(
            "Tag collision: %s -> %s vs %s",
            "OPENAI_text_embedding_3_large",
            "hub-id-a",
            "hub-id-b",
        )


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


class TestResolveEffectiveEmbeddingModel:
    """Test resolve_effective_embedding_model_sync edge cases."""

    @patch(
        "xagent.core.tools.core.RAG_tools.management.collection_manager.mark_collection_accessed_sync"
    )
    @patch(
        "xagent.core.tools.core.RAG_tools.management.collection_manager.get_collection_sync"
    )
    def test_empty_bound_model_falls_back_to_config(
        self, mock_get_collection: Mock, _mock_mark: Mock
    ) -> None:
        """Empty bound model ID should be treated as missing and use config fallback."""
        mock_get_collection.return_value = CollectionInfo(
            name="test_collection",
            embedding_model_id="",
            embedding_dimension=1536,
        )

        resolved = resolve_effective_embedding_model_sync(
            "test_collection", config_model_id="text-embedding-v4"
        )
        assert resolved == "text-embedding-v4"


# --- rebuild_collection_metadata Tests (Issue #14) ---


class TestRebuildCollectionMetadata:
    """Test rebuild_collection_metadata function."""

    @patch(
        "xagent.core.tools.core.RAG_tools.management.collection_manager.get_vector_index_store"
    )
    @patch("xagent.core.tools.core.RAG_tools.management.collections")
    @pytest.mark.asyncio
    async def test_rebuild_with_embeddings_and_dimension(
        self, mock_collections_module, mock_get_vector_store
    ):
        """Test rebuild with embeddings table and vector dimension."""
        from types import SimpleNamespace

        # Mock collections.list_collections response (async)
        async def mock_list_collections(**kwargs):
            mock_collection = SimpleNamespace(
                name="test_collection",
                embeddings=10,
                model_copy=lambda update: SimpleNamespace(
                    name="test_collection",
                    embedding_model_id="test-model",
                    embedding_dimension=1536,
                ),
            )
            return SimpleNamespace(status="success", collections=[mock_collection])

        mock_collections_module.list_collections = mock_list_collections

        # Mock vector_store.list_table_names
        mock_vector_store = Mock()
        mock_get_vector_store.return_value = mock_vector_store
        mock_vector_store.list_table_names.return_value = [
            "documents",
            "chunks",
            "embeddings_test_model",
        ]

        # Mock count_rows_or_zero - only embeddings table has data
        mock_vector_store.count_rows_or_zero.side_effect = (
            lambda table_name, **kwargs: (
                10 if table_name == "embeddings_test_model" else 0
            )
        )

        # Mock get_vector_dimension
        mock_vector_store.get_vector_dimension.return_value = 1536

        from xagent.core.tools.core.RAG_tools.management.collection_manager import (
            rebuild_collection_metadata,
        )

        await rebuild_collection_metadata()

        # Verify count_rows_or_zero was called
        assert mock_vector_store.count_rows_or_zero.called
        # Verify get_vector_dimension was called
        mock_vector_store.get_vector_dimension.assert_called_with(
            "embeddings_test_model"
        )

    @patch(
        "xagent.core.tools.core.RAG_tools.management.collection_manager.get_vector_index_store"
    )
    @patch("xagent.core.tools.core.RAG_tools.management.collections")
    @pytest.mark.asyncio
    async def test_rebuild_no_embeddings(
        self, mock_collections_module, mock_get_vector_store
    ):
        """Test rebuild with collection having no embeddings."""
        from types import SimpleNamespace

        # Mock collection with no embeddings
        mock_collection = SimpleNamespace(
            name="empty_collection",
            embeddings=0,
            model_copy=lambda update: SimpleNamespace(
                name="empty_collection",
                embedding_model_id=None,
                embedding_dimension=None,
            ),
        )
        mock_result = SimpleNamespace(status="success", collections=[mock_collection])

        async def mock_list_collections(**kwargs):
            return mock_result

        mock_collections_module.list_collections = mock_list_collections

        # Mock vector_store
        mock_vector_store = Mock()
        mock_get_vector_store.return_value = mock_vector_store
        mock_vector_store.list_table_names.return_value = ["documents"]

        from xagent.core.tools.core.RAG_tools.management.collection_manager import (
            rebuild_collection_metadata,
        )

        await rebuild_collection_metadata()

        # Should not call count_rows_or_zero for collections with no embeddings
        assert not mock_vector_store.count_rows_or_zero.called

    @patch(
        "xagent.core.tools.core.RAG_tools.management.collection_manager.get_vector_index_store"
    )
    @patch("xagent.core.tools.core.RAG_tools.management.collections")
    @pytest.mark.asyncio
    async def test_rebuild_list_collections_fails(
        self, mock_collections_module, mock_get_vector_store
    ):
        """Test rebuild when list_collections fails."""
        from types import SimpleNamespace

        # Mock list_collections to return failure
        mock_result = SimpleNamespace(
            status="error", message="Failed to list collections"
        )

        async def mock_list_collections(**kwargs):
            return mock_result

        mock_collections_module.list_collections = mock_list_collections

        from xagent.core.tools.core.RAG_tools.management.collection_manager import (
            rebuild_collection_metadata,
        )

        # Should return early without error
        await rebuild_collection_metadata()

        # Vector store should not be accessed
        assert not mock_get_vector_store.called

    @patch(
        "xagent.core.tools.core.RAG_tools.management.collection_manager.get_vector_index_store"
    )
    @patch("xagent.core.tools.core.RAG_tools.management.collections")
    @pytest.mark.asyncio
    async def test_rebuild_empty_collections_list(
        self, mock_collections_module, mock_get_vector_store
    ):
        """Test rebuild when no collections exist."""
        from types import SimpleNamespace

        # Mock empty collections list
        mock_result = SimpleNamespace(status="success", collections=[])

        async def mock_list_collections(**kwargs):
            return mock_result

        mock_collections_module.list_collections = mock_list_collections

        from xagent.core.tools.core.RAG_tools.management.collection_manager import (
            rebuild_collection_metadata,
        )

        await rebuild_collection_metadata()

        # Vector store should not be accessed for empty list
        assert not mock_get_vector_store.called
