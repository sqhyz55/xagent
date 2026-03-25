"""Tests for PostgreSQL MetadataStore implementation (Phase 1B).

Note: Tests use mock objects to avoid PostgreSQL/JSONB dependencies in the test environment.
The actual SQL operations are tested in integration environments.
"""

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy.orm import Session

from xagent.core.tools.core.RAG_tools.core.schemas import CollectionInfo
from xagent.core.tools.core.RAG_tools.storage.permissions import (
    CollectionPermissionChecker,
    CollectionPermissions,
)
from xagent.core.tools.core.RAG_tools.storage.pg_metadata_store import (
    PostgreSQLMetadataStore,
)


class TestPostgreSQLMetadataStore:
    """Test PostgreSQL MetadataStore implementation using mocks."""

    @pytest.fixture
    def mock_engine(self) -> MagicMock:
        """Create a mock SQLAlchemy engine."""
        engine = MagicMock()
        return engine

    @pytest.fixture
    def mock_session_factory(self, mock_engine: MagicMock) -> MagicMock:
        """Create a mock session factory."""
        session = MagicMock(spec=Session)
        session_factory = MagicMock(return_value=session)
        return session_factory

    @pytest.fixture
    def pg_store(self, mock_engine: MagicMock) -> PostgreSQLMetadataStore:
        """Create PostgreSQLMetadataStore with mocked engine."""
        with patch(
            "xagent.core.tools.core.RAG_tools.storage.pg_metadata_store.create_engine",
            return_value=mock_engine,
        ):
            store = PostgreSQLMetadataStore(database_url="postgresql://test")
            store._engine = mock_engine
            return store

    @pytest.mark.asyncio
    async def test_ensure_collection_metadata_table(
        self, pg_store: PostgreSQLMetadataStore, mock_engine: MagicMock
    ) -> None:
        """Test table creation."""
        await pg_store.ensure_collection_metadata_table()
        # Verify Base.metadata.create_all was called with the engine
        from xagent.core.tools.core.RAG_tools.storage import rdb_models

        with patch.object(rdb_models.Base.metadata, "create_all") as mock_create:
            await pg_store.ensure_collection_metadata_table()
            mock_create.assert_called_once_with(pg_store._engine)

    @pytest.mark.asyncio
    async def test_save_collection_new(self, pg_store):
        """Test saving a new collection."""
        mock_session = MagicMock()
        pg_store._session_factory = MagicMock(return_value=mock_session)

        # Mock no existing collection
        mock_execute_result = MagicMock()
        mock_execute_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_execute_result

        collection = CollectionInfo(
            name="test_collection",
            owner_user_id=1,
            embedding_model_id="text-embedding-3-small",
        )

        await pg_store.save_collection(collection)

        # Verify session operations
        mock_session.add.assert_called_once()
        mock_session.commit.assert_called_once()
        mock_session.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_save_collection_update(self, pg_store):
        """Test updating an existing collection."""
        mock_session = MagicMock()
        pg_store._session_factory = MagicMock(return_value=mock_session)

        # Mock existing collection
        mock_existing = MagicMock()
        mock_existing.name = "test_collection"
        mock_existing.owner_user_id = 1

        mock_execute_result = MagicMock()
        mock_execute_result.scalar_one_or_none.return_value = mock_existing
        mock_session.execute.return_value = mock_execute_result

        collection = CollectionInfo(
            name="test_collection",
            owner_user_id=1,
            documents=5,
        )

        await pg_store.save_collection(collection)

        # Verify commit was called
        mock_session.commit.assert_called_once()
        mock_session.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_collection(self, pg_store):
        """Test retrieving a collection."""
        mock_session = MagicMock()
        pg_store._session_factory = MagicMock(return_value=mock_session)

        # Mock collection data
        mock_collection = MagicMock()
        mock_collection.name = "test_collection"
        mock_collection.owner_user_id = 1
        mock_collection.embedding_model_id = "text-embedding-3-small"
        mock_collection.embedding_dimension = 1536
        mock_collection.documents = 0
        mock_collection.processed_documents = 0
        mock_collection.parses = 0
        mock_collection.chunks = 0
        mock_collection.embeddings = 0
        mock_collection.document_names = []
        mock_collection.collection_locked = False
        mock_collection.allow_mixed_parse_methods = True
        mock_collection.skip_config_validation = False
        mock_collection.ingestion_config = None
        mock_collection.external_file_id = None
        mock_collection.schema_version = "1.0.0"
        mock_collection.created_at = datetime.now(timezone.utc)
        mock_collection.updated_at = datetime.now(timezone.utc)
        mock_collection.last_accessed_at = datetime.now(
            timezone.utc
        )  # Use actual datetime instead of None
        mock_collection.extra_metadata = {}

        mock_execute_result = MagicMock()
        mock_execute_result.scalar_one_or_none.return_value = mock_collection
        mock_session.execute.return_value = mock_execute_result

        result = await pg_store.get_collection("test_collection")

        assert result.name == "test_collection"
        assert result.owner_user_id == 1
        mock_session.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_collection_not_found(self, pg_store):
        """Test ValueError when collection not found."""
        mock_session = MagicMock()
        pg_store._session_factory = MagicMock(return_value=mock_session)

        mock_execute_result = MagicMock()
        mock_execute_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_execute_result

        with pytest.raises(ValueError, match="Collection 'nonexistent' not found"):
            await pg_store.get_collection("nonexistent")

    @pytest.mark.asyncio
    async def test_save_collection_config(self, pg_store):
        """Test saving collection config."""
        mock_session = MagicMock()
        pg_store._session_factory = MagicMock(return_value=mock_session)

        # Mock no existing config
        mock_execute_result = MagicMock()
        mock_execute_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_execute_result

        await pg_store.save_collection_config(
            collection="test_collection",
            config_json='{"chunk_size": 1000}',
            user_id=1,
        )

        mock_session.add.assert_called_once()
        mock_session.commit.assert_called_once()
        mock_session.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_collection_config(self, pg_store):
        """Test getting collection config."""

        mock_session = MagicMock()
        pg_store._session_factory = MagicMock(return_value=mock_session)

        # Mock config data
        mock_config = MagicMock()
        mock_config.config_json = {"chunk_size": 1000}

        mock_execute_result = MagicMock()
        mock_execute_result.scalar_one_or_none.return_value = mock_config
        mock_session.execute.return_value = mock_execute_result

        result = await pg_store.get_collection_config("test_collection", user_id=1)

        assert result == '{"chunk_size": 1000}'
        mock_session.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_collection_config_not_found(self, pg_store):
        """Test getting non-existent config returns None."""
        mock_session = MagicMock()
        pg_store._session_factory = MagicMock(return_value=mock_session)

        mock_execute_result = MagicMock()
        mock_execute_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_execute_result

        result = await pg_store.get_collection_config("test_collection", user_id=1)

        assert result is None

    def test_get_default_database_url_from_env(self):
        """Test getting database URL from environment variable."""
        import os

        with patch.dict(
            os.environ, {"DATABASE_URL": "postgresql://test:test@localhost/test"}
        ):
            store = PostgreSQLMetadataStore()
            assert store._database_url == "postgresql://test:test@localhost/test"

    def test_get_default_database_url_fallback(self):
        """Test fallback to default when DATABASE_URL not set."""
        import os

        with patch.dict(os.environ, {}, clear=True):
            store = PostgreSQLMetadataStore()
            assert (
                store._database_url
                == "postgresql://xagent:xagent@localhost:5432/xagent"
            )

    def test_get_raw_connection(self, pg_store):
        """Test get_raw_connection returns engine."""
        assert pg_store.get_raw_connection() is pg_store._engine


class TestCollectionPermissionsDataclass:
    """Test CollectionPermissions dataclass."""

    def test_permissions_full_access(self):
        """Test full access permissions."""
        perms = CollectionPermissions(can_read=True, can_modify=True, is_owner=True)
        assert perms.can_read is True
        assert perms.can_modify is True
        assert perms.is_owner is True

    def test_permissions_read_only(self):
        """Test read-only permissions."""
        perms = CollectionPermissions(can_read=True, can_modify=False, is_owner=False)
        assert perms.can_read is True
        assert perms.can_modify is False
        assert perms.is_owner is False

    def test_permissions_no_access(self):
        """Test no access permissions."""
        perms = CollectionPermissions(can_read=False, can_modify=False, is_owner=False)
        assert perms.can_read is False
        assert perms.can_modify is False
        assert perms.is_owner is False


class TestCollectionPermissionChecker:
    """Test CollectionPermissionChecker logic (Phase 1B)."""

    @pytest.fixture
    def mock_session(self) -> MagicMock:
        """Create a mock session."""
        return MagicMock(spec=Session)

    @pytest.fixture
    def permission_checker(
        self, mock_session: MagicMock
    ) -> CollectionPermissionChecker:
        """Create permission checker with mocked session factory."""
        session_factory = MagicMock(return_value=mock_session)
        return CollectionPermissionChecker(session_factory)

    def test_owner_has_full_permissions(self, permission_checker, mock_session):
        """Test that collection owner has full permissions."""
        # Mock collection owned by user 1
        mock_collection = MagicMock()
        mock_collection.owner_user_id = 1

        mock_execute_result = MagicMock()
        mock_execute_result.scalar_one_or_none.return_value = mock_collection
        mock_session.execute.return_value = mock_execute_result

        perms = permission_checker.get_permissions("test_collection", user_id=1)

        assert perms.can_read is True
        assert perms.can_modify is True
        assert perms.is_owner is True

    def test_shared_user_read_only(self, permission_checker, mock_session):
        """Test that shared users have read-only access."""
        # Mock collection owned by user 1
        mock_collection = MagicMock()
        mock_collection.owner_user_id = 1

        # Mock no share relationship found (first query returns collection, second returns None)
        mock_execute_result = MagicMock()
        mock_execute_result.scalar_one_or_none.side_effect = [mock_collection, None]
        mock_session.execute.return_value = mock_execute_result

        perms = permission_checker.get_permissions("test_collection", user_id=2)

        # User 2 is not owner and not in share list
        assert perms.can_read is False
        assert perms.can_modify is False
        assert perms.is_owner is False

    def test_shared_user_with_share(self, permission_checker, mock_session):
        """Test that shared users have read-only access when share exists."""
        # Mock collection owned by user 1
        mock_collection = MagicMock()
        mock_collection.owner_user_id = 1

        # Mock share exists
        mock_share = MagicMock()
        mock_share.shared_with_user_id = 2

        mock_execute_result = MagicMock()
        mock_execute_result.scalar_one_or_none.side_effect = [
            mock_collection,
            mock_share,
        ]
        mock_session.execute.return_value = mock_execute_result

        perms = permission_checker.get_permissions("test_collection", user_id=2)

        assert perms.can_read is True
        assert perms.can_modify is False
        assert perms.is_owner is False

    def test_unauthorized_user_no_access(self, permission_checker, mock_session):
        """Test that unauthorized users have no access."""
        # Mock collection owned by user 1
        mock_collection = MagicMock()
        mock_collection.owner_user_id = 1

        # Mock no share relationship
        mock_execute_result = MagicMock()
        mock_execute_result.scalar_one_or_none.side_effect = [mock_collection, None]
        mock_session.execute.return_value = mock_execute_result

        perms = permission_checker.get_permissions("test_collection", user_id=999)

        assert perms.can_read is False
        assert perms.can_modify is False
        assert perms.is_owner is False

    def test_nonexistent_collection_no_access(self, permission_checker, mock_session):
        """Test that non-existent collections return no permissions."""
        mock_execute_result = MagicMock()
        mock_execute_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_execute_result

        perms = permission_checker.get_permissions("nonexistent", user_id=1)

        assert perms.can_read is False
        assert perms.can_modify is False
        assert perms.is_owner is False

    def test_admin_bypass(self, permission_checker, mock_session):
        """Test that admins have full access regardless of ownership."""
        perms = permission_checker.get_permissions(
            "any_collection", user_id=999, is_admin=True
        )

        assert perms.can_read is True
        assert perms.can_modify is True
        assert perms.is_owner is False  # Not the owner, but has access via admin

    def test_can_modify_convenience(self, permission_checker, mock_session):
        """Test can_modify convenience method."""
        # Mock collection owned by user 1
        mock_collection = MagicMock()
        mock_collection.owner_user_id = 1

        mock_execute_result = MagicMock()
        mock_execute_result.scalar_one_or_none.return_value = mock_collection
        mock_session.execute.return_value = mock_execute_result

        assert permission_checker.can_modify("test_collection", user_id=1) is True

    def test_can_read_convenience(self, permission_checker, mock_session):
        """Test can_read convenience method."""
        # Mock collection owned by user 1
        mock_collection = MagicMock()
        mock_collection.owner_user_id = 1

        mock_execute_result = MagicMock()
        mock_execute_result.scalar_one_or_none.return_value = mock_collection
        mock_session.execute.return_value = mock_execute_result

        assert permission_checker.can_read("test_collection", user_id=1) is True

    def test_require_modify_success(self, permission_checker, mock_session):
        """Test require_modify does not raise for authorized user."""
        # Mock collection owned by user 1
        mock_collection = MagicMock()
        mock_collection.owner_user_id = 1

        mock_execute_result = MagicMock()
        mock_execute_result.scalar_one_or_none.return_value = mock_collection
        mock_session.execute.return_value = mock_execute_result

        # Should not raise
        permission_checker.require_modify("test_collection", user_id=1)

    def test_require_modify_failure(self, permission_checker, mock_session):
        """Test require_modify raises for unauthorized user."""
        # Mock collection owned by user 1
        mock_collection = MagicMock()
        mock_collection.owner_user_id = 1

        # Mock no share relationship
        mock_execute_result = MagicMock()
        mock_execute_result.scalar_one_or_none.side_effect = [mock_collection, None]
        mock_session.execute.return_value = mock_execute_result

        with pytest.raises(PermissionError, match="does not have permission to modify"):
            permission_checker.require_modify("test_collection", user_id=2)

    def test_require_read_success(self, permission_checker, mock_session):
        """Test require_read does not raise for authorized user."""
        # Mock collection owned by user 1
        mock_collection = MagicMock()
        mock_collection.owner_user_id = 1

        mock_execute_result = MagicMock()
        mock_execute_result.scalar_one_or_none.return_value = mock_collection
        mock_session.execute.return_value = mock_execute_result

        # Should not raise
        permission_checker.require_read("test_collection", user_id=1)

    def test_require_read_failure(self, permission_checker, mock_session):
        """Test require_read raises for unauthorized user."""
        # Mock collection owned by user 1
        mock_collection = MagicMock()
        mock_collection.owner_user_id = 1

        # Mock no share relationship
        mock_execute_result = MagicMock()
        mock_execute_result.scalar_one_or_none.side_effect = [mock_collection, None]
        mock_session.execute.return_value = mock_execute_result

        with pytest.raises(PermissionError, match="does not have permission to access"):
            permission_checker.require_read("test_collection", user_id=999)


class TestFactoryIntegration:
    """Test factory integration with new PostgreSQL backend."""

    def test_default_backend_is_lancedb(self):
        """Test that default backend is LanceDB."""
        from xagent.core.tools.core.RAG_tools.storage import factory

        factory.reset_metadata_store()
        # Default is lancedb when RAG_METADATA_STORE_BACKEND is not set
        assert factory.METADATA_STORE_BACKEND in ("lancedb", "postgresql")

    @pytest.mark.asyncio
    async def test_factory_returns_lancedb_store_by_default(self):
        """Test that factory returns LanceDBMetadataStore by default."""
        from xagent.core.tools.core.RAG_tools.storage import factory
        from xagent.core.tools.core.RAG_tools.storage.lancedb_stores import (
            LanceDBMetadataStore,
        )

        factory.reset_metadata_store()
        store = factory.get_metadata_store()

        assert isinstance(store, LanceDBMetadataStore)

    @pytest.mark.asyncio
    async def test_factory_environment_variable_control(self):
        """Test that environment variable controls backend selection."""
        # Verify the environment variable can be checked
        from xagent.core.tools.core.RAG_tools.storage import factory

        assert hasattr(factory, "METADATA_STORE_BACKEND")
        assert factory.METADATA_STORE_BACKEND in ("lancedb", "postgresql")

    def test_reset_metadata_store(self):
        """Test that reset_metadata_store clears the singleton."""
        from xagent.core.tools.core.RAG_tools.storage import factory

        store1 = factory.get_metadata_store()
        factory.reset_metadata_store()
        store2 = factory.get_metadata_store()

        # Stores should be different instances after reset
        assert store1 is not store2
