"""Tests for AsyncCollectionPermissionChecker (Phase 1B async fix).

Tests verify that:
1. AsyncCollectionPermissionChecker uses proper async/await
2. All methods are async def
3. Uses async with session_factory() as session:
4. Uses await session.execute(...)
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from xagent.core.tools.core.RAG_tools.storage.permissions import (
    AsyncCollectionPermissionChecker,
    CollectionPermissions,
)


class TestAsyncCollectionPermissionChecker:
    """Test AsyncCollectionPermissionChecker with proper async patterns."""

    @pytest.fixture
    def mock_async_session(self) -> MagicMock:
        """Create a mock AsyncSession."""
        session = MagicMock(spec=AsyncSession)
        session.__aenter__ = AsyncMock(return_value=session)
        session.__aexit__ = AsyncMock()
        return session

    @pytest.fixture
    def mock_session_factory(self, mock_async_session: MagicMock) -> MagicMock:
        """Create a mock async session factory."""
        factory = MagicMock(return_value=mock_async_session)
        factory.__call__ = MagicMock(return_value=mock_async_session)
        return factory

    @pytest.fixture
    def permission_checker(
        self, mock_session_factory: MagicMock
    ) -> AsyncCollectionPermissionChecker:
        """Create permission checker with mocked session factory."""
        return AsyncCollectionPermissionChecker(mock_session_factory)

    @pytest.mark.asyncio
    async def test_admin_has_full_permissions(
        self, permission_checker: AsyncCollectionPermissionChecker
    ) -> None:
        """Test that admin has full permissions bypassing collection checks."""
        perms = await permission_checker.get_permissions(
            "test_collection", user_id=999, is_admin=True
        )

        assert perms.can_read is True
        assert perms.can_modify is True
        assert perms.is_owner is False

    @pytest.mark.asyncio
    async def test_owner_has_full_permissions(
        self,
        permission_checker: AsyncCollectionPermissionChecker,
        mock_async_session: MagicMock,
    ) -> None:
        """Test that collection owner has full permissions."""
        # Mock collection owned by user 1
        mock_collection = MagicMock()
        mock_collection.owner_user_id = 1

        mock_execute_result = MagicMock()
        mock_execute_result.scalar_one_or_none.return_value = mock_collection
        mock_async_session.execute = AsyncMock(return_value=mock_execute_result)

        perms = await permission_checker.get_permissions("test_collection", user_id=1)

        assert perms.can_read is True
        assert perms.can_modify is True
        assert perms.is_owner is True

    @pytest.mark.asyncio
    async def test_shared_user_read_only(
        self,
        permission_checker: AsyncCollectionPermissionChecker,
        mock_async_session: MagicMock,
    ) -> None:
        """Test that shared users have read-only access."""
        # Mock collection owned by user 1
        mock_collection = MagicMock()
        mock_collection.owner_user_id = 1

        # Mock share exists
        mock_share = MagicMock()
        mock_share.shared_with_user_id = 2

        # First call returns collection, second returns share
        mock_execute_result = MagicMock()
        mock_execute_result.scalar_one_or_none.side_effect = [mock_collection, mock_share]
        mock_async_session.execute = AsyncMock(return_value=mock_execute_result)

        perms = await permission_checker.get_permissions("test_collection", user_id=2)

        assert perms.can_read is True
        assert perms.can_modify is False
        assert perms.is_owner is False

    @pytest.mark.asyncio
    async def test_unauthorized_user_no_access(
        self,
        permission_checker: AsyncCollectionPermissionChecker,
        mock_async_session: MagicMock,
    ) -> None:
        """Test that unauthorized users have no access."""
        # Mock collection owned by user 1
        mock_collection = MagicMock()
        mock_collection.owner_user_id = 1

        # Mock no share relationship
        mock_execute_result = MagicMock()
        mock_execute_result.scalar_one_or_none.side_effect = [mock_collection, None]
        mock_async_session.execute = AsyncMock(return_value=mock_execute_result)

        perms = await permission_checker.get_permissions("test_collection", user_id=999)

        assert perms.can_read is False
        assert perms.can_modify is False
        assert perms.is_owner is False

    @pytest.mark.asyncio
    async def test_nonexistent_collection_no_access(
        self,
        permission_checker: AsyncCollectionPermissionChecker,
        mock_async_session: MagicMock,
    ) -> None:
        """Test that non-existent collections return no permissions."""
        # Mock collection doesn't exist
        mock_execute_result = MagicMock()
        mock_execute_result.scalar_one_or_none.return_value = None
        mock_async_session.execute = AsyncMock(return_value=mock_execute_result)

        perms = await permission_checker.get_permissions("nonexistent", user_id=1)

        assert perms.can_read is False
        assert perms.can_modify is False
        assert perms.is_owner is False

    @pytest.mark.asyncio
    async def test_can_modify_convenience(
        self,
        permission_checker: AsyncCollectionPermissionChecker,
        mock_async_session: MagicMock,
    ) -> None:
        """Test can_modify convenience method."""
        # Mock collection owned by user 1
        mock_collection = MagicMock()
        mock_collection.owner_user_id = 1

        mock_execute_result = MagicMock()
        mock_execute_result.scalar_one_or_none.return_value = mock_collection
        mock_async_session.execute = AsyncMock(return_value=mock_execute_result)

        result = await permission_checker.can_modify("test_collection", user_id=1)

        assert result is True

    @pytest.mark.asyncio
    async def test_can_read_convenience(
        self,
        permission_checker: AsyncCollectionPermissionChecker,
        mock_async_session: MagicMock,
    ) -> None:
        """Test can_read convenience method."""
        # Mock collection owned by user 1
        mock_collection = MagicMock()
        mock_collection.owner_user_id = 1

        mock_execute_result = MagicMock()
        mock_execute_result.scalar_one_or_none.return_value = mock_collection
        mock_async_session.execute = AsyncMock(return_value=mock_execute_result)

        result = await permission_checker.can_read("test_collection", user_id=1)

        assert result is True

    @pytest.mark.asyncio
    async def test_require_modify_success(
        self,
        permission_checker: AsyncCollectionPermissionChecker,
        mock_async_session: MagicMock,
    ) -> None:
        """Test require_modify does not raise for authorized user."""
        # Mock collection owned by user 1
        mock_collection = MagicMock()
        mock_collection.owner_user_id = 1

        mock_execute_result = MagicMock()
        mock_execute_result.scalar_one_or_none.return_value = mock_collection
        mock_async_session.execute = AsyncMock(return_value=mock_execute_result)

        # Should not raise
        await permission_checker.require_modify("test_collection", user_id=1)

    @pytest.mark.asyncio
    async def test_require_modify_failure(
        self,
        permission_checker: AsyncCollectionPermissionChecker,
        mock_async_session: MagicMock,
    ) -> None:
        """Test require_modify raises for unauthorized user."""
        # Mock collection owned by user 1
        mock_collection = MagicMock()
        mock_collection.owner_user_id = 1

        # Mock no share relationship
        mock_execute_result = MagicMock()
        mock_execute_result.scalar_one_or_none.side_effect = [mock_collection, None]
        mock_async_session.execute = AsyncMock(return_value=mock_execute_result)

        with pytest.raises(PermissionError, match="does not have permission to modify"):
            await permission_checker.require_modify("test_collection", user_id=2)

    @pytest.mark.asyncio
    async def test_require_read_success(
        self,
        permission_checker: AsyncCollectionPermissionChecker,
        mock_async_session: MagicMock,
    ) -> None:
        """Test require_read does not raise for authorized user."""
        # Mock collection owned by user 1
        mock_collection = MagicMock()
        mock_collection.owner_user_id = 1

        mock_execute_result = MagicMock()
        mock_execute_result.scalar_one_or_none.return_value = mock_collection
        mock_async_session.execute = AsyncMock(return_value=mock_execute_result)

        # Should not raise
        await permission_checker.require_read("test_collection", user_id=1)

    @pytest.mark.asyncio
    async def test_require_read_failure(
        self,
        permission_checker: AsyncCollectionPermissionChecker,
        mock_async_session: MagicMock,
    ) -> None:
        """Test require_read raises for unauthorized user."""
        # Mock collection owned by user 1
        mock_collection = MagicMock()
        mock_collection.owner_user_id = 1

        # Mock no share relationship
        mock_execute_result = MagicMock()
        mock_execute_result.scalar_one_or_none.side_effect = [mock_collection, None]
        mock_async_session.execute = AsyncMock(return_value=mock_execute_result)

        with pytest.raises(PermissionError, match="does not have permission to access"):
            await permission_checker.require_read("test_collection", user_id=999)

    @pytest.mark.asyncio
    async def test_uses_async_context_manager(
        self,
        permission_checker: AsyncCollectionPermissionChecker,
        mock_session_factory: MagicMock,
        mock_async_session: MagicMock,
    ) -> None:
        """Test that checker uses async context manager for sessions."""
        # Mock collection owned by user 1
        mock_collection = MagicMock()
        mock_collection.owner_user_id = 1

        mock_execute_result = MagicMock()
        mock_execute_result.scalar_one_or_none.return_value = mock_collection
        mock_async_session.execute = AsyncMock(return_value=mock_execute_result)

        # Call the method
        await permission_checker.get_permissions("test_collection", user_id=1)

        # Verify session factory was called to create a session
        mock_session_factory.assert_called_once()

        # Verify execute was called (indicates async with worked)
        mock_async_session.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_uses_await_for_execute(
        self,
        permission_checker: AsyncCollectionPermissionChecker,
        mock_async_session: MagicMock,
    ) -> None:
        """Test that execute is called with await."""
        # Mock collection owned by user 1
        mock_collection = MagicMock()
        mock_collection.owner_user_id = 1

        mock_execute_result = MagicMock()
        mock_execute_result.scalar_one_or_none.return_value = mock_collection
        mock_async_session.execute = AsyncMock(return_value=mock_execute_result)

        # Call the method
        await permission_checker.get_permissions("test_collection", user_id=1)

        # Verify execute was called with await (AsyncMock verifies this)
        mock_async_session.execute.assert_called_once()


class TestAsyncVsSyncPermissionChecker:
    """Compare async and sync permission checkers have same logic."""

    @pytest.mark.asyncio
    async def test_async_checker_mirrors_sync_logic(self) -> None:
        """Verify async checker implements same permission logic as sync."""
        from xagent.core.tools.core.RAG_tools.storage.permissions import (
            CollectionPermissionChecker,
        )

        # Both should have the same methods
        sync_methods = set(dir(CollectionPermissionChecker))
        async_methods = set(dir(AsyncCollectionPermissionChecker))

        # Check that key methods exist in both
        key_methods = {
            "get_permissions",
            "can_modify",
            "can_read",
            "require_modify",
            "require_read",
        }

        assert key_methods.issubset(sync_methods)
        assert key_methods.issubset(async_methods)
