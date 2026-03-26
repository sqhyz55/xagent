"""Permission checking for KB collections (Phase 1B).

Simplified model:
- Owner: full control (upload, delete, process, read, search)
- Shared users: read-only (view, search)
- System admins: full control (bypasses collection checks)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable

from sqlalchemy import select
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


@dataclass
class CollectionPermissions:
    """Collection access permissions."""

    can_read: bool
    can_modify: bool  # upload, delete, process
    is_owner: bool


class CollectionPermissionChecker:
    """Check and enforce collection permissions (Phase 1B)."""

    def __init__(self, session_factory: Callable[[], Session]) -> None:
        """Initialize permission checker.

        Args:
            session_factory: SQLAlchemy session factory (e.g., sessionmaker or async_sessionmaker).
                             Should return a Session when called.
        """
        self._session_factory = session_factory

    def get_permissions(
        self,
        collection_name: str,
        user_id: int,
        is_admin: bool = False,
    ) -> CollectionPermissions:
        """Get user permissions for a collection.

        Args:
            collection_name: Target collection name.
            user_id: User ID to check.
            is_admin: Whether user is a system admin (bypasses collection checks).

        Returns:
            CollectionPermissions object.
        """
        # System admins have full access (used for operations/debug)
        if is_admin:
            return CollectionPermissions(
                can_read=True,
                can_modify=True,
                is_owner=False,
            )

        from .rdb_models import KBCollectionMetadata, KBCollectionShare

        session = self._session_factory()
        try:
            # Check if user is the owner
            stmt = select(KBCollectionMetadata).where(
                KBCollectionMetadata.name == collection_name
            )
            collection = session.execute(stmt).scalar_one_or_none()

            if collection is None:
                # Collection doesn't exist - treat as no access
                return CollectionPermissions(
                    can_read=False, can_modify=False, is_owner=False
                )

            if collection.owner_user_id == user_id:
                return CollectionPermissions(
                    can_read=True,
                    can_modify=True,
                    is_owner=True,
                )

            # Check if user has read-only share access
            share_stmt = select(KBCollectionShare).where(
                KBCollectionShare.collection == collection_name,
                KBCollectionShare.shared_with_user_id == user_id,
            )
            share = session.execute(share_stmt).scalar_one_or_none()

            if share is not None:
                return CollectionPermissions(
                    can_read=True,
                    can_modify=False,  # Shared users are read-only
                    is_owner=False,
                )

            # No access
            return CollectionPermissions(
                can_read=False, can_modify=False, is_owner=False
            )

        finally:
            session.close()

    def can_modify(
        self, collection_name: str, user_id: int, is_admin: bool = False
    ) -> bool:
        """Check if user can modify collection (upload, delete, process).

        Args:
            collection_name: Target collection name.
            user_id: User ID to check.
            is_admin: Whether user is a system admin.

        Returns:
            True if user can modify the collection.
        """
        perms = self.get_permissions(collection_name, user_id, is_admin)
        return perms.can_modify

    def can_read(
        self, collection_name: str, user_id: int, is_admin: bool = False
    ) -> bool:
        """Check if user can read/search collection.

        Args:
            collection_name: Target collection name.
            user_id: User ID to check.
            is_admin: Whether user is a system admin.

        Returns:
            True if user can read the collection.
        """
        perms = self.get_permissions(collection_name, user_id, is_admin)
        return perms.can_read

    def require_modify(
        self, collection_name: str, user_id: int, is_admin: bool = False
    ) -> None:
        """Raise exception if user cannot modify collection.

        Args:
            collection_name: Target collection name.
            user_id: User ID to check.
            is_admin: Whether user is a system admin.

        Raises:
            PermissionError: If user cannot modify the collection.
        """
        if not self.can_modify(collection_name, user_id, is_admin):
            raise PermissionError(
                f"User {user_id} does not have permission to modify collection '{collection_name}'. "
                "Only the collection owner can upload, delete, or process documents."
            )

    def require_read(
        self, collection_name: str, user_id: int, is_admin: bool = False
    ) -> None:
        """Raise exception if user cannot read collection.

        Args:
            collection_name: Target collection name.
            user_id: User ID to check.
            is_admin: Whether user is a system admin.

        Raises:
            PermissionError: If user cannot read the collection.
        """
        if not self.can_read(collection_name, user_id, is_admin):
            raise PermissionError(
                f"User {user_id} does not have permission to access collection '{collection_name}'. "
                "Only the collection owner and shared users can read the collection."
            )


class AsyncCollectionPermissionChecker:
    """Async version of permission checker for PostgreSQL (Phase 1B).

    Uses AsyncSession for non-blocking database operations.
    """

    def __init__(self, session_factory: Any) -> None:
        """Initialize async permission checker.

        Args:
            session_factory: SQLAlchemy async session factory (async_sessionmaker).
                             Should return an AsyncSession when called.
        """
        self._session_factory = session_factory

    async def get_permissions(
        self,
        collection_name: str,
        user_id: int,
        is_admin: bool = False,
    ) -> CollectionPermissions:
        """Get user permissions for a collection.

        Args:
            collection_name: Target collection name.
            user_id: User ID to check.
            is_admin: Whether user is a system admin (bypasses collection checks).

        Returns:
            CollectionPermissions object.
        """
        # System admins have full access (used for operations/debug)
        if is_admin:
            return CollectionPermissions(
                can_read=True,
                can_modify=True,
                is_owner=False,
            )

        from .rdb_models import KBCollectionMetadata, KBCollectionShare

        async with self._session_factory() as session:
            # Check if user is the owner
            stmt = select(KBCollectionMetadata).where(
                KBCollectionMetadata.name == collection_name
            )
            result = await session.execute(stmt)
            collection = result.scalar_one_or_none()

            if collection is None:
                # Collection doesn't exist - treat as no access
                return CollectionPermissions(
                    can_read=False, can_modify=False, is_owner=False
                )

            if collection.owner_user_id == user_id:
                return CollectionPermissions(
                    can_read=True,
                    can_modify=True,
                    is_owner=True,
                )

            # Check if user has read-only share access
            share_stmt = select(KBCollectionShare).where(
                KBCollectionShare.collection == collection_name,
                KBCollectionShare.shared_with_user_id == user_id,
            )
            share_result = await session.execute(share_stmt)
            share = share_result.scalar_one_or_none()

            if share is not None:
                return CollectionPermissions(
                    can_read=True,
                    can_modify=False,  # Shared users are read-only
                    is_owner=False,
                )

            # No access
            return CollectionPermissions(
                can_read=False, can_modify=False, is_owner=False
            )

    async def can_modify(
        self, collection_name: str, user_id: int, is_admin: bool = False
    ) -> bool:
        """Check if user can modify collection (upload, delete, process).

        Args:
            collection_name: Target collection name.
            user_id: User ID to check.
            is_admin: Whether user is a system admin.

        Returns:
            True if user can modify the collection.
        """
        perms = await self.get_permissions(collection_name, user_id, is_admin)
        return perms.can_modify

    async def can_read(
        self, collection_name: str, user_id: int, is_admin: bool = False
    ) -> bool:
        """Check if user can read/search collection.

        Args:
            collection_name: Target collection name.
            user_id: User ID to check.
            is_admin: Whether user is a system admin.

        Returns:
            True if user can read the collection.
        """
        perms = await self.get_permissions(collection_name, user_id, is_admin)
        return perms.can_read

    async def require_modify(
        self, collection_name: str, user_id: int, is_admin: bool = False
    ) -> None:
        """Raise exception if user cannot modify collection.

        Args:
            collection_name: Target collection name.
            user_id: User ID to check.
            is_admin: Whether user is a system admin.

        Raises:
            PermissionError: If user cannot modify the collection.
        """
        if not await self.can_modify(collection_name, user_id, is_admin):
            raise PermissionError(
                f"User {user_id} does not have permission to modify collection '{collection_name}'. "
                "Only the collection owner can upload, delete, or process documents."
            )

    async def require_read(
        self, collection_name: str, user_id: int, is_admin: bool = False
    ) -> None:
        """Raise exception if user cannot read collection.

        Args:
            collection_name: Target collection name.
            user_id: User ID to check.
            is_admin: Whether user is a system admin.

        Raises:
            PermissionError: If user cannot read the collection.
        """
        if not await self.can_read(collection_name, user_id, is_admin):
            raise PermissionError(
                f"User {user_id} does not have permission to access collection '{collection_name}'. "
                "Only the collection owner and shared users can read the collection."
            )
