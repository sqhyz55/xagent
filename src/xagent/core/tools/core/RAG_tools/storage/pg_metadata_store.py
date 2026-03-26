"""PostgreSQL implementation for MetadataStore contract (Phase 1B - Fixed).

Provides RDB-backed control-plane metadata storage for Phase 1B with true async support.

Changes:
- Migrated to SQLAlchemy async (create_async_engine + AsyncSession)
- All DB operations now truly non-blocking
- Fixed get_raw_connection contract violation
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from ..core.schemas import CollectionInfo
from .contracts import MetadataStore
from .rdb_models import Base, KBCollectionConfig, KBCollectionMetadata

logger = logging.getLogger(__name__)


class PostgreSQLMetadataStore(MetadataStore):
    """PostgreSQL implementation for control-plane metadata operations.

    Uses true async SQLAlchemy for non-blocking database operations.

    Usage:
        store = PostgreSQLMetadataStore()
        await store.ensure_collection_metadata_table()
        await store.save_collection(collection_info)
        collection = await store.get_collection("my_collection")
    """

    def __init__(self, database_url: str | None = None) -> None:
        """Initialize PostgreSQL metadata store.

        Args:
            database_url: SQLAlchemy database URL. If None, uses settings or environment.
        """
        self._database_url = database_url or self._get_default_database_url()
        # Use async engine with proper asyncpg driver
        self._engine = create_async_engine(
            self._database_url,
            pool_pre_ping=True,
            echo=False,
        )
        self._session_factory = async_sessionmaker(
            bind=self._engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )

    def _get_default_database_url(self) -> str:
        """Get default database URL from environment.

        Tries in order:
        1. DATABASE_URL environment variable
        2. Default localhost PostgreSQL

        Returns:
            Database URL string.
        """
        import os

        url = os.environ.get(
            "DATABASE_URL", "postgresql://xagent:xagent@localhost:5432/xagent"
        )
        # Ensure async driver is used
        if url.startswith("postgresql://"):
            url = url.replace("postgresql://", "postgresql+asyncpg://", 1)
        return url

    async def _get_session(self) -> AsyncSession:
        """Get a new database session.

        Returns:
            SQLAlchemy AsyncSession object.
        """
        return self._session_factory()

    async def get_collection(self, collection_name: str) -> CollectionInfo:
        """Read collection metadata from PostgreSQL.

        Args:
            collection_name: Target collection name.

        Returns:
            Collection metadata.

        Raises:
            ValueError: If collection is not found.
        """
        async with self._session_factory() as session:
            stmt = select(KBCollectionMetadata).where(
                KBCollectionMetadata.name == collection_name
            )
            result = await session.execute(stmt)
            orm_obj = result.scalar_one_or_none()
            if orm_obj is None:
                raise ValueError(
                    f"Collection '{collection_name}' not found in PostgreSQL"
                )
            return self._orm_to_collection_info(orm_obj)

    async def save_collection(self, collection: CollectionInfo) -> None:
        """Create or update collection metadata in PostgreSQL.

        Args:
            collection: Collection metadata to save.
        """
        async with self._session_factory() as session:
            stmt = select(KBCollectionMetadata).where(
                KBCollectionMetadata.name == collection.name
            )
            result = await session.execute(stmt)
            existing = result.scalar_one_or_none()

            if existing:
                # Update existing record
                data = collection.to_storage()
                for key, value in data.items():
                    if hasattr(existing, key):
                        setattr(existing, key, value)
                existing.updated_at = datetime.now(timezone.utc)
            else:
                # Insert new record
                orm_obj = self._collection_info_to_orm(collection)
                session.add(orm_obj)

            await session.commit()

    async def ensure_collection_metadata_table(self) -> None:
        """Create metadata tables if they don't exist.

        This creates all KB metadata tables including:
        - kb_collection_metadata
        - kb_collection_shares
        - kb_document_staging
        - kb_collection_config
        """
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("PostgreSQL KB metadata tables ensured")

    async def save_collection_config(
        self,
        collection: str,
        config_json: str,
        user_id: int,
    ) -> None:
        """Save collection ingestion configuration to PostgreSQL.

        Args:
            collection: Collection name.
            config_json: JSON string of IngestionConfig.
            user_id: User ID for multi-tenancy.
        """
        import json

        async with self._session_factory() as session:
            # Delete existing config for this collection+user
            stmt = select(KBCollectionConfig).where(
                KBCollectionConfig.collection == collection,
                KBCollectionConfig.user_id == user_id,
            )
            result = await session.execute(stmt)
            existing = result.scalar_one_or_none()
            if existing:
                await session.delete(existing)

            # Insert new config
            new_config = KBCollectionConfig(
                collection=collection,
                user_id=user_id,
                config_json=json.loads(config_json),
            )
            session.add(new_config)
            await session.commit()

            logger.debug(
                "Saved config for collection '%s', user %s", collection, user_id
            )

    async def get_collection_config(
        self,
        collection: str,
        user_id: int,
    ) -> str | None:
        """Get collection ingestion configuration from PostgreSQL.

        Args:
            collection: Collection name.
            user_id: User ID for multi-tenancy.

        Returns:
            Config JSON string if found, None otherwise.
        """
        import json

        async with self._session_factory() as session:
            stmt = select(KBCollectionConfig).where(
                KBCollectionConfig.collection == collection,
                KBCollectionConfig.user_id == user_id,
            )
            result = await session.execute(stmt)
            row = result.scalar_one_or_none()
            if row is None:
                return None
            return json.dumps(row.config_json)

    def get_raw_connection(self) -> Any:
        """Return raw engine for legacy compatibility paths.

        Note: This returns SQLAlchemy async Engine, not a synchronous connection.
        The contract is intentionally loose here since different backends
        have different connection types.

        For PostgreSQL async operations, use the async methods directly.
        For legacy sync code that needs a connection, this provides access
        but callers must handle the async nature appropriately.
        """
        return self._engine

    # Private helper methods

    def _orm_to_collection_info(self, orm: KBCollectionMetadata) -> CollectionInfo:
        """Convert ORM object to CollectionInfo.

        Args:
            orm: KBCollectionMetadata ORM instance.

        Returns:
            CollectionInfo instance.
        """
        # Handle nullable last_accessed_at - use created_at if None
        last_accessed = orm.last_accessed_at if orm.last_accessed_at else orm.created_at

        data = {
            "name": orm.name,
            "schema_version": orm.schema_version,
            "embedding_model_id": orm.embedding_model_id,
            "embedding_dimension": orm.embedding_dimension,
            "documents": orm.documents,
            "processed_documents": orm.processed_documents,
            "parses": orm.parses,
            "chunks": orm.chunks,
            "embeddings": orm.embeddings,
            "document_names": orm.document_names,
            "collection_locked": orm.collection_locked,
            "allow_mixed_parse_methods": orm.allow_mixed_parse_methods,
            "skip_config_validation": orm.skip_config_validation,
            "ingestion_config": orm.ingestion_config,
            "external_file_id": orm.external_file_id,
            "owner_user_id": orm.owner_user_id,
            "created_at": orm.created_at,
            "updated_at": orm.updated_at,
            "last_accessed_at": last_accessed,
            "extra_metadata": orm.extra_metadata,
        }
        return CollectionInfo.from_storage(data)

    def _collection_info_to_orm(self, info: CollectionInfo) -> KBCollectionMetadata:
        """Convert CollectionInfo to ORM object.

        Args:
            info: CollectionInfo instance.

        Returns:
            KBCollectionMetadata ORM instance.
        """
        data = info.to_storage()
        return KBCollectionMetadata(
            name=data.get("name", ""),
            schema_version=data.get("schema_version", "1.0.0"),
            embedding_model_id=data.get("embedding_model_id"),
            embedding_dimension=data.get("embedding_dimension"),
            documents=data.get("documents", 0),
            processed_documents=data.get("processed_documents", 0),
            parses=data.get("parses", 0),
            chunks=data.get("chunks", 0),
            embeddings=data.get("embeddings", 0),
            document_names=data.get("document_names", []),
            collection_locked=data.get("collection_locked", False),
            allow_mixed_parse_methods=data.get("allow_mixed_parse_methods", True),
            skip_config_validation=data.get("skip_config_validation", False),
            ingestion_config=data.get("ingestion_config"),
            external_file_id=data.get("external_file_id"),
            owner_user_id=data.get("owner_user_id", 0),
            extra_metadata=data.get("extra_metadata", {}),
        )
