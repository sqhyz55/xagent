"""Dual-write coordinator for LanceDB to PostgreSQL migration (Phase 1B).

Coordinates writes between LanceDB (legacy) and PostgreSQL (new) during migration.
Provides backfill, reconcile, and rollback capabilities.

Migration phases:
1. Dual-write mode: Write to both backends, read from LanceDB
2. Reconcile mode: Verify data consistency between backends
3. Cutover mode: Write to PostgreSQL, read from PostgreSQL
4. Rollback: Revert to LanceDB if issues found

Environment variables:
- RAG_DUAL_WRITE_ENABLED: Enable dual-write mode (default: false)
- RAG_READ_BACKEND: 'lancedb' or 'postgresql' (default: lancedb)
- RAG_WRITE_BACKEND: 'lancedb', 'postgresql', or 'both' (default: lancedb)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from ..core.schemas import CollectionInfo
from .contracts import KBWriteCoordinator, MetadataStore, VectorIndexStore
from .lancedb_stores import LanceDBMetadataStore, LanceDBVectorIndexStore
from .pg_metadata_store import PostgreSQLMetadataStore

logger = logging.getLogger(__name__)


class MetadataBackend(str, Enum):
    """Metadata storage backend types."""

    LANCEDB = "lancedb"
    POSTGRESQL = "postgresql"


@dataclass
class DualWriteStats:
    """Statistics for dual-write operations."""

    writes_to_primary: int = 0
    writes_to_secondary: int = 0
    write_failures: int = 0
    last_write_time: Optional[datetime] = None
    reconcile_checks: int = 0
    reconcile_mismatches: int = 0


@dataclass
class ReconcileResult:
    """Result of a reconcile operation."""

    collection_name: str
    primary_backend: MetadataBackend
    secondary_backend: MetadataBackend
    records_checked: int
    mismatches: List[Dict[str, Any]] = field(default_factory=list)
    is_consistent: bool = True
    checked_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class DualWriteCoordinator(KBWriteCoordinator):
    """Coordinator for dual-write operations during LanceDB to PostgreSQL migration.

    Usage:
        coordinator = DualWriteCoordinator(
            primary_backend='lancedb',  # Legacy backend
            secondary_backend='postgresql',  # New backend
            write_mode='both',  # Write to both during migration
        )

        # Writes go to both backends
        await coordinator.metadata_store().save_collection(collection)

        # Verify data consistency
        result = await coordinator.reconcile_collection('my_collection')
    """

    def __init__(
        self,
        read_backend: MetadataBackend = MetadataBackend.LANCEDB,
        write_mode: Literal["lancedb", "postgresql", "both"] = "lancedb",
        metadata_store_pg: Optional[PostgreSQLMetadataStore] = None,
        metadata_store_lancedb: Optional[LanceDBMetadataStore] = None,
        vector_index: Optional[VectorIndexStore] = None,
    ) -> None:
        """Initialize dual-write coordinator.

        Args:
            read_backend: Which backend to read from (default: LanceDB).
            write_mode: Where to write - 'lancedb', 'postgresql', or 'both'.
            metadata_store_pg: PostgreSQL metadata store instance.
            metadata_store_lancedb: LanceDB metadata store instance.
            vector_index: Vector index store (always LanceDB in Phase 1B).
        """
        if write_mode not in ("lancedb", "postgresql", "both"):
            raise ValueError(
                f"Invalid write_mode: {write_mode}. Must be 'lancedb', 'postgresql', or 'both'"
            )
        if not isinstance(read_backend, MetadataBackend):
            raise ValueError(
                f"Invalid read_backend: {read_backend}. Must be MetadataBackend enum"
            )

        self._read_backend = read_backend
        self._write_mode = write_mode
        self._stats = DualWriteStats()

        # Initialize stores
        self._metadata_lancedb = metadata_store_lancedb or LanceDBMetadataStore()
        self._metadata_postgres = metadata_store_pg or PostgreSQLMetadataStore()
        self._vector_index = vector_index or LanceDBVectorIndexStore()

        # Create dual-write metadata store based on configuration
        self._metadata = self._create_metadata_store()

        logger.info(
            "DualWriteCoordinator initialized: write_mode=%s, read_backend=%s",
            write_mode,
            read_backend.value,
        )

    def _create_metadata_store(self) -> MetadataStore:
        """Create metadata store based on write and read mode."""
        if self._write_mode == "both":
            return DualWriteMetadataStore(
                lancedb_store=self._metadata_lancedb,
                pg_store=self._metadata_postgres,
                stats=self._stats,
                read_backend=self._read_backend,
            )
        elif self._write_mode == "postgresql":
            return self._metadata_postgres
        else:
            return self._metadata_lancedb

    def metadata_store(self) -> MetadataStore:
        """Return configured metadata store."""
        return self._metadata

    def vector_index_store(self) -> VectorIndexStore:
        """Return vector index store (always LanceDB in Phase 1B)."""
        return self._vector_index

    def get_stats(self) -> DualWriteStats:
        """Get dual-write statistics."""
        return self._stats

    async def reconcile_collection(self, collection_name: str) -> ReconcileResult:
        """Reconcile collection data between backends.

        Compares collection metadata between primary and secondary backends.
        Logs any mismatches found.

        Args:
            collection_name: Collection name to reconcile.

        Returns:
            ReconcileResult with details of any mismatches.
        """
        self._stats.reconcile_checks += 1
        mismatches = []

        try:
            # Get collection from both backends
            primary_data = await self._metadata_lancedb.get_collection(collection_name)
            secondary_data = await self._metadata_postgres.get_collection(
                collection_name
            )

            # Compare key fields
            fields_to_check = [
                "name",
                "owner_user_id",
                "embedding_model_id",
                "embedding_dimension",
                "documents",
                "processed_documents",
                "parses",
                "chunks",
                "embeddings",
            ]

            for field in fields_to_check:
                primary_val = getattr(primary_data, field, None)
                secondary_val = getattr(secondary_data, field, None)

                if primary_val != secondary_val:
                    mismatches.append(
                        {
                            "field": field,
                            "primary_value": str(primary_val),
                            "secondary_value": str(secondary_val),
                        }
                    )
                    self._stats.reconcile_mismatches += 1

            result = ReconcileResult(
                collection_name=collection_name,
                primary_backend=MetadataBackend.LANCEDB,
                secondary_backend=MetadataBackend.POSTGRESQL,
                records_checked=1,
                mismatches=mismatches,
                is_consistent=len(mismatches) == 0,
            )

            if mismatches:
                logger.warning(
                    "Reconcile found %d mismatches for collection '%s': %s",
                    len(mismatches),
                    collection_name,
                    mismatches,
                )
            else:
                logger.info("Reconcile passed for collection '%s'", collection_name)

            return result

        except Exception as e:
            logger.error("Failed to reconcile collection '%s': %s", collection_name, e)
            return ReconcileResult(
                collection_name=collection_name,
                primary_backend=MetadataBackend.LANCEDB,
                secondary_backend=MetadataBackend.POSTGRESQL,
                records_checked=0,
                is_consistent=False,
            )

    async def backfill_collection(self, collection_name: str) -> Dict[str, Any]:
        """Backfill collection data from LanceDB to PostgreSQL.

        Reads collection metadata from LanceDB and writes to PostgreSQL.
        Useful for initial data migration.

        Args:
            collection_name: Collection name to backfill.

        Returns:
            Dict with backfill status and details.
        """
        logger.info("Starting backfill for collection '%s'", collection_name)

        try:
            # Read from LanceDB
            lancedb_data = await self._metadata_lancedb.get_collection(collection_name)

            # Write to PostgreSQL
            await self._metadata_postgres.save_collection(lancedb_data)

            logger.info("Successfully backfilled collection '%s'", collection_name)

            return {
                "status": "success",
                "collection": collection_name,
                "message": f"Collection '{collection_name}' backfilled from LanceDB to PostgreSQL",
            }

        except Exception as e:
            logger.error("Failed to backfill collection '%s': %s", collection_name, e)
            return {
                "status": "error",
                "collection": collection_name,
                "error": str(e),
            }

    async def backfill_all_collections(self) -> Dict[str, Any]:
        """Backfill all collections from LanceDB to PostgreSQL.

        Returns:
            Dict with backfill summary including success/failed counts.
        """
        from ..core.schemas import ListCollectionsResult
        from ..management.collections import list_collections

        logger.info("Starting backfill for all collections")

        result: ListCollectionsResult = list_collections()
        success_count = 0
        failed_count = 0
        failed_collections = []

        for collection_info in result.collections:
            collection_name = collection_info.name
            backfill_result = await self.backfill_collection(collection_name)
            if backfill_result["status"] == "success":
                success_count += 1
            else:
                failed_count += 1
                failed_collections.append(collection_name)

        logger.info(
            "Backfill completed: %d succeeded, %d failed",
            success_count,
            failed_count,
        )

        return {
            "status": "complete",
            "total_collections": result.total_count,
            "success_count": success_count,
            "failed_count": failed_count,
            "failed_collections": failed_collections,
        }

    def set_write_mode(self, mode: str) -> None:
        """Change write mode dynamically.

        Args:
            mode: New write mode - 'lancedb', 'postgresql', or 'both'.
        """
        if mode not in ("lancedb", "postgresql", "both"):
            raise ValueError(f"Invalid write_mode: {mode}")

        old_mode = self._write_mode
        self._write_mode = mode  # type: ignore[assignment]
        self._metadata = self._create_metadata_store()

        logger.info("Write mode changed from '%s' to '%s'", old_mode, mode)

    def set_read_backend(self, backend: MetadataBackend) -> None:
        """Change read backend dynamically.

        This method immediately affects read operations. If using dual-write mode,
        the DualWriteMetadataStore's read backend will also be updated.

        Args:
            backend: New read backend (must be MetadataBackend enum).
        """
        if not isinstance(backend, MetadataBackend):
            raise ValueError(
                f"Invalid backend: {backend}. Must be MetadataBackend enum. "
                "Use MetadataBackend.LANCEDB or MetadataBackend.POSTGRESQL"
            )

        old_backend = self._read_backend
        self._read_backend = backend

        # If using dual-write mode, also update the metadata store's read backend
        if isinstance(self._metadata, DualWriteMetadataStore):
            self._metadata.set_read_backend(backend)

        logger.info(
            "Read backend changed from '%s' to '%s'",
            old_backend.value,
            backend.value,
        )


class DualWriteMetadataStore(MetadataStore):
    """Metadata store that writes to both LanceDB and PostgreSQL.

    Used during migration phase to ensure both backends stay in sync.
    Reads from the configured read backend (can be switched dynamically).
    """

    def __init__(
        self,
        lancedb_store: MetadataStore,
        pg_store: MetadataStore,
        stats: DualWriteStats,
        read_backend: MetadataBackend = MetadataBackend.LANCEDB,
    ) -> None:
        """Initialize dual-write metadata store.

        Args:
            lancedb_store: LanceDB metadata store.
            pg_store: PostgreSQL metadata store.
            stats: Statistics tracker for dual-write operations.
            read_backend: Which backend to read from (default: LanceDB).
        """
        self._lancedb_store = lancedb_store
        self._pg_store = pg_store
        self._stats = stats
        self._read_backend = read_backend

    def set_read_backend(self, backend: MetadataBackend) -> None:
        """Switch the read backend dynamically.

        Args:
            backend: New backend to read from.
        """
        if not isinstance(backend, MetadataBackend):
            raise ValueError(
                f"Invalid backend: {backend}. Must be MetadataBackend enum"
            )

        old_backend = self._read_backend
        self._read_backend = backend
        logger.info(
            "Read backend switched from '%s' to '%s'",
            old_backend.value,
            backend.value,
        )

    def _get_read_store(self) -> MetadataStore:
        """Get the backend to read from based on current configuration.

        Returns:
            MetadataStore to read from.
        """
        if self._read_backend == MetadataBackend.POSTGRESQL:
            return self._pg_store
        return self._lancedb_store

    async def get_collection(self, collection_name: str) -> CollectionInfo:
        """Read from the configured read backend."""
        store = self._get_read_store()
        return await store.get_collection(collection_name)

    async def save_collection(self, collection: CollectionInfo) -> None:
        """Write to both backends."""
        self._stats.last_write_time = datetime.now(timezone.utc)

        # Write to LanceDB
        try:
            await self._lancedb_store.save_collection(collection)
            self._stats.writes_to_primary += 1
        except Exception as e:
            logger.error("Failed to write to LanceDB backend: %s", e)
            self._stats.write_failures += 1
            raise

        # Write to PostgreSQL
        try:
            await self._pg_store.save_collection(collection)
            self._stats.writes_to_secondary += 1
        except Exception as e:
            logger.error("Failed to write to PostgreSQL backend: %s", e)
            self._stats.write_failures += 1
            # Don't raise - allow LanceDB write to succeed

    async def ensure_collection_metadata_table(self) -> None:
        """Ensure tables exist in both backends."""
        await self._lancedb_store.ensure_collection_metadata_table()
        await self._pg_store.ensure_collection_metadata_table()

    async def save_collection_config(
        self,
        collection: str,
        config_json: str,
        user_id: int,
    ) -> None:
        """Save config to both backends."""
        self._stats.last_write_time = datetime.now(timezone.utc)

        # Write to LanceDB
        try:
            await self._lancedb_store.save_collection_config(
                collection, config_json, user_id
            )
            self._stats.writes_to_primary += 1
        except Exception as e:
            logger.error("Failed to write config to LanceDB backend: %s", e)
            self._stats.write_failures += 1
            raise

        # Write to PostgreSQL
        try:
            await self._pg_store.save_collection_config(
                collection, config_json, user_id
            )
            self._stats.writes_to_secondary += 1
        except Exception as e:
            logger.error("Failed to write config to PostgreSQL backend: %s", e)
            self._stats.write_failures += 1

    async def get_collection_config(
        self,
        collection: str,
        user_id: int,
    ) -> str | None:
        """Read from the configured read backend."""
        store = self._get_read_store()
        return await store.get_collection_config(collection, user_id)

    def get_raw_connection(self) -> Any:
        """Return LanceDB backend connection."""
        return self._lancedb_store.get_raw_connection()
