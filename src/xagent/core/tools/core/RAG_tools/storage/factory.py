"""Factory and default coordinator for KB storage contracts.

Phase 1B: Backend selection via environment variable with dual-write support.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Literal

from .contracts import KBWriteCoordinator, MetadataStore, VectorIndexStore
from .dual_write_coordinator import DualWriteCoordinator
from .lancedb_stores import LanceDBMetadataStore, LanceDBVectorIndexStore

# Import PostgreSQL store for Phase 1B
try:
    from .pg_metadata_store import PostgreSQLMetadataStore

    _POSTGRESQL_AVAILABLE = True
except Exception:
    _POSTGRESQL_AVAILABLE = False

logger = logging.getLogger(__name__)

# Environment variables to control storage backends
# RAG_METADATA_STORE_BACKEND: 'lancedb', 'postgresql' (default: 'lancedb')
# RAG_DUAL_WRITE_ENABLED: Enable dual-write mode (default: 'false')
# RAG_READ_BACKEND: 'lancedb' or 'postgresql' (default: 'lancedb')
# RAG_WRITE_BACKEND: 'lancedb', 'postgresql', or 'both' (default: 'lancedb')
METADATA_STORE_BACKEND: Literal["lancedb", "postgresql"] = os.environ.get(
    "RAG_METADATA_STORE_BACKEND", "lancedb"
).lower()  # type: ignore

DUAL_WRITE_ENABLED: bool = (
    os.environ.get("RAG_DUAL_WRITE_ENABLED", "false").lower() == "true"
)

READ_BACKEND: Literal["lancedb", "postgresql"] = os.environ.get(
    "RAG_READ_BACKEND", "lancedb"
).lower()  # type: ignore

WRITE_BACKEND: Literal["lancedb", "postgresql", "both"] = os.environ.get(
    "RAG_WRITE_BACKEND", "lancedb"
).lower()  # type: ignore


class DefaultKBWriteCoordinator(KBWriteCoordinator):
    """Default in-process coordinator with backend selection (Phase 1B).

    Supports dual-write mode for LanceDB to PostgreSQL migration.
    """

    def __init__(
        self,
        metadata: MetadataStore | None = None,
        vector_index: VectorIndexStore | None = None,
    ) -> None:
        if vector_index is None:
            vector_index = LanceDBVectorIndexStore()
        self._vector_index = vector_index
        self._dual_write_coordinator: DualWriteCoordinator | None = None

        # Check if dual-write mode is enabled
        if DUAL_WRITE_ENABLED:
            logger.info(
                "Dual-write mode enabled: read=%s, write=%s",
                READ_BACKEND,
                WRITE_BACKEND,
            )
            self._metadata = self._create_dual_write_coordinator()
        else:
            if metadata is None:
                metadata = self._create_metadata_store()
            self._metadata = metadata

    def _create_metadata_store(self) -> MetadataStore:
        """Create metadata store based on environment configuration.

        Returns:
            Configured MetadataStore instance.
        """
        if METADATA_STORE_BACKEND == "postgresql":
            if not _POSTGRESQL_AVAILABLE:
                logger.warning(
                    "PostgreSQL backend requested but dependencies not available. "
                    "Falling back to LanceDB."
                )
                return LanceDBMetadataStore()
            logger.info("Using PostgreSQL MetadataStore (Phase 1B)")
            return PostgreSQLMetadataStore()
        else:
            logger.info("Using LanceDB MetadataStore (Phase 1A)")
            return LanceDBMetadataStore()

    def _create_dual_write_coordinator(self) -> MetadataStore:
        """Create dual-write coordinator for migration mode.

        Returns:
            MetadataStore from DualWriteCoordinator.
        """
        if not _POSTGRESQL_AVAILABLE:
            logger.warning(
                "Dual-write requested but PostgreSQL not available. "
                "Falling back to LanceDB-only mode."
            )
            return LanceDBMetadataStore()

        coordinator = DualWriteCoordinator(
            primary_backend="lancedb",
            secondary_backend="postgresql",
            write_mode=WRITE_BACKEND,
            read_backend=READ_BACKEND,
        )
        # Store coordinator for stats access
        self._dual_write_coordinator = coordinator
        return coordinator.metadata_store()

    def metadata_store(self) -> MetadataStore:
        return self._metadata

    def vector_index_store(self) -> VectorIndexStore:
        return self._vector_index

    def get_dual_write_stats(self) -> Any:
        """Get dual-write statistics if dual-write mode is enabled.

        Returns:
            DualWriteStats instance or None if not in dual-write mode.
        """
        if self._dual_write_coordinator is not None:
            return self._dual_write_coordinator.get_stats()
        return None


_default_coordinator: KBWriteCoordinator | None = None


def reset_kb_write_coordinator() -> None:
    """Reset process-global coordinator (useful for tests/fixtures)."""
    global _default_coordinator
    _default_coordinator = None


def get_kb_write_coordinator() -> KBWriteCoordinator:
    """Return process-global KB write coordinator."""
    global _default_coordinator
    if _default_coordinator is None:
        _default_coordinator = DefaultKBWriteCoordinator()
    return _default_coordinator


def get_metadata_store() -> MetadataStore:
    """Convenience accessor for metadata store."""
    return get_kb_write_coordinator().metadata_store()


def get_vector_index_store() -> VectorIndexStore:
    """Convenience accessor for vector index store."""
    return get_kb_write_coordinator().vector_index_store()


def reset_metadata_store() -> None:
    """Reset metadata store singleton.

    Mainly used for testing. Clears the cached coordinator so the next call
    creates a new one with potentially different backend settings.
    """
    global _default_coordinator
    _default_coordinator = None
    logger.debug("KB write coordinator (and metadata store) reset")
