"""Unified factory for all KB storage contracts.

Phase 1A Part 2: StorageFactory manages singleton instances of all stores
with lazy initialization and thread-safe access.

Backward compatibility: Convenience functions (get_vector_index_store, etc.)
are provided for existing code.
"""

from __future__ import annotations

import threading
from typing import Optional

from .contracts import (
    KBWriteCoordinator,
    MainPointerStore,
    MetadataStore,
    PromptTemplateStore,
    VectorIndexStore,
    IngestionStatusStore,
)
from .lancedb_stores import (
    LanceDBIngestionStatusStore,
    LanceDBMainPointerStore,
    LanceDBMetadataStore,
    LanceDBPromptTemplateStore,
    LanceDBVectorIndexStore,
)


class StorageFactory:
    """Unified factory for all storage contracts.

    Manages singleton instances of all stores with lazy initialization
    and thread-safe access using double-checked locking.

    Usage:
        factory = StorageFactory.get_factory()
        vector_store = factory.get_vector_index_store()
        metadata_store = factory.get_metadata_store()
    """

    _instance: Optional[StorageFactory] = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        """Private constructor - use get_factory() instead."""
        if StorageFactory._instance is not None:
            raise RuntimeError("Use get_factory() to get StorageFactory instance")

        # Store instances (lazy initialization)
        self._vector_index_store: Optional[VectorIndexStore] = None
        self._metadata_store: Optional[MetadataStore] = None
        self._ingestion_status_store: Optional[IngestionStatusStore] = None
        self._prompt_template_store: Optional[PromptTemplateStore] = None
        self._main_pointer_store: Optional[MainPointerStore] = None
        self._coordinator: Optional[KBWriteCoordinator] = None

    @classmethod
    def get_factory(cls) -> StorageFactory:
        """Get singleton factory instance.

        Uses double-checked locking for thread-safe lazy initialization.

        Returns:
            The singleton StorageFactory instance.
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def reset_all(self) -> None:
        """Reset all store instances.

        Useful for tests/fixtures that need isolated storage.
        Thread-safe: uses factory lock to prevent race conditions.
        """
        with self._lock:
            self._vector_index_store = None
            self._metadata_store = None
            self._ingestion_status_store = None
            self._prompt_template_store = None
            self._main_pointer_store = None
            self._coordinator = None

    # --- VectorIndexStore ---

    def get_vector_index_store(self) -> VectorIndexStore:
        """Get or create vector index store.

        Returns:
            LanceDBVectorIndexStore instance.
        """
        if self._vector_index_store is None:
            with self._lock:
                if self._vector_index_store is None:
                    self._vector_index_store = LanceDBVectorIndexStore()
        return self._vector_index_store

    # --- MetadataStore ---

    def get_metadata_store(self) -> MetadataStore:
        """Get or create metadata store.

        Returns:
            LanceDBMetadataStore instance.
        """
        if self._metadata_store is None:
            with self._lock:
                if self._metadata_store is None:
                    self._metadata_store = LanceDBMetadataStore()
        return self._metadata_store

    # --- IngestionStatusStore ---

    def get_ingestion_status_store(self) -> IngestionStatusStore:
        """Get or create ingestion status store.

        Returns:
            LanceDBIngestionStatusStore instance.
        """
        if self._ingestion_status_store is None:
            with self._lock:
                if self._ingestion_status_store is None:
                    self._ingestion_status_store = LanceDBIngestionStatusStore()
        return self._ingestion_status_store

    # --- PromptTemplateStore ---

    def get_prompt_template_store(self) -> PromptTemplateStore:
        """Get or create prompt template store.

        Returns:
            LanceDBPromptTemplateStore instance.
        """
        if self._prompt_template_store is None:
            with self._lock:
                if self._prompt_template_store is None:
                    self._prompt_template_store = LanceDBPromptTemplateStore()
        return self._prompt_template_store

    # --- MainPointerStore ---

    def get_main_pointer_store(self) -> MainPointerStore:
        """Get or create main pointer store.

        Returns:
            LanceDBMainPointerStore instance.
        """
        if self._main_pointer_store is None:
            with self._lock:
                if self._main_pointer_store is None:
                    self._main_pointer_store = LanceDBMainPointerStore()
        return self._main_pointer_store

    # --- KBWriteCoordinator ---

    def get_kb_write_coordinator(self) -> KBWriteCoordinator:
        """Get or create KB write coordinator.

        Returns:
            DefaultKBWriteCoordinator instance.
        """
        if self._coordinator is None:
            with self._lock:
                if self._coordinator is None:
                    self._coordinator = DefaultKBWriteCoordinator(
                        metadata=self.get_metadata_store(),
                        vector_index=self.get_vector_index_store(),
                    )
        return self._coordinator


# ============================================================================
# Backward Compatibility Functions
# ============================================================================

# Module-level lock for backward compatibility functions
_compat_lock = threading.Lock()
_default_factory: Optional[StorageFactory] = None


def _get_default_factory() -> StorageFactory:
    """Get or create default factory instance (thread-safe)."""
    global _default_factory
    if _default_factory is None:
        with _compat_lock:
            if _default_factory is None:
                _default_factory = StorageFactory.get_factory()
    return _default_factory


def reset_kb_write_coordinator() -> None:
    """Reset process-global coordinator (useful for tests/fixtures).

    Deprecated: Use StorageFactory.get_factory().reset_all() instead.
    """
    _get_default_factory().reset_all()


def get_kb_write_coordinator() -> KBWriteCoordinator:
    """Return process-global KB write coordinator.

    Deprecated: Use StorageFactory.get_factory().get_kb_write_coordinator() instead.
    """
    return _get_default_factory().get_kb_write_coordinator()


def get_metadata_store() -> MetadataStore:
    """Convenience accessor for metadata store.

    Deprecated: Use StorageFactory.get_factory().get_metadata_store() instead.
    """
    return _get_default_factory().get_metadata_store()


def get_vector_index_store() -> VectorIndexStore:
    """Convenience accessor for vector index store.

    Deprecated: Use StorageFactory.get_factory().get_vector_index_store() instead.
    """
    return _get_default_factory().get_vector_index_store()


def get_ingestion_status_store() -> IngestionStatusStore:
    """Get ingestion status store.

    Returns:
        LanceDBIngestionStatusStore instance.
    """
    return _get_default_factory().get_ingestion_status_store()


def get_prompt_template_store() -> PromptTemplateStore:
    """Get prompt template store.

    Returns:
        LanceDBPromptTemplateStore instance.
    """
    return _get_default_factory().get_prompt_template_store()


def get_main_pointer_store() -> MainPointerStore:
    """Get main pointer store.

    Returns:
        LanceDBMainPointerStore instance.
    """
    return _get_default_factory().get_main_pointer_store()


# ============================================================================
# Default Coordinator Implementation
# ============================================================================


class DefaultKBWriteCoordinator(KBWriteCoordinator):
    """Default in-process coordinator (Phase 1A contract shell)."""

    def __init__(
        self,
        metadata: MetadataStore | None = None,
        vector_index: VectorIndexStore | None = None,
    ) -> None:
        self._metadata = metadata or LanceDBMetadataStore()
        self._vector_index = vector_index or LanceDBVectorIndexStore()

    def metadata_store(self) -> MetadataStore:
        return self._metadata

    def vector_index_store(self) -> VectorIndexStore:
        return self._vector_index
