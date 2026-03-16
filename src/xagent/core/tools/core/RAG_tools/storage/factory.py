"""Factory and default coordinator for KB storage contracts."""

from __future__ import annotations

from .contracts import KBWriteCoordinator, MetadataStore, VectorIndexStore
from .lancedb_stores import LanceDBMetadataStore, LanceDBVectorIndexStore


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
