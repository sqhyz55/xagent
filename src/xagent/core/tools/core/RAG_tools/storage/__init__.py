"""Storage contracts and default implementations for KB."""

from .contracts import (
    KBWriteCoordinator,
    MetadataStore,
    VectorIndexStore,
)
from .factory import (
    get_kb_write_coordinator,
    get_metadata_store,
    get_vector_index_store,
    reset_kb_write_coordinator,
)

__all__ = [
    "KBWriteCoordinator",
    "MetadataStore",
    "VectorIndexStore",
    "get_kb_write_coordinator",
    "get_metadata_store",
    "get_vector_index_store",
    "reset_kb_write_coordinator",
]
