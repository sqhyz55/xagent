"""Storage contracts for KB control-plane and vector-plane operations.

Phase 1A introduces these contracts to decouple API/business modules from
backend-specific database semantics.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Sequence

from lancedb.db import DBConnection

from ..core.config import DEFAULT_VECTOR_STORE_SCAN_LIMIT
from ..core.schemas import CollectionInfo


@dataclass(frozen=True)
class DocumentRecord:
    """Lightweight document projection for metadata/control operations.

    Attributes:
        doc_id: Document identifier.
        file_id: Optional file identifier for uploaded file tracking.
        source_path: Original source path if available.
    """

    doc_id: str
    file_id: Optional[str] = None
    source_path: Optional[str] = None


class MetadataStore(ABC):
    """Control-plane metadata storage contract."""

    @abstractmethod
    async def get_collection(self, collection_name: str) -> CollectionInfo:
        """Read collection metadata.

        Args:
            collection_name: Target collection name.

        Returns:
            Collection metadata.

        Raises:
            ValueError: If collection is not found.
        """

    @abstractmethod
    async def save_collection(self, collection: CollectionInfo) -> None:
        """Create or update collection metadata."""

    @abstractmethod
    async def ensure_collection_metadata_table(self) -> None:
        """Ensure control-plane metadata table exists."""

    @abstractmethod
    def get_raw_connection(self) -> DBConnection:
        """Return raw backend connection for legacy compatibility paths."""


class VectorIndexStore(ABC):
    """Vector/data-plane storage contract."""

    @abstractmethod
    def list_document_records(
        self,
        collection_name: str,
        user_id: Optional[int],
        is_admin: bool,
        max_results: int = DEFAULT_VECTOR_STORE_SCAN_LIMIT,
    ) -> List[DocumentRecord]:
        """List document records from vector index side."""

    @abstractmethod
    def rename_collection_data(
        self,
        collection_name: str,
        new_name: str,
    ) -> List[str]:
        """Rename collection key across vector-side tables.

        Returns:
            Warning messages generated during best-effort updates.
        """

    @abstractmethod
    def list_table_names(self) -> Sequence[str]:
        """List backend table names."""

    @abstractmethod
    def get_raw_connection(self) -> DBConnection:
        """Return raw backend connection for legacy compatibility paths."""


class KBWriteCoordinator(ABC):
    """Coordinator contract for write/delete orchestration."""

    @abstractmethod
    def metadata_store(self) -> MetadataStore:
        """Return configured metadata store."""

    @abstractmethod
    def vector_index_store(self) -> VectorIndexStore:
        """Return configured vector index store."""
