"""Storage contracts for KB control-plane and vector-plane operations.

Phase 1A introduces these contracts to decouple API/business modules from
backend-specific database semantics.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import (
    Any,
    Dict,
    Iterator,
    List,
    Optional,
    Protocol,
    Sequence,
    Union,
    runtime_checkable,
)

from ..core.config import DEFAULT_VECTOR_STORE_SCAN_LIMIT
from ..core.schemas import CollectionInfo


@runtime_checkable
class DatabaseConnection(Protocol):
    """Backend-agnostic database connection protocol.

    This protocol defines the minimal interface required for storage
    implementations to work with different database backends without
    importing concrete types like LanceDB's DBConnection.
    """

    def open_table(self, name: str) -> Any: ...

    def table_names(self) -> Sequence[str]: ...


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


class FilterOperator(str, Enum):
    """Comparison operators for filter expressions.

    These operators provide a backend-agnostic way to express filter conditions
    that can be translated to backend-specific query languages.
    """

    EQ = "eq"  # Equal
    NE = "ne"  # Not equal
    GT = "gt"  # Greater than
    GTE = "gte"  # Greater than or equal
    LT = "lt"  # Less than
    LTE = "lte"  # Less than or equal
    IN = "in"  # In list
    CONTAINS = "contains"  # String contains


@dataclass(frozen=True)
class FilterCondition:
    """Single filter condition.

    Attributes:
        field: Field name to filter on.
        operator: Comparison operator.
        value: Value to compare against.

    Raises:
        ValueError: If operator requires list value but value is not a list.
    """

    field: str
    operator: FilterOperator
    value: Any

    def __post_init__(self):
        # Validate operator matches value type
        if self.operator in {FilterOperator.IN}:
            if not isinstance(self.value, (list, tuple, set)):
                raise ValueError(
                    f"IN operator requires list/tuple/set value, got {type(self.value)}"
                )


# Filter expression can be a single condition, AND combination (tuple), or OR combination (list)
# Use string annotation for recursive type definition
FilterExpression = Union[
    FilterCondition,  # Single condition
    "tuple[FilterExpression, ...]",  # AND combination
    "list[FilterExpression]",  # OR combination
]


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
    async def save_collection_config(
        self,
        collection: str,
        config_json: str,
        user_id: int,
    ) -> None:
        """Save collection ingestion configuration.

        Args:
            collection: Collection name.
            config_json: JSON string of IngestionConfig.
            user_id: User ID for multi-tenancy.
        """

    @abstractmethod
    async def get_collection_config(
        self,
        collection: str,
        user_id: int,
    ) -> str | None:
        """Get collection ingestion configuration.

        Args:
            collection: Collection name.
            user_id: User ID for multi-tenancy.

        Returns:
            Config JSON string if found, None otherwise.
        """

    @abstractmethod
    def get_raw_connection(self) -> Any:
        """Return raw backend connection for legacy compatibility paths.

        The returned object conforms to the DatabaseConnection protocol but
        uses Any type to avoid importing backend-specific types.
        """


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
    def delete_collection_data(
        self,
        collection_name: str,
    ) -> Dict[str, int]:
        """Delete all data for a collection from vector-side tables.

        Args:
            collection_name: Name of the collection to delete.

        Returns:
            Dictionary mapping table names to deleted row counts.
        """

    @abstractmethod
    def aggregate_collection_stats(
        self,
        user_id: Optional[int],
        is_admin: bool,
    ) -> Dict[str, Dict[str, int]]:
        """Aggregate statistics for all collections.

        Returns:
            Dictionary mapping collection names to their stats:
            {
                "collection_name": {
                    "documents": int,
                    "parses": int,
                    "chunks": int,
                    "embeddings": int,
                }
            }
        """

    @abstractmethod
    def aggregate_document_stats(
        self,
        collection_name: str,
        doc_id: str,
        user_id: Optional[int],
        is_admin: bool,
    ) -> Dict[str, int]:
        """Aggregate statistics for a single document.

        Returns:
            Dictionary with counts:
            {
                "documents": int,
                "parses": int,
                "chunks": int,
                "embeddings": int,
            }
        """

    @abstractmethod
    def list_table_names(self) -> Sequence[str]:
        """List backend table names."""

    @abstractmethod
    def iter_batches(
        self,
        table_name: str,
        columns: Optional[Sequence[str]] = None,
        batch_size: int = 1000,
        filters: Optional[Dict[str, Any]] = None,
        user_id: Optional[int] = None,
        is_admin: bool = False,
    ) -> Iterator[Any]:
        """Iterate over table data in batches.

        Yields backend-specific batch objects (e.g., PyArrow RecordBatch).
        This method is designed for memory-efficient processing of large tables.

        Args:
            table_name: Name of table to iterate.
            columns: Optional columns to select. If None, selects all columns.
            batch_size: Rows per batch.
            filters: Optional filter criteria (key-value pairs for equality).
            user_id: Optional user filter for multi-tenancy.
            is_admin: Admin privilege flag.

        Yields:
            Backend-specific batch objects (e.g., PyArrow RecordBatch).
        """

    @abstractmethod
    def count_rows(
        self,
        table_name: str,
        filters: Optional[Dict[str, Any]] = None,
        user_id: Optional[int] = None,
        is_admin: bool = False,
    ) -> int:
        """Count rows in a table with optional filters.

        Args:
            table_name: Name of table to count.
            filters: Optional filter criteria (key-value pairs for equality).
            user_id: Optional user filter for multi-tenancy.
            is_admin: Admin privilege flag.

        Returns:
            Row count (0 on error).
        """

    @abstractmethod
    def aggregate_document_counts(
        self,
        table_name: str,
        doc_id_column: str,
        collection_name: str,
        user_id: Optional[int] = None,
        is_admin: bool = False,
    ) -> Dict[str, int]:
        """Aggregate records per document for a specific table.

        Args:
            table_name: Table to aggregate from.
            doc_id_column: Column containing document IDs.
            collection_name: Collection to scope to.
            user_id: Optional user filter.
            is_admin: Admin privilege flag.

        Returns:
            Dictionary mapping doc_id to count.
        """

    @abstractmethod
    def build_filter_expression(
        self,
        filters: Optional[FilterExpression],
        user_id: Optional[int] = None,
        is_admin: bool = False,
    ) -> Optional[str]:
        """Convert abstract filter expression to backend-specific syntax.

        Args:
            filters: Abstract filter expression.
            user_id: Optional user for multi-tenancy.
            is_admin: Admin privilege flag.

        Returns:
            Backend-specific filter string, or None if no filters.
        """

    @abstractmethod
    def get_raw_connection(self) -> Any:
        """Return raw backend connection for legacy compatibility paths.

        The returned object conforms to the DatabaseConnection protocol but
        uses Any type to avoid importing backend-specific types.
        """


class KBWriteCoordinator(ABC):
    """Coordinator contract for write/delete orchestration."""

    @abstractmethod
    def metadata_store(self) -> MetadataStore:
        """Return configured metadata store."""

    @abstractmethod
    def vector_index_store(self) -> VectorIndexStore:
        """Return configured vector index store."""
