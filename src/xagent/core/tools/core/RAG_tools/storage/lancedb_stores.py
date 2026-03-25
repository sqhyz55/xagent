"""LanceDB-backed implementations of storage contracts."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence

import pyarrow as pa  # type: ignore
from lancedb.db import DBConnection

from xagent.providers.vector_store.lancedb import get_connection_from_env

from ..core.config import DEFAULT_VECTOR_STORE_SCAN_LIMIT
from ..core.schemas import CollectionInfo
from ..LanceDB.schema_manager import ensure_documents_table
from ..utils.lancedb_query_utils import query_to_list
from ..utils.string_utils import build_lancedb_filter_expression, escape_lancedb_string
from ..utils.user_permissions import UserPermissions
from .contracts import DocumentRecord, MetadataStore, VectorIndexStore

logger = logging.getLogger(__name__)


class LanceDBMetadataStore(MetadataStore):
    """LanceDB implementation for control-plane metadata operations."""

    def __init__(self) -> None:
        self._conn: Optional[DBConnection] = None

    async def _get_connection(self) -> DBConnection:
        if self._conn is None:
            self._conn = get_connection_from_env()
        return self._conn

    async def get_collection(self, collection_name: str) -> CollectionInfo:
        conn = await self._get_connection()
        table = conn.open_table("collection_metadata")
        safe_name = escape_lancedb_string(collection_name)
        result = table.search().where(f"name = '{safe_name}'").to_pandas()
        if result.empty:
            raise ValueError(f"Collection '{collection_name}' not found")
        data = result.iloc[0].to_dict()
        return CollectionInfo.from_storage(data)

    async def save_collection(self, collection: CollectionInfo) -> None:
        conn = await self._get_connection()
        await self.ensure_collection_metadata_table()

        data = collection.to_storage()
        data["updated_at"] = datetime.now(timezone.utc).replace(tzinfo=None)

        table = conn.open_table("collection_metadata")
        safe_name = escape_lancedb_string(collection.name)
        existing = table.search().where(f"name = '{safe_name}'").to_pandas()
        if not existing.empty:
            table.delete(f"name = '{safe_name}'")
        table.add([data])

    async def ensure_collection_metadata_table(self) -> None:
        conn = await self._get_connection()
        schema = pa.schema(
            [
                ("name", pa.string()),
                ("schema_version", pa.string()),
                ("embedding_model_id", pa.string()),
                ("embedding_dimension", pa.int32()),
                ("documents", pa.int32()),
                ("processed_documents", pa.int32()),
                ("parses", pa.int32()),
                ("chunks", pa.int32()),
                ("embeddings", pa.int32()),
                ("document_names", pa.string()),
                ("collection_locked", pa.bool_()),
                ("allow_mixed_parse_methods", pa.bool_()),
                ("skip_config_validation", pa.bool_()),
                ("ingestion_config", pa.string()),
                # Phase 1B fields
                ("owner_user_id", pa.int32()),
                ("external_file_id", pa.string()),
                ("created_at", pa.timestamp("us")),
                ("updated_at", pa.timestamp("us")),
                ("last_accessed_at", pa.timestamp("us")),
                ("extra_metadata", pa.string()),
            ]
        )
        table_names_fn = getattr(conn, "table_names", None)
        table_exists = False
        if table_names_fn:
            try:
                table_exists = "collection_metadata" in table_names_fn()
            except Exception as exc:  # noqa: BLE001
                logger.debug("collection_metadata existence check failed: %s", exc)
        if not table_exists:
            try:
                conn.create_table("collection_metadata", schema=schema)
            except Exception as exc:  # noqa: BLE001
                logger.debug("collection_metadata create_table no-op/failure: %s", exc)

    async def save_collection_config(
        self,
        collection: str,
        config_json: str,
        user_id: int,
    ) -> None:
        """Save collection ingestion configuration to LanceDB."""
        from ..LanceDB.schema_manager import ensure_collection_config_table

        conn = await self._get_connection()
        ensure_collection_config_table(conn)

        table = conn.open_table("collection_config")
        safe_collection = escape_lancedb_string(collection)

        # Delete existing config for this collection and user
        try:
            table.delete(f"collection = '{safe_collection}' AND user_id = {user_id}")
        except Exception as exc:
            logger.debug("Error deleting old config: %s", exc)

        # Insert new config
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        data = [
            {
                "collection": collection,
                "config_json": config_json,
                "updated_at": now,
                "user_id": user_id,
            }
        ]
        table.add(data)

    async def get_collection_config(
        self,
        collection: str,
        user_id: int,
    ) -> str | None:
        """Get collection ingestion configuration from LanceDB."""
        from ..LanceDB.schema_manager import ensure_collection_config_table

        try:
            conn = await self._get_connection()
            ensure_collection_config_table(conn)

            table = conn.open_table("collection_config")
            safe_collection = escape_lancedb_string(collection)
            result = (
                table.search()
                .where(f"collection = '{safe_collection}' AND user_id = {user_id}")
                .to_pandas()
            )

            if result.empty:
                return None
            return str(result.iloc[0]["config_json"])
        except Exception as exc:
            logger.debug("Error reading collection config: %s", exc)
            return None

    def get_raw_connection(self) -> DBConnection:
        return get_connection_from_env() if self._conn is None else self._conn


class LanceDBVectorIndexStore(VectorIndexStore):
    """LanceDB implementation for vector/data-plane operations."""

    def __init__(self) -> None:
        self._conn: Optional[DBConnection] = None

    def _get_connection(self) -> DBConnection:
        if self._conn is None:
            self._conn = get_connection_from_env()
        return self._conn

    def list_document_records(
        self,
        collection_name: str,
        user_id: Optional[int],
        is_admin: bool,
        max_results: int = DEFAULT_VECTOR_STORE_SCAN_LIMIT,
    ) -> List[DocumentRecord]:
        conn = self._get_connection()
        ensure_documents_table(conn)
        table = conn.open_table("documents")
        base_filter = build_lancedb_filter_expression({"collection": collection_name})
        user_filter = UserPermissions.get_user_filter(user_id, is_admin)
        if user_filter and base_filter:
            combined_filter = f"({base_filter}) and ({user_filter})"
        else:
            combined_filter = user_filter or base_filter

        raw_records = query_to_list(
            table.search().where(combined_filter).limit(max_results)
            if combined_filter
            else table.search().limit(max_results)
        )

        records: List[DocumentRecord] = []
        for item in raw_records:
            raw_doc_id = item.get("doc_id")
            if not raw_doc_id:
                continue
            records.append(
                DocumentRecord(
                    doc_id=str(raw_doc_id),
                    source_path=(
                        str(item["source_path"]) if item.get("source_path") else None
                    ),
                )
            )
        return records

    def rename_collection_data(
        self,
        collection_name: str,
        new_name: str,
    ) -> List[str]:
        warnings: List[str] = []
        safe_old_name = escape_lancedb_string(collection_name)
        conn = self._get_connection()
        for table_name in self.list_table_names():
            if table_name not in {
                "documents",
                "parses",
                "chunks",
            } and not table_name.startswith("embeddings_"):
                continue
            try:
                table = conn.open_table(table_name)
                table.update(
                    f"collection = '{safe_old_name}'",
                    {"collection": new_name},
                )
            except Exception as exc:  # noqa: BLE001
                message = f"Failed to update '{table_name}': {exc}"
                logger.warning(message)
                warnings.append(message)
        return warnings

    def list_table_names(self) -> Sequence[str]:
        conn = self._get_connection()
        table_names_fn = getattr(conn, "table_names", None)
        if table_names_fn is None:
            return []
        try:
            return [str(name) for name in table_names_fn()]
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to list LanceDB tables: %s", exc)
            return []

    def delete_collection_data(
        self,
        collection_name: str,
    ) -> Dict[str, int]:
        """Delete all data for a collection from vector-side tables."""
        from ..LanceDB.schema_manager import (
            ensure_chunks_table,
            ensure_documents_table,
            ensure_parses_table,
        )

        deleted_counts: Dict[str, int] = {}
        conn = self._get_connection()
        safe_collection = escape_lancedb_string(collection_name)

        # Ensure tables exist before attempting deletion
        ensure_documents_table(conn)
        ensure_parses_table(conn)
        ensure_chunks_table(conn)

        # Delete from core tables
        for table_name in ["documents", "parses", "chunks"]:
            try:
                table = conn.open_table(table_name)
                original_count = table.count_rows()
                table.delete(f"collection = '{safe_collection}'")
                deleted_count = original_count - table.count_rows()
                if deleted_count > 0:
                    deleted_counts[table_name] = deleted_count
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to delete from '%s': %s", table_name, exc)

        # Delete embeddings data
        for table_name in self.list_table_names():
            if not table_name.startswith("embeddings_"):
                continue
            try:
                table = conn.open_table(table_name)
                original_count = table.count_rows()
                table.delete(f"collection = '{safe_collection}'")
                deleted_count = original_count - table.count_rows()
                if deleted_count > 0:
                    deleted_counts[table_name] = deleted_count
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to delete from '%s': %s", table_name, exc)

        return deleted_counts

    def aggregate_collection_stats(
        self,
        user_id: Optional[int],
        is_admin: bool,
    ) -> Dict[str, Dict[str, int]]:
        """Aggregate statistics for all collections."""
        from ..LanceDB.schema_manager import (
            ensure_chunks_table,
            ensure_documents_table,
            ensure_parses_table,
        )
        from ..utils.lancedb_query_utils import query_to_list

        stats: Dict[str, Dict[str, int]] = {}
        conn = self._get_connection()

        # Ensure tables exist
        ensure_documents_table(conn)
        ensure_parses_table(conn)
        ensure_chunks_table(conn)

        # Get user filter for multi-tenancy
        user_filter = UserPermissions.get_user_filter(user_id, is_admin)

        def _count_table(table_name: str, stat_key: str) -> None:
            try:
                table = conn.open_table(table_name)
                if user_filter:
                    results = query_to_list(table.search().where(user_filter))
                else:
                    results = query_to_list(table.search())

                for item in results:
                    collection = str(item.get("collection", ""))
                    if collection:
                        if collection not in stats:
                            stats[collection] = {
                                "documents": 0,
                                "parses": 0,
                                "chunks": 0,
                                "embeddings": 0,
                            }
                        stats[collection][stat_key] += 1
            except Exception as exc:  # noqa: BLE001
                logger.debug("Failed to count table '%s': %s", table_name, exc)

        # Count documents
        _count_table("documents", "documents")
        _count_table("parses", "parses")
        _count_table("chunks", "chunks")

        # Count embeddings
        for table_name in self.list_table_names():
            if not table_name.startswith("embeddings_"):
                continue
            _count_table(table_name, "embeddings")

        return stats

    def aggregate_document_stats(
        self,
        collection_name: str,
        doc_id: str,
        user_id: Optional[int],
        is_admin: bool,
    ) -> Dict[str, int]:
        """Aggregate statistics for a single document."""
        from ..LanceDB.schema_manager import (
            ensure_chunks_table,
            ensure_documents_table,
            ensure_parses_table,
        )

        stats = {"documents": 0, "parses": 0, "chunks": 0, "embeddings": 0}
        conn = self._get_connection()

        # Ensure tables exist
        ensure_documents_table(conn)
        ensure_parses_table(conn)
        ensure_chunks_table(conn)

        safe_collection = escape_lancedb_string(collection_name)
        safe_doc_id = escape_lancedb_string(doc_id)

        base_filter = f"collection = '{safe_collection}' AND doc_id = '{safe_doc_id}'"

        def _count_table(table_name: str) -> int:
            try:
                table = conn.open_table(table_name)
                return int(table.count_rows(base_filter))
            except Exception:  # noqa: BLE001
                return 0

        stats["documents"] = _count_table("documents")
        stats["parses"] = _count_table("parses")
        stats["chunks"] = _count_table("chunks")

        # Count embeddings across all embeddings tables
        for table_name in self.list_table_names():
            if not table_name.startswith("embeddings_"):
                continue
            stats["embeddings"] += _count_table(table_name)

        return stats

    def get_raw_connection(self) -> DBConnection:
        return self._get_connection()
