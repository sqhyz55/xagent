"""Collection management utilities for RAG pipelines.

This module provides utilities for managing collections in a RAG (Retrieval-Augmented Generation)
system, including listing collections, managing documents, and handling deletion operations.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Set

import pyarrow as pa  # type: ignore
from lancedb.db import DBConnection

from xagent.providers.vector_store.lancedb import get_connection_from_env

from ..core.config import DEFAULT_LANCEDB_SCAN_BATCH_SIZE
from ..core.schemas import (
    CollectionInfo,
    CollectionOperationDetail,
    CollectionOperationResult,
    DocumentListResult,
    DocumentOperationResult,
    DocumentProcessingStatus,
    DocumentStats,
    DocumentStatsResult,
    DocumentSummary,
    ListCollectionsResult,
)
from ..LanceDB.model_tag_utils import embeddings_table_name
from ..LanceDB.schema_manager import (
    ensure_chunks_table,
    ensure_collection_config_table,
    ensure_documents_table,
    ensure_ingestion_runs_table,
    ensure_parses_table,
)
from ..management.status import (
    clear_ingestion_status,
    load_ingestion_status,
    write_ingestion_status,
)
from ..utils.string_utils import build_lancedb_filter_expression, escape_lancedb_string
from ..utils.user_permissions import UserPermissions
from ..version_management.cascade_cleaner import cleanup_document_cascade

logger = logging.getLogger(__name__)

DEFAULT_BATCH_SIZE = DEFAULT_LANCEDB_SCAN_BATCH_SIZE


def _iter_batches(
    conn: DBConnection,
    table_name: str,
    warnings: List[str],
    columns: Optional[Sequence[str]] = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    user_id: Optional[int] = None,
    is_admin: bool = False,
) -> Any:
    """Yield record batches from a LanceDB table while minimizing memory footprint.

    This generator function iterates through a LanceDB table in batches to
    minimize memory usage, with support for user filtering and column selection.

    Args:
        conn: LanceDB database connection
        table_name: Name of the table to iterate
        warnings: List to collect any warnings encountered
        columns: Optional list of column names to select
        batch_size: Number of rows per batch
        user_id: Optional user ID for multi-tenancy filtering
        is_admin: Whether user has admin privileges

    Yields:
        PyArrow RecordBatch objects containing the data
    """

    try:
        table = conn.open_table(table_name)
    except Exception as exc:  # noqa: BLE001 - convert to warning
        message = f"Unable to open table '{table_name}': {exc}"
        logger.warning(message)
        warnings.append(message)
        return

    column_list = list(columns) if columns is not None else None

    # Apply user filter for multi-tenancy
    user_filter = UserPermissions.get_user_filter(user_id, is_admin)

    # Preferred path: streaming batches directly from LanceDB
    try:
        # Use filter if provided (for multi-tenancy)
        if user_filter is not None:
            for raw_batch in table.to_batches(
                filter=user_filter, batch_size=batch_size
            ):
                batch = raw_batch
                if column_list is not None:
                    arrays = []
                    names = []
                    for col_name in column_list:
                        idx = batch.schema.get_field_index(col_name)
                        if idx == -1:
                            continue
                        arrays.append(batch.column(idx))
                        names.append(col_name)
                    if not arrays:
                        continue
                    batch = pa.RecordBatch.from_arrays(arrays, names=names)
                if batch.num_rows > 0:
                    yield batch
            return

        for raw_batch in table.to_batches(batch_size=batch_size):
            batch = raw_batch
            if column_list is not None:
                arrays = []
                names = []
                for col_name in column_list:
                    idx = batch.schema.get_field_index(col_name)
                    if idx == -1:
                        continue
                    arrays.append(batch.column(idx))
                    names.append(col_name)
                if not arrays:
                    continue
                batch = pa.RecordBatch.from_arrays(arrays, names=names)
            if batch.num_rows > 0:
                yield batch
        return
    except Exception as exc:  # noqa: BLE001 - continue to Arrow fallback
        logger.debug("Batch streaming unavailable for table '%s': %s", table_name, exc)

    # Arrow fallback: materialize table as Arrow then iterate
    try:
        arrow_table = table.to_arrow()
    except Exception as exc:  # noqa: BLE001
        message = f"Unable to read table '{table_name}' via to_arrow(): {exc}"
        logger.warning(message)
        warnings.append(message)
        return

    # Apply user filter on Arrow table if needed
    if user_filter is not None and "user_id" in arrow_table.schema.names:
        try:
            # Parse filter and apply to Arrow table
            # Simple filter parsing for "user_id == X" or "user_id IS NULL"
            import re

            if "user_id IS NULL" in user_filter:
                # Filter for NULL user_id
                mask = pa.compute.is_null(arrow_table["user_id"])
                arrow_table = arrow_table.filter(mask)
            elif "user_id ==" in user_filter:
                # Filter for specific user_id
                match = re.search(r"user_id == (-?\d+)", user_filter)
                if match:
                    user_val = int(match.group(1))
                    mask = pa.compute.equal(
                        arrow_table["user_id"], pa.scalar(user_val, type=pa.int64())
                    )
                    arrow_table = arrow_table.filter(mask)
            elif "user_id == -1" in user_filter:
                # Impossible condition - return empty result
                arrow_table = arrow_table.slice(0, 0)
        except Exception as filter_exc:
            logger.warning("Failed to apply user filter on Arrow table: %s", filter_exc)
            # Continue without filter if filtering fails

    if column_list is not None:
        try:
            arrow_table = arrow_table.select(column_list)
        except Exception as exc:  # noqa: BLE001
            message = (
                f"Table '{table_name}' missing expected columns {column_list}: {exc}"
            )
            logger.warning(message)
            warnings.append(message)
            return

    for batch in arrow_table.to_batches(max_chunksize=batch_size):
        if batch.num_rows > 0:
            yield batch


def _count_rows(
    conn: DBConnection,
    table_name: str,
    filters: Dict[str, str],
    warnings: List[str],
) -> int:
    """Count rows in a LanceDB table while handling failures gracefully.

    This function counts rows in a LanceDB table with optional filters,
    returning 0 on any error and logging warnings.

    Args:
        conn: LanceDB database connection
        table_name: Name of the table to count rows in
        filters: Dictionary of field-value pairs to filter by
        warnings: List to collect any warnings encountered

    Returns:
        Number of rows matching the filter, or 0 on error
    """

    try:
        table = conn.open_table(table_name)
    except Exception as exc:  # noqa: BLE001 - convert to warning
        message = f"Unable to open table '{table_name}': {exc}"
        logger.warning(message)
        warnings.append(message)
        return 0

    filter_expr = build_lancedb_filter_expression(filters)

    try:
        if filter_expr:
            return int(table.count_rows(filter_expr))
        return int(table.count_rows())
    except Exception as exc:  # noqa: BLE001 - convert to warning
        message = f"Failed to count rows in '{table_name}': {exc}"
        logger.warning(message)
        warnings.append(message)
        return 0


def _list_table_names(conn: DBConnection, warnings: List[str]) -> List[str]:
    """Return available LanceDB table names with graceful degradation.

    This function retrieves the list of table names from a LanceDB connection,
    handling errors gracefully by returning an empty list and logging warnings.

    Args:
        conn: LanceDB database connection
        warnings: List to collect any warnings encountered

    Returns:
        List of table names as strings, or empty list on error
    """

    try:
        table_names_fn = getattr(conn, "table_names")
    except AttributeError as exc:
        message = f"LanceDB connection missing table_names(): {exc}"
        logger.warning(message)
        warnings.append(message)
        return []

    try:
        names = table_names_fn()
    except Exception as exc:  # noqa: BLE001 - convert to warning
        message = f"Failed to list LanceDB tables: {exc}"
        logger.warning(message)
        warnings.append(message)
        return []

    return [str(name) for name in names]


def _collect_doc_counts_for_collection(
    conn: DBConnection,
    table_name: str,
    doc_column: str,
    target_collection: str,
    warnings: List[str],
    user_id: Optional[int] = None,
    is_admin: bool = False,
) -> Dict[str, int]:
    """Aggregate per-document counts for the specified table within a collection.

    This function iterates through batches of a table and counts records
    per document for a specific collection.

    Args:
        conn: LanceDB database connection
        table_name: Name of the table to count from
        doc_column: Name of the column containing document IDs
        target_collection: Name of the collection to filter by
        warnings: List to collect any warnings encountered
        user_id: Optional user ID for multi-tenancy filtering
        is_admin: Whether user has admin privileges

    Returns:
        Dictionary mapping document IDs to their counts
    """

    counts: Dict[str, int] = defaultdict(int)

    for batch in _iter_batches(
        conn,
        table_name,
        warnings,
        columns=["collection", doc_column],
        user_id=user_id,
        is_admin=is_admin,
    ):
        collection_idx = batch.schema.get_field_index("collection")
        doc_idx = batch.schema.get_field_index(doc_column)
        if collection_idx == -1 or doc_idx == -1:
            continue
        collection_array = batch.column(collection_idx)
        doc_array = batch.column(doc_idx)
        for idx in range(batch.num_rows):
            collection_raw = collection_array[idx].as_py()
            if not collection_raw or str(collection_raw) != target_collection:
                continue
            doc_raw = doc_array[idx].as_py()
            if not doc_raw:
                continue
            counts[str(doc_raw)] += 1

    return counts


def _collect_document_ids(
    conn: DBConnection,
    collection: str,
    warnings: List[str],
    user_id: Optional[int] = None,
    is_admin: bool = False,
) -> Set[str]:
    """Collect all known document identifiers for a collection.

    This function aggregates document IDs from multiple sources including
    the documents table, chunks table, embeddings tables, and status records.

    Args:
        conn: LanceDB database connection
        collection: Name of the collection to collect document IDs from
        warnings: List to collect any warnings encountered
        user_id: Optional user ID for multi-tenancy filtering
        is_admin: Whether user has admin privileges

    Returns:
        Set of unique document identifiers
    """

    doc_ids: Set[str] = set()

    for batch in _iter_batches(
        conn,
        "documents",
        warnings,
        columns=["collection", "doc_id"],
        user_id=user_id,
        is_admin=is_admin,
    ):
        collection_idx = batch.schema.get_field_index("collection")
        doc_idx = batch.schema.get_field_index("doc_id")
        if collection_idx == -1 or doc_idx == -1:
            continue
        collection_array = batch.column(collection_idx)
        doc_array = batch.column(doc_idx)
        for idx in range(batch.num_rows):
            collection_raw = collection_array[idx].as_py()
            doc_raw = doc_array[idx].as_py()
            if not collection_raw or not doc_raw:
                continue
            if str(collection_raw) == collection:
                doc_ids.add(str(doc_raw))

    chunk_docs = _collect_doc_counts_for_collection(
        conn, "chunks", "doc_id", collection, warnings, user_id, is_admin
    )
    doc_ids.update(chunk_docs.keys())

    for table_name in _list_table_names(conn, warnings):
        if not table_name.startswith("embeddings_"):
            continue
        embed_docs = _collect_doc_counts_for_collection(
            conn, table_name, "doc_id", collection, warnings, user_id, is_admin
        )
        doc_ids.update(embed_docs.keys())

    for status_entry in load_ingestion_status(collection=collection):
        raw_doc = status_entry.get("doc_id")
        if isinstance(raw_doc, str):
            doc_ids.add(raw_doc)

    return doc_ids


def _coerce_timestamp(value: Any) -> datetime | None:
    """Normalize timestamp-like values to aware datetime or None.

    This function attempts to convert various timestamp representations
    (datetime objects, Arrow timestamps, ISO format strings) to a
    standard Python datetime object.

    Args:
        value: The value to convert to datetime

    Returns:
        datetime object if conversion succeeds, None otherwise
    """

    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if hasattr(value, "to_pydatetime"):
        candidate = value.to_pydatetime()
        if isinstance(candidate, datetime):
            return candidate
        return None
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return None
    return None


def list_collections(
    user_id: Optional[int] = None, is_admin: bool = False
) -> ListCollectionsResult:
    """List all knowledge base collections along with aggregated statistics.

    This function returns a list of all available knowledge bases (collections)
    in the system, including document counts and metadata.

    Args:
        user_id: Optional user ID for filtering (for multi-tenancy).
        is_admin: Whether user has admin privileges.

    Returns:
        Aggregated collection metadata and status information for each
        knowledge base, including total documents, parses, chunks, and embeddings.
    """

    logger.info("Listing LanceDB collections")

    warnings: List[str] = []

    try:
        conn = get_connection_from_env()
        ensure_documents_table(conn)
        ensure_parses_table(conn)
        ensure_chunks_table(conn)

        stats: Dict[str, Dict[str, int]] = defaultdict(
            lambda: {"documents": 0, "parses": 0, "chunks": 0, "embeddings": 0}
        )
        document_names: Dict[str, Set[str]] = defaultdict(set)

        def _collect_documents() -> None:
            for batch in _iter_batches(
                conn,
                "documents",
                warnings,
                columns=["collection", "source_path"],
                user_id=user_id,
                is_admin=is_admin,
            ):
                collection_idx = batch.schema.get_field_index("collection")
                source_idx = batch.schema.get_field_index("source_path")
                if collection_idx == -1:
                    continue
                collection_array = batch.column(collection_idx)
                source_array = (
                    batch.column(source_idx)
                    if source_idx != -1
                    else pa.array([None] * batch.num_rows)
                )
                for idx in range(batch.num_rows):
                    collection_raw = collection_array[idx].as_py()
                    if not collection_raw:
                        continue
                    collection_key = str(collection_raw)
                    stats[collection_key]["documents"] += 1
                    source_value = source_array[idx].as_py()
                    if source_value:
                        import os

                        document_names[collection_key].add(
                            os.path.basename(str(source_value))
                        )

        def _collect_simple(table_name: str, stat_key: str) -> None:
            for batch in _iter_batches(
                conn,
                table_name,
                warnings,
                columns=["collection"],
                user_id=user_id,
                is_admin=is_admin,
            ):
                collection_idx = batch.schema.get_field_index("collection")
                if collection_idx == -1:
                    continue
                collection_array = batch.column(collection_idx)
                for idx in range(batch.num_rows):
                    collection_raw = collection_array[idx].as_py()
                    if not collection_raw:
                        continue
                    collection_key = str(collection_raw)
                    stats[collection_key][stat_key] += 1

        _collect_documents()
        _collect_simple("parses", "parses")
        _collect_simple("chunks", "chunks")

        for table_name in _list_table_names(conn, warnings):
            if not table_name.startswith("embeddings_"):
                continue
            for batch in _iter_batches(
                conn,
                table_name,
                warnings,
                columns=["collection"],
                user_id=user_id,
                is_admin=is_admin,
            ):
                collection_idx = batch.schema.get_field_index("collection")
                if collection_idx == -1:
                    continue
                collection_array = batch.column(collection_idx)
                for idx in range(batch.num_rows):
                    collection_raw = collection_array[idx].as_py()
                    if not collection_raw:
                        continue
                    collection_key = str(collection_raw)
                    stats[collection_key]["embeddings"] += 1

        collection_keys = sorted(stats.keys() | document_names.keys())

        # Load configs for collections
        collection_configs = {}
        try:
            # TODO(refactor): this still reads per-user config from
            # collection_config for backward compatibility. Move to the unified
            # metadata/config store after migration semantics are defined.
            ensure_collection_config_table(conn)
            table = conn.open_table("collection_config")

            # Apply user filter if needed
            config_filter = UserPermissions.get_user_filter(user_id, is_admin)

            if config_filter:
                try:
                    df = table.search().where(config_filter).to_pandas()
                except Exception as e:
                    logger.warning(f"Failed to apply filter to collection_config: {e}")
                    df = table.to_pandas()
            else:
                df = table.to_pandas()

            for _, row in df.iterrows():
                col_name = row["collection"]
                config_json = row.get("config_json")
                if col_name and config_json:
                    import json

                    from ..core.schemas import IngestionConfig

                    try:
                        config_dict = json.loads(config_json)
                        collection_configs[col_name] = IngestionConfig(**config_dict)
                    except Exception as e:
                        logger.warning(
                            f"Failed to parse config for collection {col_name}: {e}"
                        )
        except Exception as e:
            logger.warning(f"Could not load collection configs: {e}")

        collections = [
            CollectionInfo(
                name=collection,
                documents=stats[collection]["documents"],
                parses=stats[collection]["parses"],
                chunks=stats[collection]["chunks"],
                embeddings=stats[collection]["embeddings"],
                processed_documents=stats[collection][
                    "parses"
                ],  # Use parses count as processed documents
                document_names=sorted(document_names[collection]),
                ingestion_config=collection_configs.get(collection),
            )
            for collection in collection_keys
        ]

        message = f"Found {len(collections)} collections"
        logger.info(message)
        return ListCollectionsResult(
            status="success",
            collections=collections,
            total_count=len(collections),
            message=message,
            warnings=warnings,
        )

    except Exception as exc:  # noqa: BLE001 - convert to structured failure
        logger.error("Failed to list collections: %s", exc, exc_info=True)
        return ListCollectionsResult(
            status="error",
            collections=[],
            total_count=0,
            message=f"Failed to list collections: {exc}",
            warnings=warnings,
        )


def get_document_stats(
    collection: str,
    doc_id: str,
    model_tag: Optional[str] = None,
    user_id: Optional[int] = None,
    is_admin: bool = False,
) -> DocumentStatsResult:
    """Return statistics for a single document within a collection.

    Args:
        collection: Target collection name.
        doc_id: Document identifier inside the collection.
        model_tag: Optional embedding model name; when provided the
            statistics only include the corresponding embeddings table.
        user_id: Optional user ID for filtering (for multi-tenancy).
        is_admin: Whether user has admin privileges.

    Returns:
        DocumentStatsResult: Structured statistics and warnings.
    """

    warnings: List[str] = []

    try:
        conn = get_connection_from_env()
        ensure_documents_table(conn)
        ensure_parses_table(conn)
        ensure_chunks_table(conn)
    except Exception as exc:  # noqa: BLE001 - convert to structured failure
        logger.error("Failed to initialise LanceDB tables: %s", exc, exc_info=True)
        return DocumentStatsResult(
            status="error",
            data=None,
            message=f"Failed to initialise LanceDB tables: {exc}",
            warnings=warnings,
        )

    ensure_ingestion_runs_table(conn)

    filters = {"collection": collection, "doc_id": doc_id}

    document_count = _count_rows(conn, "documents", filters, warnings)
    document_exists = document_count > 0
    parse_count = _count_rows(conn, "parses", filters, warnings)
    chunk_count = _count_rows(conn, "chunks", filters, warnings)

    embedding_breakdown: Dict[str, int] = {}

    def _count_embeddings(table_name: str) -> int:
        return _count_rows(conn, table_name, filters, warnings)

    if model_tag:
        table_name = embeddings_table_name(model_tag)
        embedding_count = _count_embeddings(table_name)
        embedding_breakdown[table_name] = embedding_count
    else:
        try:
            table_names = _list_table_names(conn, warnings)
        except Exception as exc:  # noqa: BLE001 - convert to warning
            message = f"Unable to enumerate embeddings tables: {exc}"
            logger.warning(message)
            warnings.append(message)
            table_names = []

        for table_name in table_names:
            if not table_name.startswith("embeddings_"):
                continue
            embedding_count = _count_embeddings(table_name)
            if embedding_count:
                embedding_breakdown[table_name] = embedding_count

        embedding_count = sum(embedding_breakdown.values())

    if model_tag:
        embedding_count = embedding_breakdown.get(embeddings_table_name(model_tag), 0)

    status_record = None
    status_entries = load_ingestion_status(collection=collection, doc_id=doc_id)
    if status_entries:
        status_record = status_entries[-1]

    status_value = DocumentProcessingStatus.PENDING
    last_message = None
    updated_at_ts = None
    if status_record:
        raw_status = status_record.get("status")
        if isinstance(raw_status, str):
            try:
                status_value = DocumentProcessingStatus(raw_status)
            except ValueError:
                logger.warning(
                    "Unknown ingestion status '%s' for %s/%s",
                    raw_status,
                    collection,
                    doc_id,
                )
        last_message = status_record.get("message") or None
        updated_at_ts = _coerce_timestamp(status_record.get("updated_at"))
    else:
        # If no status record exists, infer status from chunk and embedding counts
        if chunk_count > 0:
            if embedding_count == 0:
                status_value = DocumentProcessingStatus.CHUNKED
            elif embedding_count < chunk_count:
                status_value = DocumentProcessingStatus.PARTIALLY_EMBEDDED
            else:  # embedding_count >= chunk_count
                status_value = DocumentProcessingStatus.SUCCESS
        else:
            status_value = DocumentProcessingStatus.PENDING

    stats = DocumentStats(
        collection=collection,
        doc_id=doc_id,
        document_exists=document_exists,
        parse_count=parse_count,
        chunk_count=chunk_count,
        embedding_count=embedding_count,
        embedding_breakdown=embedding_breakdown,
        status=status_value,
        last_message=last_message,
        updated_at=updated_at_ts,
    )

    if document_exists:
        message = "Document statistics collected successfully"
    else:
        message = (
            f"Document '{doc_id}' not found in collection '{collection}'."
            " Counts reflect zero state."
        )

    return DocumentStatsResult(
        status="success",
        data=stats,
        message=message,
        warnings=warnings,
    )


def list_documents(
    collection: str,
    user_id: Optional[int] = None,
    is_admin: bool = False,
) -> DocumentListResult:
    """List documents for a collection including latest processing status.

    Uses batch processing to minimize memory footprint.

    Args:
        collection: Target collection name.
        user_id: Optional user ID for filtering (for multi-tenancy).
        is_admin: Whether user has admin privileges.

    Returns:
        DocumentListResult: List of documents with status information.
    """
    warnings: List[str] = []

    try:
        conn = get_connection_from_env()
        ensure_documents_table(conn)
        ensure_parses_table(conn)
        ensure_chunks_table(conn)
        ensure_ingestion_runs_table(conn)
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to initialise LanceDB tables: %s", exc, exc_info=True)
        return DocumentListResult(
            status="error",
            documents=[],
            total_count=0,
            message=f"Failed to initialise LanceDB tables: {exc}",
            warnings=warnings,
        )

    document_info: Dict[str, Dict[str, Any]] = {}
    for batch in _iter_batches(
        conn,
        "documents",
        warnings,
        columns=["collection", "doc_id", "source_path", "uploaded_at"],
        user_id=user_id,
        is_admin=is_admin,
    ):
        collection_idx = batch.schema.get_field_index("collection")
        doc_idx = batch.schema.get_field_index("doc_id")
        if collection_idx == -1 or doc_idx == -1:
            continue
        source_idx = batch.schema.get_field_index("source_path")
        uploaded_idx = batch.schema.get_field_index("uploaded_at")
        collection_array = batch.column(collection_idx)
        doc_array = batch.column(doc_idx)
        for idx in range(batch.num_rows):
            collection_raw = collection_array[idx].as_py()
            if not collection_raw or str(collection_raw) != collection:
                continue
            doc_raw = doc_array[idx].as_py()
            if not doc_raw:
                continue
            info: Dict[str, Any] = {}
            if source_idx != -1:
                info["source_path"] = batch.column(source_idx)[idx].as_py()
            if uploaded_idx != -1:
                info["uploaded_at"] = batch.column(uploaded_idx)[idx].as_py()
            document_info[str(doc_raw)] = info

    chunk_counts = _collect_doc_counts_for_collection(
        conn, "chunks", "doc_id", collection, warnings, user_id, is_admin
    )

    embedding_counts: Dict[str, int] = defaultdict(int)
    for table_name in _list_table_names(conn, warnings):
        if not table_name.startswith("embeddings_"):
            continue
        table_counts = _collect_doc_counts_for_collection(
            conn, table_name, "doc_id", collection, warnings, user_id, is_admin
        )
        for doc_id, value in table_counts.items():
            embedding_counts[doc_id] += value

    status_records = {
        entry["doc_id"]: entry for entry in load_ingestion_status(collection=collection)
    }

    doc_ids = (
        set(document_info.keys())
        | set(chunk_counts.keys())
        | set(embedding_counts.keys())
        | set(status_records.keys())
    )

    summaries: List[DocumentSummary] = []
    for doc_id in sorted(doc_ids):
        info = document_info.get(doc_id, {})
        status_entry = status_records.get(doc_id)
        chunk_count = chunk_counts.get(doc_id, 0)
        embedding_count = embedding_counts.get(doc_id, 0)

        if status_entry and isinstance(status_entry.get("status"), str):
            try:
                status_value = DocumentProcessingStatus(status_entry["status"])
            except ValueError:
                status_value = DocumentProcessingStatus.PENDING
        elif chunk_count > 0:
            if embedding_count == 0:
                status_value = DocumentProcessingStatus.CHUNKED
            elif embedding_count < chunk_count:
                status_value = DocumentProcessingStatus.PARTIALLY_EMBEDDED
            else:  # embedding_count >= chunk_count
                status_value = DocumentProcessingStatus.SUCCESS
        else:
            status_value = DocumentProcessingStatus.PENDING

        updated_at = _coerce_timestamp(
            status_entry.get("updated_at") if status_entry else None
        )
        message = status_entry.get("message") if status_entry else None

        summaries.append(
            DocumentSummary(
                collection=collection,
                doc_id=doc_id,
                source_path=info.get("source_path"),
                status=status_value,
                message=message or None,
                created_at=_coerce_timestamp(info.get("uploaded_at")),
                updated_at=updated_at or _coerce_timestamp(info.get("uploaded_at")),
                chunk_count=chunk_count,
                embedding_count=embedding_count,
            )
        )

    message = f"Retrieved {len(summaries)} documents for collection '{collection}'."
    return DocumentListResult(
        status="success",
        documents=summaries,
        total_count=len(summaries),
        message=message,
        warnings=warnings,
    )


def delete_collection(
    collection: str,
    user_id: Optional[int] = None,
    is_admin: bool = False,
) -> CollectionOperationResult:
    """Delete all documents and associated artifacts for a collection.

    This function will delete:
    - All documents in the collection
    - All parsing data for the collection
    - All chunks for the collection
    - All embeddings for the collection

    Args:
        collection: Name of the collection to delete
        user_id: Optional user ID for permission check (for multi-tenancy).
        is_admin: Whether user has admin privileges.

    Returns:
        CollectionOperationResult indicating success or failure
    """

    warnings: List[str] = []

    try:
        conn = get_connection_from_env()
        ensure_documents_table(conn)
        ensure_parses_table(conn)
        ensure_chunks_table(conn)
        ensure_ingestion_runs_table(conn)
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "Failed to initialise LanceDB tables for delete_collection: %s",
            exc,
            exc_info=True,
        )
        return CollectionOperationResult(
            status="error",
            collection=collection,
            message=f"Failed to initialise LanceDB tables: {exc}",
            warnings=warnings,
            affected_documents=[],
            deleted_counts={},
        )

    # Collect doc_ids before deletion for affected_documents
    doc_ids = sorted(
        _collect_document_ids(conn, collection, warnings, user_id, is_admin)
    )

    # Delete all data using direct table.delete() with escaped collection name
    deleted_counts: Dict[str, int] = defaultdict(int)
    table_names = _list_table_names(conn, warnings)

    # Delete from core tables
    for table_name in ["documents", "parses", "chunks"]:
        if table_name in table_names:
            try:
                table = conn.open_table(table_name)
                original_count = table.count_rows()
                # Delete all rows for this collection using escaped string
                table.delete(f"collection = '{escape_lancedb_string(collection)}'")
                deleted_count = original_count - table.count_rows()
                if deleted_count > 0:
                    deleted_counts[table_name] = deleted_count
            except Exception as exc:  # noqa: BLE001
                warning = f"Failed to delete from '{table_name}': {exc}"
                logger.warning(warning)
                warnings.append(warning)

    # Delete embeddings data
    embeddings_tables = [t for t in table_names if t.startswith("embeddings_")]
    for table_name in embeddings_tables:
        try:
            table = conn.open_table(table_name)
            original_count = table.count_rows()
            # Delete all rows for this collection using escaped string
            table.delete(f"collection = '{escape_lancedb_string(collection)}'")
            deleted_count = original_count - table.count_rows()
            if deleted_count > 0:
                deleted_counts[table_name] = deleted_count
        except Exception as exc:  # noqa: BLE001
            warning = f"Failed to delete from '{table_name}': {exc}"
            logger.warning(warning)
            warnings.append(warning)

    # Clear ingestion status for all documents
    for doc_id in doc_ids:
        try:
            clear_ingestion_status(collection, doc_id)
        except Exception as exc:  # noqa: BLE001
            warning = f"Failed to clear ingestion status for '{doc_id}': {exc}"
            logger.warning(warning)
            warnings.append(warning)

    # Construct affected_documents list
    affected: List[CollectionOperationDetail] = [
        CollectionOperationDetail(
            doc_id=doc_id,
            status=DocumentProcessingStatus.FAILED,
            message="Document deleted successfully.",
        )
        for doc_id in doc_ids
    ]

    if not doc_ids and not deleted_counts:
        summary = f"No documents found in collection '{collection}'."
        return CollectionOperationResult(
            status="success",
            collection=collection,
            message=summary,
            warnings=warnings,
            affected_documents=[],
            deleted_counts={},
        )

    if affected and not warnings:
        status = "success"
    elif affected:
        status = "partial_success"
    else:
        status = "error"

    summary = f"Deleted {len(affected)} documents from collection '{collection}'."
    logger.info(
        f"Deleted collection '{collection}' - {sum(deleted_counts.values())} total rows across {len(deleted_counts)} tables"
    )
    return CollectionOperationResult(
        status=status,
        collection=collection,
        message=summary,
        warnings=warnings,
        affected_documents=affected,
        deleted_counts=dict(deleted_counts),
    )


def delete_document(
    collection: str, doc_id: str, user_id: int, is_admin: bool = False
) -> DocumentOperationResult:
    """Delete a document and all its associated data.

    This performs a cascade delete of the document's parses, chunks,
    embeddings, and related data.

    Args:
        collection: Collection name.
        doc_id: Document identifier.

    Returns:
        DocumentOperationResult: Operation result with deletion counts.
    """
    try:
        # Use cascade cleanup to delete all related data
        counts = cleanup_document_cascade(
            collection=collection,
            doc_id=doc_id,
            preview_only=False,
            confirm=True,
        )
        clear_ingestion_status(collection, doc_id)
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to delete document %s/%s: %s", collection, doc_id, exc)
        return DocumentOperationResult(
            status="error",
            collection=collection,
            doc_id=doc_id,
            new_status=DocumentProcessingStatus.FAILED,
            message=f"Failed to delete document: {exc}",
            warnings=[],
            details={},
        )

    return DocumentOperationResult(
        status="success",
        collection=collection,
        doc_id=doc_id,
        new_status=DocumentProcessingStatus.FAILED,
        message="Document deleted successfully.",
        warnings=[],
        details=counts,
    )


def retry_document(
    collection: str, doc_id: str, user_id: int, is_admin: bool = False
) -> DocumentOperationResult:
    """Mark a document for retry by resetting its status to pending."""

    try:
        write_ingestion_status(
            collection,
            doc_id,
            status=DocumentProcessingStatus.PENDING.value,
            message="Retry requested.",
            parse_hash="",
        )
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "Failed to mark document %s/%s for retry: %s", collection, doc_id, exc
        )
        return DocumentOperationResult(
            status="error",
            collection=collection,
            doc_id=doc_id,
            new_status=DocumentProcessingStatus.PENDING,
            message=f"Failed to mark document for retry: {exc}",
            warnings=[],
            details={},
        )
    return DocumentOperationResult(
        status="success",
        collection=collection,
        doc_id=doc_id,
        new_status=DocumentProcessingStatus.PENDING,
        message="Document marked for retry.",
        warnings=[],
        details={},
    )


def cancel_document(
    collection: str,
    doc_id: str,
    user_id: int,
    is_admin: bool = False,
    reason: Optional[str] = None,
) -> DocumentOperationResult:
    """Mark a document ingestion process as cancelled."""

    message = reason or "Cancelled by user."
    try:
        write_ingestion_status(
            collection,
            doc_id,
            status=DocumentProcessingStatus.FAILED.value,
            message=message,
            parse_hash="",
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to cancel document %s/%s: %s", collection, doc_id, exc)
        return DocumentOperationResult(
            status="error",
            collection=collection,
            doc_id=doc_id,
            new_status=DocumentProcessingStatus.FAILED,
            message=f"Failed to cancel document: {exc}",
            warnings=[],
            details={},
        )
    return DocumentOperationResult(
        status="success",
        collection=collection,
        doc_id=doc_id,
        new_status=DocumentProcessingStatus.FAILED,
        message="Document ingestion cancelled.",
        warnings=[],
        details={},
    )


def cancel_collection(
    collection: str,
    reason: Optional[str] = None,
    user_id: Optional[int] = None,
    is_admin: bool = False,
) -> CollectionOperationResult:
    """Mark all documents in a collection as cancelled."""

    warnings: List[str] = []

    try:
        conn = get_connection_from_env()
        ensure_documents_table(conn)
        ensure_parses_table(conn)
        ensure_chunks_table(conn)
        ensure_ingestion_runs_table(conn)
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "Failed to initialise LanceDB tables for cancel_collection: %s",
            exc,
            exc_info=True,
        )
        return CollectionOperationResult(
            status="error",
            collection=collection,
            message=f"Failed to initialise LanceDB tables: {exc}",
            warnings=warnings,
            affected_documents=[],
            deleted_counts={},
        )

    doc_ids = sorted(
        _collect_document_ids(conn, collection, warnings, user_id, is_admin)
    )
    cancellation_message = reason or "Cancelled by user."
    affected: List[CollectionOperationDetail] = []

    for doc_id in doc_ids:
        try:
            write_ingestion_status(
                collection=collection,
                doc_id=doc_id,
                status=DocumentProcessingStatus.FAILED.value,
                message=cancellation_message,
                parse_hash="",
            )
            affected.append(
                CollectionOperationDetail(
                    doc_id=doc_id,
                    status=DocumentProcessingStatus.FAILED,
                    message=cancellation_message,
                )
            )
        except Exception as exc:  # noqa: BLE001
            warning = f"Failed to cancel document '{doc_id}': {exc}"
            logger.error(warning, exc_info=True)
            warnings.append(warning)

    if not doc_ids:
        summary = f"No documents found in collection '{collection}'."
        return CollectionOperationResult(
            status="success",
            collection=collection,
            message=summary,
            warnings=warnings,
            affected_documents=[],
            deleted_counts={},
        )

    if affected and not warnings:
        status = "success"
    elif affected:
        status = "partial_success"
    else:
        status = "error"

    summary = f"Cancelled {len(affected)} documents in collection '{collection}'."
    return CollectionOperationResult(
        status=status,
        collection=collection,
        message=summary,
        warnings=warnings,
        affected_documents=affected,
        deleted_counts={},
    )


def get_document_status(collection: str, doc_id: str) -> Dict[str, Any]:
    """Get the current ingestion status for a document.

    Args:
        collection: Collection name.
        doc_id: Document identifier.

    Returns:
        Dictionary with status information, or empty dict if not found.
    """
    try:
        status_records = load_ingestion_status(collection=collection, doc_id=doc_id)
        if status_records:
            return status_records[0]
        return {}
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to get document status: %s", exc)
        return {}
