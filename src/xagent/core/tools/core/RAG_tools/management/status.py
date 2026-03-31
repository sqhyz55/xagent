"""Helpers for tracking document ingestion status in LanceDB.

This module provides functions to track, load, and manage the ingestion status
of documents being processed in the RAG pipeline.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd

from ..LanceDB.schema_manager import ensure_ingestion_runs_table
from ..storage.factory import get_metadata_store
from ..utils.string_utils import build_lancedb_filter_expression
from ..utils.user_permissions import UserPermissions

logger = logging.getLogger(__name__)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def write_ingestion_status(
    collection: str,
    doc_id: str,
    *,
    status: str,
    message: Optional[str] = None,
    parse_hash: Optional[str] = None,
    user_id: Optional[int] = None,
) -> None:
    """Persist the latest ingestion status for a document.

    This function writes the current status of a document's ingestion process
    to the LanceDB ingestion_runs table.

    Args:
        collection: Name of the collection
        doc_id: Unique identifier for the document
        status: Current status value (e.g., 'pending', 'processing', 'success', 'failed')
        message: Optional status message or error description
        parse_hash: Optional hash of the parsed document for change detection
        user_id: Optional user ID for multi-tenancy support

    Returns:
        None
    """

    conn = get_metadata_store().get_raw_connection()
    ensure_ingestion_runs_table(conn)
    table = conn.open_table("ingestion_runs")

    # Build filter without user permissions (deleting all matching records)
    filter_expr = build_lancedb_filter_expression(
        {"collection": collection, "doc_id": doc_id}, skip_user_filter=True
    )
    if filter_expr:
        table.delete(filter_expr)

    timestamp = _now()
    record = {
        "collection": collection,
        "doc_id": doc_id,
        "status": status,
        "message": message or "",
        "parse_hash": parse_hash or "",
        "created_at": timestamp,
        "updated_at": timestamp,
        "user_id": user_id,  # Add user_id for multi-tenancy
    }
    table.add([record])


def load_ingestion_status(
    collection: Optional[str] = None,
    doc_id: Optional[str] = None,
    user_id: Optional[int] = None,
    is_admin: bool = False,
) -> List[Dict[str, Any]]:
    """Return ingestion status records filtered by collection/doc.

    This function retrieves ingestion status records from the LanceDB
    ingestion_runs table, with optional filtering by collection and document.

    Args:
        collection: Optional collection name to filter by
        doc_id: Optional document ID to filter by
        user_id: Optional user ID for multi-tenancy filtering
        is_admin: Whether the user has admin privileges (bypasses filtering)

    Returns:
        List of dictionaries containing ingestion status records with keys:
        - collection: Collection name
        - doc_id: Document identifier
        - status: Current status
        - message: Status message if any
        - parse_hash: Parse hash if any
        - created_at: Creation timestamp
        - updated_at: Last update timestamp
        - user_id: User ID who owns the document
    """

    conn = get_metadata_store().get_raw_connection()
    ensure_ingestion_runs_table(conn)
    table = conn.open_table("ingestion_runs")

    filters: Dict[str, str] = {}
    if collection is not None:
        filters["collection"] = collection
    if doc_id is not None:
        filters["doc_id"] = doc_id

    # Build base filter without user permissions (will be added separately)
    base_filter = build_lancedb_filter_expression(filters, skip_user_filter=True)
    user_filter = UserPermissions.get_user_filter(user_id, is_admin)

    if user_filter and base_filter:
        filter_expr = f"({base_filter}) and ({user_filter})"
    elif user_filter:
        filter_expr = user_filter
    else:
        filter_expr = base_filter
    try:
        search = table.search()
        if filter_expr:
            search = search.where(filter_expr)
        df = search.to_pandas()
    except Exception as e:
        logger.error(f"Failed to load ingestion status: {e}")
        df = pd.DataFrame()

    records: List[Dict[str, Any]] = df.to_dict("records")
    return records


def clear_ingestion_status(
    collection: str, doc_id: str, user_id: Optional[int] = None, is_admin: bool = False
) -> None:
    """Remove stored ingestion status for a document.

    This function deletes the ingestion status record for a specific document
    from the LanceDB ingestion_runs table.

    Args:
        collection: Name of the collection
        doc_id: Unique identifier for the document
        user_id: Optional user ID for multi-tenancy filtering
        is_admin: Whether the user has admin privileges (bypasses filtering)

    Returns:
        None
    """

    conn = get_metadata_store().get_raw_connection()
    ensure_ingestion_runs_table(conn)
    table = conn.open_table("ingestion_runs")

    # Build base filter without user permissions (will be added separately)
    base_filter = build_lancedb_filter_expression(
        {"collection": collection, "doc_id": doc_id}, skip_user_filter=True
    )
    user_filter = UserPermissions.get_user_filter(user_id, is_admin)

    if user_filter and base_filter:
        filter_expr = f"({base_filter}) and ({user_filter})"
    elif user_filter:
        filter_expr = user_filter
    else:
        filter_expr = base_filter

    if filter_expr:
        table.delete(filter_expr)
