"""Main pointer management for version control.

This module provides functionality for managing main version pointers
across different processing stages (parse, chunk, embed).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import pandas as pd

from ......providers.vector_store.lancedb import get_connection_from_env
from ..core.exceptions import MainPointerError
from ..LanceDB.schema_manager import ensure_main_pointers_table
from ..utils.string_utils import build_lancedb_filter_expression, escape_lancedb_string

logger = logging.getLogger(__name__)


def _normalize_model_tag(model_tag: Optional[str]) -> str:
    """Normalize model_tag to empty string if None."""
    return model_tag if model_tag is not None else ""


def _build_base_filter_expression(collection: str, doc_id: str, step_type: str) -> str:
    """Build the base LanceDB filter expression for a main pointer row.

    This helper escapes all string values to avoid malformed expressions and
    injection-like issues.

    Args:
        collection: Collection name.
        doc_id: Document ID.
        step_type: Processing stage type (parse, chunk, embed).

    Returns:
        A filter expression covering collection/doc_id/step_type.
    """
    return (
        f"collection == '{escape_lancedb_string(collection)}' AND "
        f"doc_id == '{escape_lancedb_string(doc_id)}' AND "
        f"step_type == '{escape_lancedb_string(step_type)}'"
    )


def get_main_pointer(
    collection: str, doc_id: str, step_type: str, model_tag: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """Get the main pointer for a specific document and stage.

    Args:
        collection: Collection name
        doc_id: Document ID
        step_type: Processing stage type (parse, chunk, embed)
        model_tag: Model tag for embed stage (optional)

    Returns:
        Main pointer data or None if not found

    Raises:
        MainPointerError: If there's an error retrieving the pointer
    """
    try:
        conn = get_connection_from_env()
        ensure_main_pointers_table(conn)

        table = conn.open_table("main_pointers")

        # Build safe filter conditions
        normalized_tag = _normalize_model_tag(model_tag)

        # Base filters for collection, doc_id, and step_type
        base_expr = _build_base_filter_expression(collection, doc_id, step_type)

        # Handle model_tag: check for both normalized empty string AND NULL for backward compatibility
        if normalized_tag == "":
            filter_expr = f"{base_expr} AND (model_tag == '' OR model_tag IS NULL)"
        else:
            filter_expr = f"{base_expr} AND model_tag == '{escape_lancedb_string(normalized_tag)}'"

        # Query the table
        result = table.search().where(filter_expr).to_pandas()

        if result.empty:
            return None

        # Return the first result, preferring non-NULL model_tag if multiple found
        if len(result) > 1:
            result = result.sort_values("model_tag", ascending=False)

        row = result.iloc[0]
        return {
            "collection": row["collection"],
            "doc_id": row["doc_id"],
            "step_type": row["step_type"],
            "model_tag": row["model_tag"] if row["model_tag"] is not None else "",
            "semantic_id": row["semantic_id"],
            "technical_id": row["technical_id"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "operator": row["operator"],
        }

    except Exception as e:
        raise MainPointerError(f"Failed to get main pointer: {e}")


def set_main_pointer(
    lancedb_dir: str,  # Kept for backward compatibility
    collection: str,
    doc_id: str,
    step_type: str,
    semantic_id: str,
    technical_id: str,
    model_tag: Optional[str] = None,
    operator: Optional[str] = None,
) -> None:
    """Set or update the main pointer for a specific document and stage.

    Uses merge_insert for atomicity and avoids 'delete-then-add' race conditions.
    Normalizes None model_tag to empty string.

    Args:
        lancedb_dir: Directory for LanceDB (unused, using connection from env)
        collection: Collection name
        doc_id: Document ID
        step_type: Processing stage type (parse, chunk, embed)
        semantic_id: Semantic identifier for the version
        technical_id: Technical identifier (hash) for the version
        model_tag: Model tag for embed stage (optional)
        operator: Operator who made the change (optional)

    Raises:
        MainPointerError: If there's an error setting the pointer
    """
    try:
        conn = get_connection_from_env()
        ensure_main_pointers_table(conn)

        table = conn.open_table("main_pointers")
        normalized_tag = _normalize_model_tag(model_tag)
        now = pd.Timestamp.now(tz="UTC")

        # Check if pointer already exists to preserve created_at
        existing = get_main_pointer(collection, doc_id, step_type, model_tag)

        if existing:
            created_at = existing["created_at"]

            # Fix-up: normalize NULL model_tag to "" in DB before merge_insert to avoid duplicates
            if normalized_tag == "":
                base_expr = _build_base_filter_expression(collection, doc_id, step_type)
                null_filter = f"{base_expr} AND model_tag IS NULL"
                try:
                    table.update(where=null_filter, values={"model_tag": ""})
                except Exception as update_err:
                    logger.warning("Failed to normalize NULL model_tag: %s", update_err)
        else:
            created_at = now

        # Prepare data for merge_insert
        update_data = {
            "collection": [collection],
            "doc_id": [doc_id],
            "step_type": [step_type],
            "model_tag": [normalized_tag],
            "semantic_id": [semantic_id],
            "technical_id": [technical_id],
            "created_at": [created_at],
            "updated_at": [now],
            "operator": [operator or "unknown"],
        }
        df = pd.DataFrame(update_data)

        (
            table.merge_insert(on=["collection", "doc_id", "step_type", "model_tag"])
            .when_matched_update_all()
            .when_not_matched_insert_all()
            .execute(df)
        )

        logger.info(
            f"Set main pointer for {collection}/{doc_id}/{step_type} to {technical_id} (semantic: {semantic_id})"
        )

    except Exception as e:
        raise MainPointerError(f"Failed to set main pointer: {e}")


def list_main_pointers(
    collection: str, doc_id: Optional[str] = None
) -> List[Dict[str, Any]]:
    """List main pointers for a collection and optionally a specific document.

    Args:
        collection: Collection name
        doc_id: Document ID (optional, if None lists all documents in collection)

    Returns:
        List of main pointer data

    Raises:
        MainPointerError: If there's an error listing pointers
    """
    try:
        conn = get_connection_from_env()
        ensure_main_pointers_table(conn)

        table = conn.open_table("main_pointers")

        # Build safe filter conditions
        filters_dict = {"collection": collection}
        if doc_id is not None:
            filters_dict["doc_id"] = doc_id

        filter_expr = build_lancedb_filter_expression(filters_dict)

        # First check if any pointers exist using efficient count_rows
        if table.search().where(filter_expr).count_rows() == 0:
            return []

        # Only load data if pointers exist
        result = table.search().where(filter_expr).to_pandas()

        # Convert to list of dictionaries
        pointers = []
        for _, row in result.iterrows():
            pointers.append(
                {
                    "collection": row["collection"],
                    "doc_id": row["doc_id"],
                    "step_type": row["step_type"],
                    "model_tag": row["model_tag"]
                    if row["model_tag"] is not None
                    else "",
                    "semantic_id": row["semantic_id"],
                    "technical_id": row["technical_id"],
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"],
                    "operator": row["operator"],
                }
            )

        return pointers

    except Exception as e:
        raise MainPointerError(f"Failed to list main pointers: {e}")


def delete_main_pointer(
    collection: str, doc_id: str, step_type: str, model_tag: Optional[str] = None
) -> bool:
    """Delete a main pointer.

    Behavior note (backward compatibility):
        When ``model_tag`` is ``None`` (normalized to empty string), this function deletes
        pointers whose ``model_tag`` is either ``''`` OR ``NULL``.

    Args:
        collection: Collection name
        doc_id: Document ID
        step_type: Processing stage type
        model_tag: Model tag for embed stage (optional)

    Returns:
        True if pointer was deleted, False if not found

    Raises:
        MainPointerError: If there's an error deleting the pointer
    """
    try:
        conn = get_connection_from_env()
        ensure_main_pointers_table(conn)

        table = conn.open_table("main_pointers")

        # Build safe filter conditions
        normalized_tag = _normalize_model_tag(model_tag)
        base_expr = _build_base_filter_expression(collection, doc_id, step_type)

        if normalized_tag == "":
            filter_expr = f"{base_expr} AND (model_tag == '' OR model_tag IS NULL)"
        else:
            filter_expr = f"{base_expr} AND model_tag == '{escape_lancedb_string(normalized_tag)}'"

        # Check if pointer exists using count_rows for efficiency
        count = table.search().where(filter_expr).count_rows()
        if count == 0:
            return False

        # Delete the pointer(s)
        table.delete(filter_expr)
        logger.info(f"Deleted main pointer for {collection}/{doc_id}/{step_type}")
        return True

    except Exception as e:
        raise MainPointerError(f"Failed to delete main pointer: {e}")
