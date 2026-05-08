"""LanceDB vector-plane cascade preview and delete (adapter-internal).

Predicates are LanceDB filter strings keyed by logical table name, including
``__embeddings__`` fan-out. Version-management code builds predicates; this
module performs count/delete only against a LanceDB connection.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from ..LanceDB.schema_manager import _safe_close_table
from ..utils.lancedb_query_utils import _safe_count_rows

logger = logging.getLogger(__name__)


def get_table_names(conn: Any) -> list[str]:
    """Return table names from a LanceDB connection (defensive)."""
    table_names_fn = getattr(conn, "table_names", None)
    if table_names_fn is None:
        return []
    try:
        names = table_names_fn()
    except Exception:
        return []
    if not names:
        return []
    return [str(name) for name in names]


def plan_by_predicates(
    conn: Any, table_to_filter: Dict[str, str], model_tag: Optional[str] = None
) -> Dict[str, int]:
    """Count rows that match each table predicate without deleting."""
    counts: Dict[str, int] = {}
    table_names = get_table_names(conn)

    for t in table_names:
        if t.startswith("embeddings_") and t in table_to_filter:
            table = None
            try:
                table = conn.open_table(t)
                counts[t] = _safe_count_rows(table, table_to_filter[t])
            finally:
                _safe_close_table(table)

    for table_name, filt in table_to_filter.items():
        if table_name == "__embeddings__":
            total = 0
            all_embed_tables = [t for t in table_names if t.startswith("embeddings_")]
            if model_tag:
                all_embed_tables = [
                    t for t in all_embed_tables if t == f"embeddings_{model_tag}"
                ]
            for t in all_embed_tables:
                table = None
                try:
                    table = conn.open_table(t)
                    count = _safe_count_rows(table, filt)
                    total += count
                finally:
                    _safe_close_table(table)
            counts[table_name] = total
            continue

        if table_name not in table_names:
            counts[table_name] = 0
            continue
        table = None
        try:
            table = conn.open_table(table_name)
            count = _safe_count_rows(table, filt)
            counts[table_name] = count
        finally:
            _safe_close_table(table)
    return counts


def delete_by_predicates(
    conn: Any, table_to_filter: Dict[str, str], model_tag: Optional[str] = None
) -> Dict[str, int]:
    """Delete rows by table predicates in a fixed, safe order."""
    deleted: Dict[str, int] = {}
    table_names = get_table_names(conn)

    for t in table_names:
        if not t.startswith("embeddings_") or t not in table_to_filter:
            continue
        filt = table_to_filter[t]
        table = None
        try:
            table = conn.open_table(t)
            cnt = _safe_count_rows(table, filt)
            if cnt > 0:
                table.delete(filt)
                logger.info("Cascade cleanup: deleted %s rows from %s", cnt, t)
            deleted[t] = cnt
        finally:
            _safe_close_table(table)

    order = [
        "__embeddings__",
        "chunks",
        "parses",
        "main_pointers",
        "ingestion_runs",
        "documents",
    ]

    if "__embeddings__" in table_to_filter:
        filt = table_to_filter["__embeddings__"]
        total = 0

        all_embed_tables = [t for t in table_names if t.startswith("embeddings_")]
        if model_tag is not None:
            target_tables = [
                t for t in all_embed_tables if t == f"embeddings_{model_tag}"
            ]
        else:
            target_tables = all_embed_tables

        for t in target_tables:
            table = None
            try:
                table = conn.open_table(t)
                cnt = _safe_count_rows(table, filt)
                if cnt > 0:
                    table.delete(filt)
                total += cnt
            finally:
                _safe_close_table(table)
        deleted["embeddings"] = total
        if total > 0:
            logger.info(
                "Cascade cleanup: deleted %s rows from embeddings tables", total
            )

    for name in order[1:]:
        if name in table_to_filter and name in table_names:
            filt = table_to_filter[name]
            table = None
            try:
                table = conn.open_table(name)
                cnt = _safe_count_rows(table, filt)
                if cnt > 0:
                    table.delete(filt)
                    logger.info("Cascade cleanup: deleted %s rows from %s", cnt, name)
                deleted[name] = cnt
            finally:
                _safe_close_table(table)

    for name, filt in table_to_filter.items():
        if name in (
            "__embeddings__",
            "chunks",
            "parses",
            "main_pointers",
            "ingestion_runs",
            "documents",
        ) or name.startswith("embeddings_"):
            continue
        if name not in table_names:
            deleted[name] = 0
            continue
        table = None
        try:
            table = conn.open_table(name)
            cnt = _safe_count_rows(table, filt)
            if cnt > 0:
                table.delete(filt)
                logger.info("Cascade cleanup: deleted %s rows from %s", cnt, name)
            deleted[name] = cnt
        finally:
            _safe_close_table(table)

    return deleted
