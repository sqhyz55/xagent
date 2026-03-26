"""LanceDB migration: Backfill user_id for chunks and embeddings tables.

This migration script backfills the user_id field in chunks and embeddings tables
by joining with the documents table. This is necessary for multi-tenancy data isolation.

Uses two-phase migration:
- Phase 1: Normal backfill, mark orphaned records with user_id = -1
- Phase 2: Retry orphaned records in case their parent documents were created concurrently
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import threading
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from lancedb.db import DBConnection

# Add parent directories to path for imports
# This must be done before importing project modules
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import after path modification (required for standalone migration scripts)
# ruff: noqa: E402
from xagent.core.tools.core.RAG_tools.LanceDB.schema_manager import (
    ensure_chunks_table,
    ensure_documents_table,
)
from xagent.core.tools.core.RAG_tools.utils.lancedb_query_utils import query_to_list
from xagent.providers.vector_store.lancedb import get_connection_from_env

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Batch size for processing records to avoid memory issues
BATCH_SIZE = 10000

# Orphaned record markers
# These are used as user_id values to mark records that couldn't be matched to a document.
# Since user_id should normally be >= 0, negative values serve as special markers.
ORPHANED_TEMPORARY = (
    -1
)  # Phase 1: Temporary orphan (may be due to concurrent document creation)
ORPHANED_PERMANENT = (
    -2
)  # Phase 2: Permanent orphan (confirmed no matching document exists)

# Global lock to prevent concurrent migrations
_migration_lock = threading.Lock()


def _backfill_table_core(
    table: Any,
    docs_table: Any,
    query_filter: str,
    filter_fields: list[str],
    failure_user_id: int,
    dry_run: bool,
    log_prefix: str = "",
) -> dict:
    """Core logic for backfilling a single table.

    Args:
        table: LanceDB table to backfill
        docs_table: Documents table for lookup
        query_filter: Filter to find records needing backfill (e.g., "user_id IS NULL")
        filter_fields: Fields used to identify a specific record for update
        failure_user_id: user_id to set if document lookup fails (e.g., -1 or -2)
        dry_run: If True, don't make actual changes
        log_prefix: Prefix for log messages

    Returns:
        Dictionary with statistics
    """
    total_backfilled = 0
    total_skipped = 0
    total_failed = 0
    batch_number = 0

    while True:
        # Get a batch of records matching the filter
        batch = query_to_list(table.search().where(query_filter).limit(BATCH_SIZE))

        if not batch:
            break

        batch_number += 1
        logger.info(
            f"{log_prefix} Processing batch #{batch_number}: {len(batch)} records..."
        )

        # Build doc_id -> user_id mapping from documents table
        doc_user_map = {}
        all_doc_ids = [
            doc_id for doc_id in set(r.get("doc_id") for r in batch) if doc_id
        ]

        if all_doc_ids:
            # Bulk lookup for documents
            doc_ids_str = ", ".join([f"'{d}'" for d in all_doc_ids])
            docs = query_to_list(
                docs_table.search()
                .where(f"doc_id IN ({doc_ids_str})")
                .limit(len(all_doc_ids))
            )
            for doc in docs:
                if doc.get("user_id") is not None:
                    doc_user_map[doc.get("doc_id")] = doc.get("user_id")

        logger.info(
            f"{log_prefix} Batch #{batch_number}: Found user_id for {len(doc_user_map)} / {len(all_doc_ids)} documents"
        )

        # Update records
        skipped = 0
        for record in batch:
            doc_id = record.get("doc_id")

            if doc_id in doc_user_map:
                user_id = doc_user_map[doc_id]
                is_recovered = True
            else:
                user_id = failure_user_id
                is_recovered = False
                skipped += 1
                total_skipped += 1

            if not dry_run:
                try:
                    # Build update filter
                    update_filter = " and ".join(
                        [f"{f} = '{record.get(f)}'" for f in filter_fields]
                    )
                    table.update(update_filter, {"user_id": user_id})

                    if is_recovered:
                        total_backfilled += 1
                except Exception as e:
                    total_failed += 1
                    logger.warning(f"{log_prefix} Failed to update record: {e}")
            else:
                if is_recovered:
                    total_backfilled += 1

        logger.info(
            f"{log_prefix} Batch #{batch_number}: {len(batch) - skipped} processed, {skipped} marked as failure_id ({failure_user_id})"
        )

    return {
        "total": total_backfilled + total_skipped + total_failed,
        "backfilled": total_backfilled,
        "skipped": total_skipped,
        "failed": total_failed,
    }


def backfill_chunks_table(
    dry_run: bool = False, conn: DBConnection | None = None
) -> dict:
    """Backfill user_id for chunks table (Phase 1)."""
    if conn is None:
        conn = get_connection_from_env()

    ensure_chunks_table(conn)
    ensure_documents_table(conn)

    chunks_table = conn.open_table("chunks")
    docs_table = conn.open_table("documents")

    logger.info("Phase 1: Starting chunks table user_id backfill...")
    result = _backfill_table_core(
        table=chunks_table,
        docs_table=docs_table,
        query_filter="user_id IS NULL",
        filter_fields=["doc_id", "chunk_id", "parse_hash"],
        failure_user_id=ORPHANED_TEMPORARY,
        dry_run=dry_run,
        log_prefix="Chunks Phase 1:",
    )
    result["table"] = "chunks"
    return result


def backfill_orphaned_chunks(
    dry_run: bool = False, conn: DBConnection | None = None
) -> dict:
    """Retry backfill for orphaned chunks (Phase 2)."""
    if conn is None:
        conn = get_connection_from_env()

    chunks_table = conn.open_table("chunks")
    docs_table = conn.open_table("documents")

    logger.info("Phase 2: Retrying orphaned chunks (user_id = -1)...")
    result = _backfill_table_core(
        table=chunks_table,
        docs_table=docs_table,
        query_filter=f"user_id = {ORPHANED_TEMPORARY}",
        filter_fields=["doc_id", "chunk_id", "parse_hash"],
        failure_user_id=ORPHANED_PERMANENT,
        dry_run=dry_run,
        log_prefix="Chunks Phase 2:",
    )
    result["table"] = "chunks"
    return result


def _get_embeddings_tables(conn: DBConnection) -> list[str]:
    """Helper to get all embeddings tables with API compatibility."""
    list_tables_fn = getattr(conn, "list_tables", None)
    if list_tables_fn is None:
        list_tables_fn = getattr(conn, "table_names", None)

    if list_tables_fn is None:
        return []

    try:
        tables_res = list_tables_fn()
        if hasattr(tables_res, "tables"):
            table_names = tables_res.tables
        else:
            table_names = list(tables_res)
        return [t for t in table_names if t.startswith("embeddings_")]
    except Exception as e:
        logger.warning(f"Failed to list LanceDB tables: {e}")
        return []


def backfill_embeddings_table(
    dry_run: bool = False, conn: DBConnection | None = None
) -> dict:
    """Backfill user_id for embeddings tables (Phase 1)."""
    if conn is None:
        conn = get_connection_from_env()

    ensure_documents_table(conn)
    embeddings_tables = _get_embeddings_tables(conn)

    if not embeddings_tables:
        return {
            "table": "embeddings",
            "total": 0,
            "backfilled": 0,
            "skipped": 0,
            "failed": 0,
            "details": [],
        }

    docs_table = conn.open_table("documents")
    all_results = []

    for table_name in embeddings_tables:
        logger.info(f"Phase 1: Processing {table_name}...")
        res = _backfill_table_core(
            table=conn.open_table(table_name),
            docs_table=docs_table,
            query_filter="user_id IS NULL",
            filter_fields=["doc_id", "chunk_id", "parse_hash", "model"],
            failure_user_id=ORPHANED_TEMPORARY,
            dry_run=dry_run,
            log_prefix=f"Embeddings Phase 1 ({table_name}):",
        )
        res["table"] = table_name
        all_results.append(res)

    return {
        "table": "embeddings",
        "total": sum(r["total"] for r in all_results),
        "backfilled": sum(r["backfilled"] for r in all_results),
        "skipped": sum(r["skipped"] for r in all_results),
        "failed": sum(r["failed"] for r in all_results),
        "details": all_results,
    }


def backfill_orphaned_embeddings(
    dry_run: bool = False, conn: DBConnection | None = None
) -> dict:
    """Retry backfill for orphaned embeddings (Phase 2)."""
    if conn is None:
        conn = get_connection_from_env()

    embeddings_tables = _get_embeddings_tables(conn)
    if not embeddings_tables:
        return {
            "table": "embeddings",
            "total": 0,
            "backfilled": 0,
            "skipped": 0,
            "failed": 0,
            "details": [],
        }

    docs_table = conn.open_table("documents")
    all_results = []

    for table_name in embeddings_tables:
        logger.info(f"Phase 2: Processing {table_name}...")
        res = _backfill_table_core(
            table=conn.open_table(table_name),
            docs_table=docs_table,
            query_filter=f"user_id = {ORPHANED_TEMPORARY}",
            filter_fields=["doc_id", "chunk_id", "parse_hash", "model"],
            failure_user_id=ORPHANED_PERMANENT,
            dry_run=dry_run,
            log_prefix=f"Embeddings Phase 2 ({table_name}):",
        )
        res["table"] = table_name
        all_results.append(res)

    return {
        "table": "embeddings",
        "total": sum(r["total"] for r in all_results),
        "backfilled": sum(r["backfilled"] for r in all_results),
        "skipped": sum(r["skipped"] for r in all_results),
        "failed": sum(r["failed"] for r in all_results),
        "details": all_results,
    }


def backfill_all(dry_run: bool = False, conn: DBConnection | None = None) -> dict:
    """Run full two-phase backfill for all tables."""
    if conn is None:
        conn = get_connection_from_env()

    if not _migration_lock.acquire(blocking=False):
        logger.warning("Another migration is already in progress")
        return {"error": "Migration lock already held"}

    try:
        logger.info("=" * 60)
        logger.info("LanceDB User ID Backfill Migration (Two-Phase)")
        logger.info("=" * 60)

        # Phase 1
        chunks_res = backfill_chunks_table(dry_run=dry_run, conn=conn)
        embeddings_res = backfill_embeddings_table(dry_run=dry_run, conn=conn)

        # Phase 2
        chunks_retry = {"backfilled": 0, "skipped": chunks_res["skipped"]}
        embeddings_retry = {"backfilled": 0, "skipped": embeddings_res["skipped"]}

        if chunks_res["skipped"] > 0:
            chunks_retry = backfill_orphaned_chunks(dry_run=dry_run, conn=conn)
            chunks_res["backfilled"] += chunks_retry["backfilled"]
            chunks_res["skipped"] = chunks_retry["skipped"]
            chunks_res["failed"] += chunks_retry["failed"]

        if embeddings_res["skipped"] > 0:
            embeddings_retry = backfill_orphaned_embeddings(dry_run=dry_run, conn=conn)
            embeddings_res["backfilled"] += embeddings_retry["backfilled"]
            embeddings_res["skipped"] = embeddings_retry["skipped"]
            embeddings_res["failed"] += embeddings_retry["failed"]

        return {
            "chunks": chunks_res,
            "embeddings": embeddings_res,
            "locked": True,
        }
    finally:
        _migration_lock.release()
        logger.info("Migration lock released")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Backfill user_id for LanceDB tables for multi-tenancy support.\n\n"
        "This script performs a two-phase migration:\n"
        "  Phase 1: Backfill records, mark orphaned records with user_id = -1\n"
        "  Phase 2: Retry orphaned records, mark permanent orphans with user_id = -2\n\n"
        "Orphaned records occur when chunks/embeddings exist without matching documents,\n"
        "which can happen due to concurrent document creation during migration.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate migration without making actual changes",
    )
    parser.add_argument(
        "--chunks-only",
        action="store_true",
        help="Only backfill chunks table (skip embeddings tables)",
    )
    parser.add_argument(
        "--embeddings-only",
        action="store_true",
        help="Only backfill embeddings tables (skip chunks table)",
    )
    args = parser.parse_args()

    try:
        if args.chunks_only:
            result = backfill_chunks_table(dry_run=args.dry_run)
        elif args.embeddings_only:
            result = backfill_embeddings_table(dry_run=args.dry_run)
        else:
            result = backfill_all(dry_run=args.dry_run)
        sys.exit(0)
    except Exception as e:
        logger.error(f"Migration failed: {e}", exc_info=True)
        sys.exit(2)
