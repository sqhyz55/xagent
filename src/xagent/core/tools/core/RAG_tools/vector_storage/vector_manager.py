"""Vector storage operations for RAG tools.

This module provides functions for:
1. Reading chunks from database for embedding computation
2. Writing embedding vectors to database with idempotency
3. Vector validation and consistency checking

This module handles pure vector data management and does not perform
any text-to-vector conversion (that's handled by AgentOS embedding nodes).
"""

from __future__ import annotations

import logging
import numbers
import os
import time
from typing import Any, Dict, List, Optional, cast

import numpy as np
import pandas as pd

from ..core.config import (
    DEFAULT_LANCEDB_BATCH_DELAY_MS,
    DEFAULT_LANCEDB_BATCH_SIZE,
)
from ..core.exceptions import (
    ConfigurationError,
    DatabaseOperationError,
    DocumentValidationError,
    VectorValidationError,
)
from ..core.schemas import (
    ChunkEmbeddingData,
    ChunkForEmbedding,
    EmbeddingReadResponse,
    EmbeddingWriteResponse,
    IndexOperation,
)
from ..LanceDB.model_tag_utils import to_model_tag
from ..LanceDB.schema_manager import ensure_embeddings_table
from ..storage.factory import get_vector_index_store
from ..utils.lancedb_query_utils import query_to_list
from ..utils.metadata_utils import deserialize_metadata, serialize_metadata
from ..utils.user_permissions import UserPermissions

logger = logging.getLogger(__name__)


def get_connection_from_env() -> Any:
    """Compatibility connection accessor for tests and legacy call sites."""
    return get_vector_index_store().get_raw_connection()


def _is_non_recoverable_merge_error(error: Exception) -> bool:
    """Classify merge_insert failures as recoverable or non-recoverable.

    Non-recoverable errors (schema/type/dimension issues) should re-raise
    immediately without fallback to add(). Recoverable errors (transient issues)
    should attempt fallback.

    Returns:
        True if error is non-recoverable (should re-raise), False otherwise.
    """
    # First, check for built-in Python exceptions that indicate non-recoverable issues
    # These are definitive regardless of LanceDB version
    if isinstance(error, (AttributeError, TypeError, ValueError)):
        return True

    # Then check for explicit LanceDB exception types when available.
    try:  # pragma: no cover - depends on installed lancedb version
        from lancedb.exceptions import (  # type: ignore[import-not-found]
            LanceDBSchemaError,
            LanceDBValidationError,
        )

        if isinstance(error, (LanceDBSchemaError, LanceDBValidationError)):
            return True
        # Known LanceDB exception type but not schema/validation -> recoverable
        return False
    except Exception:  # noqa: BLE001
        # LanceDB exception types not available - use string matching
        pass

    # String matching fallback for cases where LanceDB exceptions aren't available
    error_str = str(error).lower()

    # Narrow keyword list for cases where LanceDB exceptions aren't available.
    # This is a best-effort fallback for older LanceDB versions.
    non_recoverable_keywords = (
        "schema",
        "type mismatch",
        "type error",
        "validation",
        "dimension",
        "field",
        "column",
    )
    is_non_recoverable = any(
        keyword in error_str for keyword in non_recoverable_keywords
    )

    # Log warning about uncertain classification when using string matching
    if is_non_recoverable:
        logger.warning(
            "Error classified as non-recoverable via string matching. "
            "Upgrade LanceDB to get accurate exception-based classification. "
            "Error: %s",
            error,
        )
    else:
        logger.debug(
            "Error classified as recoverable via string matching (no schema keywords found). "
            "Attempting fallback to add() method. Error: %s",
            error,
        )

    return is_non_recoverable


def _open_embeddings_table(conn: Any, model_id: str) -> tuple[Any, str]:
    """Open an embeddings table for model_id with legacy fallback.

    If only the legacy table exists, this function performs a forward migration:
    it creates the Hub-ID-named table and copies legacy rows into it (rewriting
    the per-row ``model`` field to the Hub model ID).

    Returns:
        (table, table_name_used)
    """
    cleaned = (model_id or "").strip()
    if not cleaned:
        raise VectorValidationError("model_id must be a non-empty string")

    primary_table_name = f"embeddings_{to_model_tag(cleaned)}"

    # 1) Fast path: primary exists
    try:
        return conn.open_table(primary_table_name), primary_table_name
    except Exception as primary_exc:  # noqa: BLE001
        last_error: Exception | None = primary_exc

    # 2) Legacy fallback + forward migration
    legacy_table_name: str | None = None
    try:
        from ..utils.model_resolver import resolve_embedding_adapter

        cfg, _ = resolve_embedding_adapter(cleaned)
        legacy_table_name = f"embeddings_{to_model_tag(cfg.model_name)}"
    except Exception:
        legacy_table_name = None

    if legacy_table_name:
        try:
            legacy_table = conn.open_table(legacy_table_name)
        except Exception as legacy_exc:  # noqa: BLE001
            last_error = legacy_exc
        else:
            # Check if auto-migration is enabled
            from ..core.config import ENABLE_AUTO_EMBEDDINGS_MIGRATION

            if not ENABLE_AUTO_EMBEDDINGS_MIGRATION:
                # Auto-migration disabled: use legacy table directly
                logger.info(
                    "Auto-migration disabled. Using legacy embeddings table '%s' for hub_id=%s. "
                    "To enable automatic migration, set ENABLE_AUTO_EMBEDDINGS_MIGRATION=true",
                    legacy_table_name,
                    cleaned,
                )
                return legacy_table, legacy_table_name

            # Migrate legacy -> primary (best-effort, idempotent)
            try:
                vector_dim: int | None = None
                try:
                    vector_field = legacy_table.schema.field("vector")
                    list_size = getattr(vector_field.type, "list_size", None)
                    if list_size is not None:
                        vector_dim = int(list_size)
                except Exception:
                    vector_dim = None

                if vector_dim is None:
                    sample = legacy_table.search().limit(1).to_pandas()
                    if not sample.empty and "vector" in sample.columns:
                        vector_dim = len(sample.iloc[0]["vector"])

                ensure_embeddings_table(
                    conn, to_model_tag(cleaned), vector_dim=vector_dim
                )
                primary_table = conn.open_table(primary_table_name)

                # Copy all rows (small batches). Rewrite model -> Hub ID.
                # NOTE: This is an automatic forward migration and should be safe to re-run.
                batch_size = int(
                    os.getenv("LANCEDB_BATCH_SIZE", str(DEFAULT_LANCEDB_BATCH_SIZE))
                )
                offset = 0
                while True:
                    df = (
                        legacy_table.search()
                        .limit(batch_size)
                        .offset(offset)
                        .to_pandas()
                    )
                    if df.empty:
                        break
                    df["model"] = cleaned
                    (
                        primary_table.merge_insert(
                            on=[
                                "collection",
                                "doc_id",
                                "chunk_id",
                                "parse_hash",
                                "model",
                            ]
                        )
                        .when_matched_update_all()
                        .when_not_matched_insert_all()
                        .execute(df)
                    )
                    offset += len(df)

                logger.info(
                    "Forward-migrated embeddings table '%s' -> '%s' for hub_id=%s",
                    legacy_table_name,
                    primary_table_name,
                    cleaned,
                )
                return primary_table, primary_table_name
            except Exception as migrate_exc:  # noqa: BLE001
                logger.warning(
                    "Failed to forward-migrate legacy embeddings table '%s' -> '%s' (hub_id=%s): %s. "
                    "Falling back to legacy table for this request.",
                    legacy_table_name,
                    primary_table_name,
                    cleaned,
                    migrate_exc,
                )
                return legacy_table, legacy_table_name

    raise VectorValidationError(
        f"Embeddings table for model '{cleaned}' does not exist or is inaccessible: {last_error}"
    )


def validate_query_vector(
    query_vector: List[float],
    model_tag: Optional[str] = None,
    conn: Any = None,
    user_id: Optional[int] = None,
    is_admin: bool = False,
) -> None:
    """Validate query vector format and content.

    Args:
        query_vector: Query vector to validate
        model_tag: Optional model tag for dimension validation
        conn: Optional LanceDB connection for validation
        user_id: Optional user ID for filtering (for multi-tenancy)
        is_admin: Whether user has admin privileges
    """
    if not isinstance(query_vector, list):
        raise VectorValidationError("query_vector must be a list")

    if len(query_vector) == 0:
        raise VectorValidationError("query_vector cannot be empty")

    # Use numbers.Number to support numpy scalar types (np.int32, np.float64, etc.)
    if not all(isinstance(x, numbers.Number) for x in query_vector):
        raise VectorValidationError("query_vector must contain only numbers")

    # Check for invalid values (NaN or infinity)
    # Convert to float first to handle numpy scalar types
    for x in query_vector:
        if not isinstance(x, numbers.Real):
            continue  # Skip non-real numbers (e.g., complex numbers)
        float_val = float(x)
        if float_val != float_val or abs(float_val) == float("inf"):
            raise VectorValidationError(
                "query_vector contains invalid values (NaN or infinity)"
            )

    if model_tag and conn:
        # First validate model_tag format and table existence
        normalized_model_tag = to_model_tag(model_tag)
        validate_embed_model(conn, normalized_model_tag)

        table_name = f"embeddings_{normalized_model_tag}"
        try:
            table = conn.open_table(table_name)
            expected_dim = None

            # Method 1: Try to get dimension from schema (for fixed-size vector columns)
            try:
                vector_field = table.schema.field("vector")
                # Safely check if list_size attribute exists (fixed-size list)
                list_size = getattr(vector_field.type, "list_size", None)
                if list_size is not None:
                    expected_dim = list_size
            except (AttributeError, KeyError) as e:
                logger.debug(
                    "Could not get vector dimension from schema for %s: %s. Will try to infer from data.",
                    table_name,
                    e,
                )

            # Method 2: If schema doesn't have fixed dimension, infer from actual data
            if expected_dim is None:
                expected_dim = get_stored_vector_dimension(
                    conn, model_tag, user_id, is_admin
                )

            # Perform dimension validation if we got a dimension
            if expected_dim is not None:
                if len(query_vector) != expected_dim:
                    raise VectorValidationError(
                        f"Query vector dimension {len(query_vector)} does not match stored dimension {expected_dim} for model '{model_tag}'"
                    )
            else:
                logger.warning(
                    "Could not determine expected vector dimension for %s "
                    "(table may be empty or schema is variable-length). "
                    "Skipping dimension consistency check.",
                    table_name,
                )
        except VectorValidationError:
            # Re-raise validation errors (don't catch them)
            raise
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "Failed to perform dimension validation for %s: %s. Skipping dimension consistency check.",
                table_name,
                e,
            )


def validate_embed_model(conn: Any, model_tag: str) -> None:
    """Validate embed model exists and is accessible."""
    import re

    # Validate model_tag format (cannot contain characters that affect table name)
    if not re.match(r"^[a-zA-Z0-9_-]+$", model_tag):
        raise VectorValidationError(
            f"Invalid model_tag format: {model_tag}. Only alphanumeric, underscore, and hyphen allowed."
        )

    # Validate that at least one candidate table exists (primary hub-id naming, legacy fallback).
    try:
        _, used_name = _open_embeddings_table(conn, model_tag)
        logger.debug("validate_embed_model resolved table: %s", used_name)
    except VectorValidationError:
        raise


def get_stored_vector_dimension(
    conn: Any,
    model_tag: str,
    user_id: Optional[int] = None,
    is_admin: bool = False,
) -> Optional[int]:
    """Get the vector dimension for a model from database.

    Args:
        conn: LanceDB connection
        model_tag: Model tag to look up
        user_id: Optional user ID for filtering (for multi-tenancy)
        is_admin: Whether user has admin privileges

    Returns:
        Vector dimension if found, None otherwise
    """
    try:
        table, _ = _open_embeddings_table(conn, model_tag)

        # Apply user filter for multi-tenancy
        user_filter_expr = UserPermissions.get_user_filter(user_id, is_admin)

        # Query one record to get dimension, with optional user filtering
        # OPTIMIZATION: Use unified query_to_list() with three-tier fallback
        if user_filter_expr:
            sample_list = query_to_list(table.search().where(user_filter_expr).limit(1))
        else:
            sample_list = query_to_list(table.head(1))

        if sample_list:
            vector_dim = sample_list[0].get("vector_dimension")
            if vector_dim is not None:
                return int(vector_dim)
    except Exception as e:  # noqa: BLE001
        logger.debug(
            "Could not get stored vector dimension for %s: %s. This is expected if the table is new or empty.",
            model_tag,
            e,
        )
        pass
    return None


def _safe_int_conversion(value: Any, default: int = 0) -> int:
    """Safely convert value to int, handling None and NaN.

    Args:
        value: Value to convert (can be None, NaN, int, float, etc.)
        default: Default value if conversion fails

    Returns:
        Integer value, or default if value is None/NaN/not convertible
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return default
    try:
        return int(value)
    except (ValueError, TypeError):
        return default


def _safe_str_value(value: Any) -> Optional[str]:
    """Extract string value, returning None for NaN/None values.

    This handles pandas DataFrame's NaN preservation behavior where
    NaN values are not automatically converted to None.

    Args:
        value: Value from pandas DataFrame (can be str, None, or NaN)

    Returns:
        String value, or None if value is None/NaN
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    return str(value) if value is not None else None


def read_chunks_for_embedding(
    collection: str,
    doc_id: str,
    parse_hash: str,
    model: str,
    filters: Optional[Dict[str, Any]] = None,
    user_id: Optional[int] = None,
    is_admin: bool = False,
) -> EmbeddingReadResponse:
    """Read chunks from database for embedding computation.

    Phase 1A: Refactored to use storage abstraction layer instead of raw connection.
    """
    try:
        # Validate inputs
        if not collection or not doc_id or not parse_hash or not model:
            raise DocumentValidationError(
                "Collection, doc_id, parse_hash, and model are required"
            )

        logger.info(
            "Reading chunks for embedding: collection=%s, doc_id=%s, parse_hash=%s..., model=%s",
            collection,
            doc_id,
            parse_hash[:8],
            model,
        )

        # Use storage abstraction instead of raw connection
        from ..storage.factory import get_vector_index_store

        vector_store = get_vector_index_store()

        # Build query filters
        query_filters: Dict[str, Any] = {
            "collection": collection,
            "doc_id": doc_id,
            "parse_hash": parse_hash,
        }

        # Add additional filters if provided
        if filters:
            query_filters.update(filters)

        # Use abstraction layer for counting (returns 0 if table doesn't exist)
        total_count = vector_store.count_rows_or_zero(
            table_name="chunks",
            filters=query_filters,
            user_id=user_id,
            is_admin=is_admin,
        )
        if total_count == 0:
            logger.info("No chunks found for the given criteria")
            return EmbeddingReadResponse(chunks=[], total_count=0, pending_count=0)

        # Use abstraction layer for batch iteration
        chunks_data = []
        for batch in vector_store.iter_batches(
            table_name="chunks",
            columns=None,  # Select all columns
            batch_size=1000,
            filters=query_filters,
            user_id=user_id,
            is_admin=is_admin,
        ):
            batch_df = batch.to_pandas()
            for _, row in batch_df.iterrows():
                chunks_data.append(row.to_dict())
            if len(chunks_data) >= total_count:
                break

        # Check which chunks already have embeddings using abstraction layer
        embedded_chunk_ids = set()
        model_tag = to_model_tag(model)
        embeddings_table_name = f"embeddings_{model_tag}"

        try:
            # Get existing embeddings for these chunks
            # Only select chunk_id column to avoid loading unnecessary vector data
            embedding_filters: Dict[str, Any] = {
                "collection": collection,
                "doc_id": doc_id,
                "parse_hash": parse_hash,
            }

            # Use abstraction layer to query embeddings (returns 0 if table doesn't exist)
            # Note: We don't filter by 'model' field as it's not in current schema
            embedding_count = vector_store.count_rows_or_zero(
                table_name=embeddings_table_name,
                filters=embedding_filters,
                user_id=user_id,
                is_admin=is_admin,
            )

            if embedding_count > 0:
                # Read chunk_ids from embeddings table
                for batch in vector_store.iter_batches(
                    table_name=embeddings_table_name,
                    columns=["chunk_id"],
                    filters=embedding_filters,
                    user_id=user_id,
                    is_admin=is_admin,
                ):
                    batch_df = batch.to_pandas()
                    for chunk_id in batch_df["chunk_id"]:
                        if chunk_id is not None:
                            embedded_chunk_ids.add(chunk_id)

        except Exception as e:  # noqa: BLE001
            # If embeddings table doesn't exist or query fails, assume no embeddings exist
            logger.warning(
                "Failed to query existing embeddings for model %s (assuming none exist): %s",
                model,
                e,
            )
            embedded_chunk_ids = set()

        # OPTIMIZATION: Filter and construct ChunkForEmbedding objects in one pass
        pending_chunks = []
        for chunk_dict in chunks_data:
            chunk_id = chunk_dict["chunk_id"]
            if chunk_id not in embedded_chunk_ids:
                # Deserialize metadata from JSON string to dictionary
                metadata = deserialize_metadata(chunk_dict.get("metadata"))

                # Handle index with NaN-safe conversion
                index = _safe_int_conversion(chunk_dict.get("index"), default=0)

                page_number_value = chunk_dict.get("page_number")
                # Convert to int only if valid and > 0 (schema requires gt=0)
                if page_number_value is not None:
                    page_num = _safe_int_conversion(page_number_value, default=1)
                    page_number = page_num if page_num > 0 else None
                else:
                    page_number = None

                # Normalize optional string fields using NaN-safe helper
                # pandas to_pandas() preserves NaN values, so explicit NaN handling needed
                section = _safe_str_value(chunk_dict.get("section"))
                anchor = _safe_str_value(chunk_dict.get("anchor"))
                json_path = _safe_str_value(chunk_dict.get("json_path"))

                chunk = ChunkForEmbedding(
                    doc_id=chunk_dict["doc_id"],
                    chunk_id=chunk_id,
                    parse_hash=chunk_dict["parse_hash"],
                    index=index,
                    text=chunk_dict["text"],
                    chunk_hash=chunk_dict["chunk_hash"],
                    page_number=page_number,
                    section=section,
                    anchor=anchor,
                    json_path=json_path,
                    metadata=metadata,
                )
                pending_chunks.append(chunk)

        pending_count = len(pending_chunks)

        logger.info(
            "Found %d total chunks, %d need embedding for model %s",
            total_count,
            pending_count,
            model,
        )

        return EmbeddingReadResponse(
            chunks=pending_chunks, total_count=total_count, pending_count=pending_count
        )

    except Exception as e:
        if isinstance(
            e,
            (
                DocumentValidationError,
                DatabaseOperationError,
                ConfigurationError,
                VectorValidationError,
            ),
        ):
            raise
        logger.error("Failed to read chunks for embedding: %s", e)
        raise DatabaseOperationError(f"Failed to read chunks for embedding: {e}") from e


def _group_embeddings_by_model(
    embeddings: List[ChunkEmbeddingData],
) -> Dict[str, List[ChunkEmbeddingData]]:
    """Group embeddings by model for batch processing."""
    embeddings_by_model: Dict[str, List[ChunkEmbeddingData]] = {}
    for embedding in embeddings:
        model = embedding.model
        if model not in embeddings_by_model:
            embeddings_by_model[model] = []
        embeddings_by_model[model].append(embedding)
    return embeddings_by_model


def _validate_and_prepare_table(
    conn: Any,
    model_tag: str,
    table_name: str,
    vector_dim: int,
) -> Any:
    """Ensure database table exists and has compatible schema.

    If the table exists, checks the vector field type and dimension; drops and
    recreates the table when dimension or type is incompatible.

    Args:
        conn: LanceDB connection (e.g. from get_connection_from_env).
        model_tag: Model tag used for table naming (e.g. from to_model_tag).
        table_name: Full embeddings table name (e.g. embeddings_<model_tag>).
        vector_dim: Expected vector dimension for the table schema.

    Returns:
        LanceDB table instance for the embeddings table (existing or newly created).
    """
    conn_any = cast(Any, conn)
    try:
        existing_tables: List[str] = []
        if hasattr(conn_any, "table_names"):
            existing_tables = list(conn_any.table_names())
        if table_name in existing_tables:
            existing_table = conn.open_table(table_name)
            vector_field = existing_table.schema.field("vector")
            if hasattr(vector_field.type, "list_size"):
                existing_dim = vector_field.type.list_size
                if existing_dim != vector_dim:
                    logger.warning(
                        "Dropping table %s due to vector dimension mismatch: existing=%s, new=%s",
                        table_name,
                        existing_dim,
                        vector_dim,
                    )
                    drop_fn = getattr(conn_any, "drop_table", None)
                    if callable(drop_fn):
                        drop_fn(table_name)
            else:
                logger.warning(
                    "Dropping table %s due to incompatible vector field type",
                    table_name,
                )
                drop_fn = getattr(conn_any, "drop_table", None)
                if callable(drop_fn):
                    drop_fn(table_name)
    except Exception as schema_check_error:  # noqa: BLE001
        logger.warning("Error checking table schema: %s", schema_check_error)
        try:
            drop_fn = getattr(conn_any, "drop_table", None)
            if callable(drop_fn):
                drop_fn(table_name)
        except Exception as drop_error:  # noqa: BLE001
            logger.warning("Failed to drop table %s: %s", table_name, drop_error)
            pass

    # Ensure embeddings table exists with the correct schema
    ensure_embeddings_table(conn, model_tag, vector_dim=vector_dim)
    return conn.open_table(table_name)


def _process_batch(
    embeddings_table: Any,
    records_to_merge: List[Dict[str, Any]],
    batch_idx: int,
    total_batches: int,
    model: str,
) -> int:
    """Process a single batch of embeddings.

    Uses merge_insert for upsert; on recoverable errors falls back to add().
    Non-recoverable errors (schema/type/dimension) are re-raised without fallback.

    Args:
        embeddings_table: LanceDB table to write to (from _validate_and_prepare_table).
        records_to_merge: List of record dicts with keys matching table schema.
        batch_idx: Zero-based batch index (for logging).
        total_batches: Total number of batches (for logging).
        model: Model name (for logging).

    Returns:
        Number of upserted records (len(records_to_merge) on success).
    """
    try:
        # Try merge_insert first (preferred method for upserts)
        embeddings_table.merge_insert(
            ["collection", "doc_id", "chunk_id", "parse_hash", "model"]
        ).when_matched_update_all().when_not_matched_insert_all().execute(
            records_to_merge
        )
        method_used = "merge_insert"
    except Exception as merge_error:  # noqa: BLE001
        error_type = type(merge_error).__name__
        if _is_non_recoverable_merge_error(merge_error):
            # Log critical error and re-raise without fallback
            logger.error(
                "merge_insert failed with non-recoverable error for batch %d/%d "
                "(error_type=%s): %s. This may indicate schema mismatch or data corruption. "
                "Not attempting fallback to add() method.",
                batch_idx + 1,
                total_batches,
                error_type,
                merge_error,
            )
            raise

        # For recoverable errors (e.g., temporary issues, network errors), attempt fallback
        logger.warning(
            "merge_insert failed for batch %d/%d (error_type=%s): %s; "
            "attempting fallback to add() method",
            batch_idx + 1,
            total_batches,
            error_type,
            merge_error,
        )
        try:
            embeddings_table.add(pd.DataFrame(records_to_merge))
            method_used = "add"
            logger.info(
                "Successfully used add() fallback for batch %d/%d after merge_insert failure",
                batch_idx + 1,
                total_batches,
            )
        except Exception as add_error:  # noqa: BLE001
            logger.error(
                "Fallback add() also failed for batch %d/%d: %s. "
                "Both merge_insert and add() methods failed.",
                batch_idx + 1,
                total_batches,
                add_error,
            )
            raise

    batch_upserted = len(records_to_merge)
    logger.info(
        "Successfully processed batch %d/%d (%d embeddings) for model %s using %s",
        batch_idx + 1,
        total_batches,
        batch_upserted,
        model,
        method_used,
    )

    # Optional delay between batches to reduce I/O pressure (default: disabled)
    batch_delay_ms = int(
        os.getenv("LANCEDB_BATCH_DELAY_MS", str(DEFAULT_LANCEDB_BATCH_DELAY_MS))
    )
    if batch_delay_ms > 0 and batch_idx < total_batches - 1:  # No delay after last
        time.sleep(batch_delay_ms / 1000.0)

    return batch_upserted


def _process_model_embeddings(
    collection: str,
    model: str,
    model_embeddings: List[ChunkEmbeddingData],
    create_index: bool,
    user_id: Optional[int] = None,
) -> tuple[int, str]:
    """Process embeddings for a single model using abstraction layer.

    Returns:
        Tuple of (upserted_count, index_status)
    """
    from ..storage.factory import get_vector_index_store

    model_tag = to_model_tag(model)
    table_name = f"embeddings_{model_tag}"

    # Get vector dimension from first embedding for validation and logging
    first_embedding = model_embeddings[0]
    vector_dim = len(first_embedding.vector)

    vector_dimensions = [len(item.vector) for item in model_embeddings]
    unique_dims = set(vector_dimensions)
    if len(unique_dims) > 1:
        logger.error(
            "Multiple vector dimensions found in batch for model %s: %s",
            model,
            unique_dims,
        )
        raise VectorValidationError(
            f"Multiple vector dimensions found for model {model}: {unique_dims}"
        )
    logger.info(
        "Vector dimension consistency check passed: all vectors have dimension %d",
        vector_dim,
    )

    logger.info(
        "Writing %d embeddings for model %s to table %s (vector_dim: %d)",
        len(model_embeddings),
        model,
        table_name,
        vector_dim,
    )

    # Process embeddings in batches to prevent memory issues and LanceDB spills
    original_batch_size = int(
        os.getenv("LANCEDB_BATCH_SIZE", str(DEFAULT_LANCEDB_BATCH_SIZE))
    )
    batch_size = original_batch_size
    total_batches_for_logging = (
        len(model_embeddings) + original_batch_size - 1
    ) // original_batch_size
    logger.info(
        "Processing %d embeddings in %d batches of size %d",
        len(model_embeddings),
        total_batches_for_logging,
        original_batch_size,
    )

    # OPTIMIZATION: Use single timestamp for entire batch
    batch_timestamp = pd.Timestamp.now(tz="UTC")

    upserted_count = 0
    failed_batches = 0

    current_idx = 0
    total_embeddings = len(model_embeddings)

    max_spill_retries = int(os.getenv("LANCEDB_MAX_SPILL_RETRIES", "3"))
    spill_retry_count = 0

    vector_store = get_vector_index_store()

    while current_idx < total_embeddings:
        end_idx = min(current_idx + batch_size, total_embeddings)
        batch_embeddings = model_embeddings[current_idx:end_idx]
        current_batch_size = len(batch_embeddings)

        # OPTIMIZATION: Prepare records for this batch using list comprehension
        records_to_merge = [
            {
                "collection": collection,
                "doc_id": embedding.doc_id,
                "chunk_id": embedding.chunk_id,
                "parse_hash": embedding.parse_hash,
                "model": model,
                "vector": embedding.vector,
                "text": embedding.text,
                "chunk_hash": embedding.chunk_hash,
                "created_at": batch_timestamp,
                "vector_dimension": vector_dim,
                "metadata": serialize_metadata(embedding.metadata),
                "user_id": user_id,  # Add user_id for multi-tenancy
            }
            for embedding in batch_embeddings
        ]

        try:
            batch_idx_for_logging = current_idx // original_batch_size
            # Use abstraction layer for upsert (includes fallback logic)
            vector_store.upsert_embeddings(model_tag, records_to_merge)
            batch_upserted = len(records_to_merge)
            upserted_count += batch_upserted
            current_idx = end_idx  # Move to next batch on success
            spill_retry_count = 0  # Reset after a successful batch

            logger.info(
                "Successfully processed batch %d/%d (%d embeddings) for model %s",
                batch_idx_for_logging + 1,
                total_batches_for_logging,
                batch_upserted,
                model,
            )

        except Exception as batch_error:  # noqa: BLE001
            failed_batches += 1
            logger.error(
                "Failed to process batch at index %d: %s",
                current_idx,
                batch_error,
            )
            logger.error(
                "Batch details: start_idx=%d, end_idx=%d, batch_size=%d, model=%s",
                current_idx,
                end_idx,
                current_batch_size,
                model,
            )

            # For critical LanceDB errors, consider reducing batch size
            if "Spill has sent an error" in str(batch_error):
                logger.error(
                    "Critical LanceDB spill error detected. "
                    "Consider reducing batch size by setting LANCEDB_BATCH_SIZE environment variable."
                )
                spill_retry_count += 1
                if spill_retry_count <= max_spill_retries:
                    if batch_size > 50:  # Reduce to even smaller size
                        new_batch_size = max(50, batch_size // 2)
                        logger.info(
                            "Reducing batch size from %d to %d and retrying (spill retry %d/%d)",
                            batch_size,
                            new_batch_size,
                            spill_retry_count,
                            max_spill_retries,
                        )
                        batch_size = new_batch_size
                    else:
                        logger.info(
                            "Retrying batch with minimum batch_size=%d (spill retry %d/%d)",
                            batch_size,
                            spill_retry_count,
                            max_spill_retries,
                        )
                    # Continue without advancing current_idx to retry
                    continue

            # Re-raise for other errors or if we can't reduce batch size further
            raise

    # Log final batch processing summary
    if failed_batches > 0:
        logger.warning(
            "Batch processing completed with %d failed batches out of %d total batches for model %s",
            failed_batches,
            total_batches_for_logging,
            model,
        )
        if model_embeddings:
            logger.warning(
                "Successfully processed %d out of %d embeddings (%.1f%% success rate)",
                upserted_count,
                len(model_embeddings),
                upserted_count / len(model_embeddings) * 100,
            )
    logger.info("Successfully merged %d embeddings for model %s", upserted_count, model)

    logger.info("Processed model %s: upserted %d embeddings", model, upserted_count)

    # Handle index creation using abstraction layer
    index_status: str = IndexOperation.SKIPPED.value
    if create_index:
        try:
            index_status = vector_store.create_index(model_tag, readonly=False)
        except Exception as index_error:  # noqa: BLE001
            logger.warning("Failed to create index for %s: %s", table_name, index_error)
            index_status = IndexOperation.FAILED.value

    return upserted_count, index_status


def write_vectors_to_db(
    collection: str,
    embeddings: List[ChunkEmbeddingData],
    create_index: bool = True,
    user_id: Optional[int] = None,
) -> EmbeddingWriteResponse:
    """Write embedding vectors to database with idempotency."""
    if not embeddings:
        return EmbeddingWriteResponse(
            upsert_count=0,
            deleted_stale_count=0,
            index_status=IndexOperation.SKIPPED.value,
        )

    try:
        # Validate inputs
        if not collection:
            raise DocumentValidationError("Collection name is required")

        # Group embeddings by model for batch processing
        embeddings_by_model = _group_embeddings_by_model(embeddings)

        total_upserted = 0
        index_statuses = []

        # Process each model separately (abstraction layer handles connection internally)
        for model, model_embeddings in embeddings_by_model.items():
            upserted, idx_status = _process_model_embeddings(
                collection, model, model_embeddings, create_index, user_id
            )
            total_upserted += upserted
            index_statuses.append(idx_status)

        # Determine overall index status (map index_manager strings to IndexOperation)
        if "index_building" in index_statuses:
            overall_index_status = IndexOperation.CREATED
        elif "index_ready" in index_statuses:
            overall_index_status = IndexOperation.READY
        elif "failed" in index_statuses or "index_corrupted" in index_statuses:
            overall_index_status = IndexOperation.FAILED
        elif "below_threshold" in index_statuses:
            overall_index_status = IndexOperation.SKIPPED_THRESHOLD
        else:
            overall_index_status = IndexOperation.SKIPPED

        logger.info(
            "Embedding write completed: %d upserted, index status: %s",
            total_upserted,
            overall_index_status.value,
        )

        return EmbeddingWriteResponse(
            upsert_count=total_upserted,
            deleted_stale_count=0,  # merge_insert handles updates automatically
            index_status=overall_index_status.value,
        )

    except Exception as e:
        if isinstance(
            e,
            (
                DocumentValidationError,
                DatabaseOperationError,
                ConfigurationError,
                VectorValidationError,
            ),
        ):
            raise
        logger.error("Failed to write embeddings to database: %s", e)
        raise DatabaseOperationError(
            f"Failed to write embeddings to database: {e}"
        ) from e
