from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Protocol

import pyarrow as pa  # type: ignore
from lancedb.db import DBConnection


class DataTypeLike(Protocol):
    """Structural type placeholder for pyarrow DataType-like values."""


class FieldLike(Protocol):
    """Structural field contract used by schema migration helpers."""

    name: str
    type: DataTypeLike


logger = logging.getLogger(__name__)

__all__ = [
    "ensure_documents_table",
    "ensure_parses_table",
    "ensure_chunks_table",
    "ensure_embeddings_table",
    "ensure_main_pointers_table",
    "ensure_prompt_templates_table",
    "ensure_ingestion_runs_table",
    "ensure_collection_config_table",
    "ensure_collection_metadata_table",
]


def _table_exists(conn: DBConnection, name: str) -> bool:
    try:
        conn.open_table(name)
        return True
    except Exception:
        return False


def _is_table_already_exists_error(exc: Exception) -> bool:
    """Best-effort check for table-already-exists errors across LanceDB versions."""
    message = str(exc).lower()
    return "already exists" in message and "table" in message


def _get_sql_default_for_pa_type(pa_type: DataTypeLike) -> str:
    """Map PyArrow type to LanceDB SQL default value expression."""
    if pa.types.is_string(pa_type) or pa.types.is_large_string(pa_type):
        return "''"
    if pa.types.is_integer(pa_type):
        return "0"
    if pa.types.is_floating(pa_type):
        return "0.0"
    if pa.types.is_boolean(pa_type):
        return "false"
    if pa.types.is_timestamp(pa_type):
        return "CAST(NULL AS TIMESTAMP)"
    return "NULL"


def _ensure_schema_fields(
    conn: DBConnection, table_name: str, target_schema: Iterable[FieldLike]
) -> None:
    """Ensure an existing table matches the target schema by adding missing columns.

    Only ADDS missing columns. Does not delete extra columns nor modify existing types.
    """
    if not _table_exists(conn, table_name):
        return

    table = conn.open_table(table_name)
    existing_schema = table.schema
    existing_field_names = {field.name for field in existing_schema}
    missing_fields = [f for f in target_schema if f.name not in existing_field_names]

    if not missing_fields:
        return

    logger.info(
        "Auto-migrating schema for table '%s'. Adding missing fields: %s",
        table_name,
        [f.name for f in missing_fields],
    )
    new_cols = {}
    for field in missing_fields:
        default_expr = _get_sql_default_for_pa_type(field.type)
        new_cols[field.name] = default_expr

    try:
        table.add_columns(new_cols)
        logger.info("Successfully migrated schema for table '%s'", table_name)
    except Exception as e:
        logger.error("Failed to add columns to table '%s': %s", table_name, e)
        raise


def _create_table(
    conn: DBConnection, name: str, schema: Iterable[FieldLike] | None = None
) -> None:
    # Avoid check-then-act race: attempt creation first.
    try:
        conn.create_table(name, schema=schema)
    except Exception as exc:
        if not _is_table_already_exists_error(exc):
            raise

    # Reconcile existing/new table schema after create attempt.
    if schema:
        _ensure_schema_fields(conn, name, schema)


def ensure_documents_table(conn: DBConnection) -> None:
    schema = pa.schema(
        [
            pa.field("collection", pa.string()),
            pa.field("doc_id", pa.string()),
            pa.field("source_path", pa.string()),
            pa.field("file_type", pa.string()),
            pa.field("content_hash", pa.string()),
            pa.field("uploaded_at", pa.timestamp("us")),
            pa.field("title", pa.string()),
            pa.field("language", pa.string()),
            pa.field("user_id", pa.int64()),
        ]
    )

    # Automatic migration for existing tables missing 'user_id'
    if _table_exists(conn, "documents"):
        try:
            table = conn.open_table("documents")
            if "user_id" not in table.schema.names:
                logger.info(
                    "Migrating 'documents' table: adding missing 'user_id' column"
                )
                # Add user_id column with null default, cast to bigint (int64)
                table.add_columns({"user_id": "cast(null as bigint)"})
        except Exception as e:
            logger.warning(f"Failed to check/migrate 'documents' table schema: {e}")

    _create_table(conn, "documents", schema=schema)


def ensure_parses_table(conn: DBConnection) -> None:
    schema = pa.schema(
        [
            pa.field("collection", pa.string()),
            pa.field("doc_id", pa.string()),
            pa.field("parse_hash", pa.string()),
            pa.field("parser", pa.string()),
            pa.field("created_at", pa.timestamp("us")),
            pa.field("params_json", pa.string()),
            pa.field("parsed_content", pa.large_string()),
            pa.field("user_id", pa.int64()),
        ]
    )

    # Automatic migration for existing tables missing 'user_id'
    if _table_exists(conn, "parses"):
        try:
            table = conn.open_table("parses")
            if "user_id" not in table.schema.names:
                logger.info("Migrating 'parses' table: adding missing 'user_id' column")
                table.add_columns({"user_id": "cast(null as bigint)"})
        except Exception as e:
            logger.warning(f"Failed to check/migrate 'parses' table schema: {e}")

    _create_table(conn, "parses", schema=schema)


def ensure_chunks_table(conn: DBConnection) -> None:
    """Ensure the chunks table exists with proper schema.

    If the table already exists, we attempt best-effort schema evolution by
    adding any missing columns (see _ensure_schema_fields).
    """
    schema = pa.schema(
        [
            pa.field("collection", pa.string()),
            pa.field("doc_id", pa.string()),
            pa.field("parse_hash", pa.string()),
            pa.field("chunk_id", pa.string()),
            pa.field("index", pa.int32()),
            pa.field("text", pa.large_string()),
            pa.field("page_number", pa.int32()),
            pa.field("section", pa.string()),
            pa.field("anchor", pa.string()),
            pa.field("json_path", pa.string()),
            pa.field("chunk_hash", pa.string()),
            pa.field("config_hash", pa.string()),
            pa.field("created_at", pa.timestamp("us")),
            pa.field("metadata", pa.string()),
            pa.field("user_id", pa.int64()),
        ]
    )
    _create_table(conn, "chunks", schema=schema)


def ensure_embeddings_table(
    conn: DBConnection, model_tag: str, vector_dim: int | None = None
) -> None:
    """Ensure the embeddings table exists with proper schema.

    If the table already exists, we attempt best-effort schema evolution by
    adding any missing columns (see _ensure_schema_fields).
    """
    table_name = f"embeddings_{model_tag}"

    # Support dynamic vector dimension: if provided, create a FixedSizeList; otherwise allow variable-length
    vector_field_type = (
        pa.list_(pa.float32(), list_size=vector_dim)
        if vector_dim is not None
        else pa.list_(pa.float32())
    )
    schema = pa.schema(
        [
            pa.field("collection", pa.string()),
            pa.field("doc_id", pa.string()),
            pa.field("chunk_id", pa.string()),
            pa.field("parse_hash", pa.string()),
            pa.field("model", pa.string()),
            pa.field("vector", vector_field_type),
            pa.field("vector_dimension", pa.int32()),
            pa.field("text", pa.large_string()),
            pa.field("chunk_hash", pa.string()),
            pa.field("created_at", pa.timestamp("us")),
            pa.field("metadata", pa.string()),
            pa.field("user_id", pa.int64()),
        ]
    )
    _create_table(
        conn,
        table_name,
        schema=schema,
    )


def ensure_main_pointers_table(conn: DBConnection) -> None:
    """Ensure the main_pointers table exists with proper schema."""
    schema = pa.schema(
        [
            pa.field("collection", pa.string()),
            pa.field("doc_id", pa.string()),
            pa.field("step_type", pa.string()),
            pa.field("model_tag", pa.string()),
            pa.field("semantic_id", pa.string()),
            pa.field("technical_id", pa.string()),
            pa.field("created_at", pa.timestamp("ms")),
            pa.field("updated_at", pa.timestamp("ms")),
            pa.field("operator", pa.string()),
        ]
    )
    _create_table(conn, "main_pointers", schema=schema)


def ensure_prompt_templates_table(conn: DBConnection) -> None:
    """Ensure the prompt_templates table exists with proper schema."""
    table_name = "prompt_templates"
    schema = pa.schema(
        [
            pa.field("collection", pa.string()),
            pa.field("id", pa.string()),
            pa.field("name", pa.string()),
            pa.field("template", pa.string()),
            pa.field("version", pa.int64()),
            pa.field("is_latest", pa.bool_()),
            pa.field("metadata", pa.string()),  # JSON string, nullable
            pa.field("user_id", pa.int64()),  # Multi-tenancy support
            pa.field("created_at", pa.timestamp("us")),
            pa.field("updated_at", pa.timestamp("us")),
        ]
    )

    # Automatic migration for existing tables missing 'user_id'
    if _table_exists(conn, table_name):
        try:
            table = conn.open_table(table_name)
            if "user_id" not in table.schema.names:
                logger.info(
                    f"Migrating '{table_name}' table: adding missing 'user_id' column"
                )
                table.add_columns({"user_id": "cast(null as bigint)"})
        except Exception as e:
            logger.warning(f"Failed to check/migrate '{table_name}' table schema: {e}")

    _create_table(conn, table_name, schema=schema)


def ensure_ingestion_runs_table(conn: DBConnection) -> None:
    """Ensure the ingestion_runs table exists with proper schema."""
    schema = pa.schema(
        [
            pa.field("collection", pa.string()),
            pa.field("doc_id", pa.string()),
            pa.field("status", pa.string()),
            pa.field("message", pa.string()),
            pa.field("parse_hash", pa.string()),
            pa.field("created_at", pa.timestamp("us")),
            pa.field("updated_at", pa.timestamp("us")),
            pa.field("user_id", pa.int64()),
        ]
    )

    # Automatic migration for existing tables missing 'user_id'
    if _table_exists(conn, "ingestion_runs"):
        try:
            table = conn.open_table("ingestion_runs")
            if "user_id" not in table.schema.names:
                logger.info(
                    "Migrating 'ingestion_runs' table: adding missing 'user_id' column"
                )
                table.add_columns({"user_id": "cast(null as bigint)"})
        except Exception as e:
            logger.warning(
                f"Failed to check/migrate 'ingestion_runs' table schema: {e}"
            )

    _create_table(conn, "ingestion_runs", schema=schema)


def ensure_collection_config_table(conn: DBConnection) -> None:
    """Ensure the collection_config table exists with proper schema.

    This table stores configuration/metadata for each collection.

    Args:
        conn: LanceDB connection
    """
    table_name = "collection_config"
    schema = pa.schema(
        [
            pa.field("collection", pa.string()),
            pa.field("config_json", pa.string()),  # Stores IngestionConfig as JSON
            pa.field("updated_at", pa.timestamp("us")),
            pa.field("user_id", pa.int64()),
        ]
    )

    _create_table(conn, table_name, schema=schema)


def ensure_collection_metadata_table(conn: DBConnection) -> None:
    """Ensure the collection_metadata table exists with proper schema.

    This table stores collection metadata including embedding configuration,
    statistics, and configuration settings.

    Args:
        conn: LanceDB connection
    """
    schema = pa.schema(
        [
            pa.field("name", pa.string()),
            pa.field("schema_version", pa.string()),
            pa.field("embedding_model_id", pa.string()),
            pa.field("embedding_dimension", pa.int32()),
            pa.field("documents", pa.int32()),
            pa.field("processed_documents", pa.int32()),
            pa.field("parses", pa.int32()),
            pa.field("chunks", pa.int32()),
            pa.field("embeddings", pa.int32()),
            pa.field("document_names", pa.string()),
            pa.field("collection_locked", pa.bool_()),
            pa.field("allow_mixed_parse_methods", pa.bool_()),
            pa.field("skip_config_validation", pa.bool_()),
            pa.field("ingestion_config", pa.string()),
            pa.field("created_at", pa.timestamp("us")),
            pa.field("updated_at", pa.timestamp("us")),
            pa.field("last_accessed_at", pa.timestamp("us")),
            pa.field("extra_metadata", pa.string()),
        ]
    )
    _create_table(conn, "collection_metadata", schema=schema)
