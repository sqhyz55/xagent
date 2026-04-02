"""Utilities for handling schema migrations and backward compatibility."""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple, cast

from ..LanceDB.model_tag_utils import to_model_tag
from ..storage.factory import get_vector_store_raw_connection as get_connection_from_env
from .string_utils import escape_lancedb_string
from .tag_mapping import register_tag_mapping

logger = logging.getLogger(__name__)


def migrate_collection_metadata(
    legacy_data: Dict[str, Any],
    *,
    infer_embedding: bool = True,
) -> Dict[str, Any]:
    """Migrate legacy collection metadata to current schema version.

    Args:
        legacy_data: Legacy collection data from storage
        infer_embedding: If True (default), ``0.0.0 -> 1.0.0`` may scan LanceDB
            embedding tables to infer ``embedding_model_id`` / dimension. Use
            **False** for read-only deserialization (e.g. :meth:`CollectionInfo.from_storage`)
            to avoid I/O, heavy work, and log noise on hot paths.

    Returns:
        Migrated data compatible with current schema
    """
    data = legacy_data.copy()
    current_version = "1.0.0"
    data_version = data.get("schema_version", "0.0.0")
    collection_name = data.get("name", "unknown")

    log_info = logger.info if infer_embedding else logger.debug
    log_info(
        f"[MIGRATION_START] Collection: {collection_name}, From: {data_version}, To: {current_version}"
    )
    logger.debug(
        f"[MIGRATION_SNAPSHOT] Original data for {collection_name}: {legacy_data}"
    )

    try:
        # Apply migrations sequentially
        while data_version < current_version:
            previous_version = data_version
            if data_version == "0.0.0":
                data = _migrate_0_0_0_to_1_0_0(data, infer_embedding=infer_embedding)
                data_version = "1.0.0"

            log_info(
                f"[MIGRATION_STEP] {collection_name}: {previous_version} -> {data_version} completed."
            )

        log_info(
            f"[MIGRATION_SUCCESS] Collection '{collection_name}' is now at version {data_version}"
        )
        return data
    except Exception as e:
        logger.error(
            f"[MIGRATION_FAILED] Collection: {collection_name}, Error: {str(e)}",
            exc_info=True,
        )
        logger.error(f"[MIGRATION_RECOVERY_DATA] {legacy_data}")
        raise


def _migrate_0_0_0_to_1_0_0(
    data: Dict[str, Any],
    *,
    infer_embedding: bool = True,
) -> Dict[str, Any]:
    """Migrate from pre-versioned schema to 1.0.0."""
    collection_name = data.get("name", "")

    if infer_embedding:
        embedding_model_id, embedding_dimension = (
            _infer_embedding_config_from_collection(collection_name)
        )
    else:
        embedding_model_id = data.get("embedding_model_id")
        embedding_dimension = data.get("embedding_dimension")

    if embedding_model_id:
        logger.info(
            f"[MIGRATION_INFERENCE] Inferred embedding model '{embedding_model_id}' "
            f"(dimension: {embedding_dimension}) for collection '{collection_name}'"
        )
    else:
        logger.warning(
            f"[MIGRATION_INFERENCE_FAILED] Could not infer embedding model for collection '{collection_name}', "
            "will use lazy initialization."
        )

    migrated = {
        # Version control
        "schema_version": "1.0.0",
        # Basic fields (preserve existing or set defaults)
        "name": collection_name,
        # Embedding config (inferred from existing data or None for lazy init)
        "embedding_model_id": embedding_model_id,
        "embedding_dimension": embedding_dimension,
        # Statistics (preserve existing or set defaults)
        "documents": data.get("documents", 0),
        "embeddings": data.get("embeddings", 0),
        # Document names
        "document_names": data.get("document_names", []),
        # Configuration (defaults)
        "collection_locked": False,
        "allow_mixed_parse_methods": False,
        "skip_config_validation": False,
        # Timestamps - Use naive UTC
        "created_at": data.get("created_at")
        or datetime.now(timezone.utc).replace(tzinfo=None),
        "updated_at": data.get("updated_at")
        or datetime.now(timezone.utc).replace(tzinfo=None),
        "last_accessed_at": data.get("last_accessed_at")
        or datetime.now(timezone.utc).replace(tzinfo=None),
        # Extensions
        "extra_metadata": data.get("extra_metadata", {}),
    }

    return migrated


def _infer_embedding_config_from_collection(
    collection_name: str,
) -> Tuple[Optional[str], Optional[int]]:
    """Infer embedding model and dimension from existing collection data.

    Args:
        collection_name: Name of the collection

    Returns:
        Tuple of (embedding_model_id, embedding_dimension), or (None, None) if cannot infer
    """
    logger.debug(
        f"Starting embedding config inference for collection '{collection_name}'"
    )

    try:
        # Get LanceDB connection
        logger.debug(f"Connecting to LanceDB for collection '{collection_name}'")
        conn = get_connection_from_env()

        # Get all table names that contain embeddings
        table_names_fn = getattr(conn, "table_names", None)
        if table_names_fn is None:
            logger.info(
                f"LanceDB connection missing table_names() for collection '{collection_name}' - will use lazy initialization"
            )
            return None, None
        all_table_names = table_names_fn()
        if all_table_names is None:
            logger.info(
                f"No table names returned for collection '{collection_name}' - will use lazy initialization"
            )
            return None, None
        table_names = [
            name for name in all_table_names if name.startswith("embeddings_")
        ]
        logger.debug(f"Found {len(table_names)} embedding tables: {table_names}")

        if not table_names:
            logger.info(
                f"No embedding tables found for collection '{collection_name}' - will use lazy initialization"
            )
            return None, None

        # Track model usage by counting vectors
        model_stats = {}

        for table_name in table_names:
            logger.debug(
                f"Checking table '{table_name}' for collection '{collection_name}'"
            )

            try:
                table = conn.open_table(table_name)

                # Count vectors for this collection in this table
                try:
                    # Try to filter by collection if the column exists
                    safe_collection_name = escape_lancedb_string(collection_name)
                    count_result = (
                        table.search()
                        .where(f"collection = '{safe_collection_name}'")
                        .limit(1)
                        .to_pandas()
                    )
                    vector_count = len(count_result) if not count_result.empty else 0
                    logger.debug(
                        f"Table '{table_name}' has {vector_count} vectors for collection '{collection_name}'"
                    )

                    if vector_count == 0:
                        continue

                    # Get model tag from table name
                    model_tag = table_name.replace("embeddings_", "")
                    logger.debug(
                        f"Extracted model tag '{model_tag}' from table '{table_name}'"
                    )

                    # Get dimension from schema
                    dimension = None
                    try:
                        schema = table.schema
                        # Look for vector field dimension
                        for field in schema:
                            if field.name == "vector" and hasattr(
                                field.type, "list_size"
                            ):
                                dimension = field.type.list_size
                                logger.debug(
                                    f"Found vector dimension {dimension} in table '{table_name}'"
                                )
                                break
                    except Exception as e:
                        logger.debug(
                            f"Could not get dimension from table '{table_name}': {e}"
                        )

                    model_stats[model_tag] = {
                        "count": vector_count,
                        "dimension": dimension,
                    }

                except Exception as e:
                    logger.debug(
                        f"Error checking table '{table_name}' for collection '{collection_name}': {e}"
                    )
                    continue

            except Exception as e:
                logger.debug(f"Error opening table '{table_name}': {e}")
                continue

        if not model_stats:
            logger.info(
                f"No vectors found for collection '{collection_name}' in any embedding table - will use lazy initialization"
            )
            return None, None

        logger.info(
            f"Found vectors from {len(model_stats)} different models for collection '{collection_name}': {list(model_stats.keys())}"
        )

        # Choose the most used model
        best_model = max(
            model_stats.items(), key=lambda x: cast(int, x[1].get("count", 0))
        )
        model_tag, stats = best_model

        # Resolve Hub embedding model ID from table tag (preferred).
        embedding_model_id = None
        try:
            from xagent.core.model.model import EmbeddingModelConfig

            from .model_resolver import _get_or_init_model_hub

            hub = _get_or_init_model_hub()
            if hub is not None:
                hub_tag_to_id: Dict[str, str] = {}
                for cfg in hub.list().values():
                    if not isinstance(cfg, EmbeddingModelConfig):
                        continue
                    register_tag_mapping(
                        hub_tag_to_id,
                        to_model_tag(cfg.id),
                        cfg.id,
                        get_identity=lambda item: item,
                        logger=logger,
                    )
                    register_tag_mapping(
                        hub_tag_to_id,
                        to_model_tag(cfg.model_name),
                        cfg.id,
                        get_identity=lambda item: item,
                        logger=logger,
                    )
                embedding_model_id = hub_tag_to_id.get(model_tag)
        except Exception:
            embedding_model_id = None

        # Fallback: best-effort reverse normalization (legacy behavior)
        if not embedding_model_id:
            embedding_model_id = _model_tag_to_model_id(model_tag)
        embedding_dimension = stats["dimension"]

        logger.info(
            f"Selected embedding model '{embedding_model_id}' (dimension: {embedding_dimension}) "
            f"for collection '{collection_name}' based on {stats['count']} vectors"
        )

        if len(model_stats) > 1:
            logger.warning(
                f"Collection '{collection_name}' has vectors from multiple models: {list(model_stats.keys())}. "
                f"Choosing '{embedding_model_id}' (most used with {stats['count']} vectors). "
                f"Consider migrating old vectors to maintain consistency."
            )

        return embedding_model_id, embedding_dimension

    except Exception as e:
        logger.error(
            f"Error inferring embedding config for collection '{collection_name}': {e}",
            exc_info=True,
        )
        return None, None


def _model_tag_to_model_id(model_tag: str) -> str:
    """Convert model tag back to model ID."""
    logger.debug(f"Converting model tag '{model_tag}' back to model ID")

    # Handle common cases
    if model_tag.startswith("OPENAI_"):
        result = model_tag.replace("OPENAI_", "").replace("_", "-").lower()
        logger.debug(f"Converted OpenAI model tag to: {result}")
        return result
    elif model_tag.startswith("BAAI_"):
        result = model_tag.replace("BAAI_", "").replace("_", "-").lower()
        logger.debug(f"Converted BAAI model tag to: {result}")
        return result
    else:
        # Fallback: try to reverse the normalization
        import re

        match = re.match(r"^([A-Z]+)_(.+)$", model_tag)
        if match:
            result = match.group(2).replace("_", "-").lower()
        else:
            result = model_tag.replace("_", "-").lower()
        logger.debug(f"Used fallback conversion for model tag: {result}")
        return result
