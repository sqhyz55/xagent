"""Core prompt manager for CRUD operations and version management.

This module provides functions for managing prompt templates
with full CRUD operations and transparent version management using LanceDB.
"""

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

from ..core.exceptions import (
    ConfigurationError,
    DatabaseOperationError,
    DocumentNotFoundError,
)
from ..core.schemas import PromptTemplate
from ..LanceDB.schema_manager import ensure_prompt_templates_table
from ..storage.factory import get_metadata_store
from ..utils.string_utils import escape_lancedb_string

logger = logging.getLogger(__name__)


def _serialize_metadata(metadata: Optional[Dict[str, Any]]) -> Optional[str]:
    """Serialize metadata dictionary to JSON string.

    Args:
        metadata: Metadata dictionary to serialize.

    Returns:
        JSON string or None.
    """
    if metadata is None:
        return None
    return json.dumps(metadata, ensure_ascii=False, sort_keys=True)


def _deserialize_metadata(metadata_json: Optional[str]) -> Optional[Dict[str, Any]]:
    """Deserialize metadata JSON string to dictionary.

    Args:
        metadata_json: JSON string to deserialize.

    Returns:
        Metadata dictionary or None.
    """
    if metadata_json is None or pd.isna(metadata_json):
        return None
    result: Dict[str, Any] = json.loads(metadata_json)
    return result


def _get_prompt_table() -> Any:
    """Get LanceDB table for prompt templates.

    Returns:
        LanceDB table instance.

    Raises:
        DatabaseOperationError: If table access fails.
    """
    try:
        db = get_metadata_store().get_raw_connection()
        table_name = "prompt_templates"

        # Ensure table exists with proper schema
        ensure_prompt_templates_table(db)

        # Open and return the table
        return db.open_table(table_name)

    except Exception as e:
        logger.error(f"Failed to get prompt templates table: {str(e)}")
        raise DatabaseOperationError(
            f"Failed to access prompt templates table: {str(e)}"
        ) from e


# ------------------------- Public Functions -------------------------


def create_prompt_template(
    collection: str,
    name: str,
    template: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> PromptTemplate:
    """Create a new prompt template or new version.

    Args:
        collection: Collection name for data isolation.
        name: Human-readable name for the prompt template.
        template: The actual prompt template content.
        metadata: Optional metadata dictionary.

    Returns:
        Created PromptTemplate instance.

    Raises:
        ConfigurationError: If collection, name or template is empty.
        DatabaseOperationError: If database operation fails.
    """
    if not collection:
        raise ConfigurationError("Collection name cannot be empty.")
    if not name or not name.strip():
        raise ConfigurationError("Prompt template name cannot be empty.")
    if not template or not template.strip():
        raise ConfigurationError("Prompt template content cannot be empty.")

    # Normalize name
    name = name.strip()

    try:
        table = _get_prompt_table()

        # Check if a template with this name already exists (using safe filter)
        # Filter by both collection and name
        escaped_collection = escape_lancedb_string(collection)
        escaped_name = escape_lancedb_string(name)
        collection_name_filter = (
            f"collection == '{escaped_collection}' AND name == '{escaped_name}'"
        )
        existing_templates = table.search().where(collection_name_filter).to_pandas()

        if existing_templates.empty:
            # Create first version
            version = 1
            is_latest = True
        else:
            # Create new version - find the highest version number
            max_version = existing_templates["version"].max()
            version = max_version + 1
            is_latest = True

            # Mark all previous versions as not latest
            table.update(where=collection_name_filter, values={"is_latest": False})

        # Create new prompt template
        prompt_template = PromptTemplate(
            name=name,
            template=template.strip(),
            version=version,
            is_latest=is_latest,
            metadata=_serialize_metadata(metadata),
        )

        # Convert to DataFrame for LanceDB insertion, including collection
        template_dict = prompt_template.model_dump()
        template_dict["collection"] = collection
        df = pd.DataFrame([template_dict])
        table.add(df)

        logger.info(f"Created prompt template '{name}' version {version}")
        return prompt_template

    except (ConfigurationError, DatabaseOperationError):
        raise
    except Exception as e:
        logger.error(f"Failed to create prompt template '{name}': {str(e)}")
        raise DatabaseOperationError(
            f"Failed to create prompt template: {str(e)}"
        ) from e


def read_prompt_template(
    collection: str,
    prompt_id: Optional[str] = None,
    name: Optional[str] = None,
    version: Optional[int] = None,
) -> PromptTemplate:
    """Read a specific prompt template.

    Args:
        collection: Collection name for data isolation.
        prompt_id: UUID of the prompt template.
        name: Name of the prompt template.
        version: Specific version number (if not provided, latest is returned).

    Returns:
        PromptTemplate instance.

    Raises:
        ConfigurationError: If collection is empty or neither prompt_id nor name is provided.
        DocumentNotFoundError: If prompt template is not found.
        DatabaseOperationError: If database operation fails.
    """
    if not collection:
        raise ConfigurationError("Collection name cannot be empty.")
    if not prompt_id and not name:
        raise ConfigurationError("Either prompt_id or name must be provided.")

    try:
        table = _get_prompt_table()
        escaped_collection = escape_lancedb_string(collection)

        if prompt_id:
            # Search by ID and collection (using safe filter)
            escaped_id = escape_lancedb_string(prompt_id)
            id_filter = f"collection == '{escaped_collection}' AND id == '{escaped_id}'"
            result = table.search().where(id_filter).to_pandas()
        else:
            # Normalize name
            name = name.strip() if name else name
            # Search by name and collection
            escaped_name = escape_lancedb_string(name)
            if version is not None:
                # Specific version - combine filters safely
                filter_expr = (
                    f"collection == '{escaped_collection}' AND "
                    f"name == '{escaped_name}' AND version == {version}"
                )
                result = table.search().where(filter_expr).to_pandas()
            else:
                # Latest version - combine filters safely
                filter_expr = (
                    f"collection == '{escaped_collection}' AND "
                    f"name == '{escaped_name}' AND is_latest == true"
                )
                result = table.search().where(filter_expr).to_pandas()

        if result.empty:
            identifier = (
                f"ID '{prompt_id}'"
                if prompt_id
                else f"name '{name}'"
                + (f" version {version}" if version else " (latest)")
            )
            raise DocumentNotFoundError(f"Prompt template with {identifier} not found.")

        # Convert to PromptTemplate
        row = result.iloc[0]
        # Note: metadata is stored as JSON string internally, keep it as is
        return PromptTemplate(
            id=str(row["id"]),
            name=row["name"],
            template=row["template"],
            version=int(row["version"]),
            is_latest=bool(row["is_latest"]),
            metadata=row["metadata"] if pd.notna(row["metadata"]) else None,
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    except (ConfigurationError, DocumentNotFoundError):
        raise
    except Exception as e:
        logger.error(f"Failed to read prompt template: {str(e)}")
        raise DatabaseOperationError(f"Failed to read prompt template: {str(e)}") from e


def get_latest_prompt_template(collection: str, name: str) -> PromptTemplate:
    """Get the latest version of a prompt template by name.

    Args:
        collection: Collection name for data isolation.
        name: Name of the prompt template.

    Returns:
        Latest PromptTemplate instance.

    Raises:
        ConfigurationError: If collection or name is empty.
        DocumentNotFoundError: If prompt template is not found.
        DatabaseOperationError: If database operation fails.
    """
    if not collection:
        raise ConfigurationError("Collection name cannot be empty.")
    if not name or not name.strip():
        raise ConfigurationError("Prompt template name cannot be empty.")

    return read_prompt_template(collection=collection, name=name.strip())


def update_prompt_template(
    collection: str,
    prompt_id: str,
    template: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> PromptTemplate:
    """Update a prompt template.

    If template content is changed, creates a new version.
    If only metadata is changed, updates the current version.

    Args:
        collection: Collection name for data isolation.
        prompt_id: UUID of the prompt template to update.
        template: New template content (creates new version if provided).
        metadata: New metadata (updates current version if provided).

    Returns:
        Updated PromptTemplate instance.

    Raises:
        ConfigurationError: If collection or prompt_id is empty.
        DocumentNotFoundError: If prompt template is not found.
        DatabaseOperationError: If database operation fails.
    """
    if not collection:
        raise ConfigurationError("Collection name cannot be empty.")
    if not prompt_id:
        raise ConfigurationError("Prompt ID must be provided.")

    if template is None and metadata is None:
        raise ConfigurationError(
            "Either template or metadata must be provided for update."
        )

    try:
        # First, get the current template
        current_template = read_prompt_template(
            collection=collection, prompt_id=prompt_id
        )
        table = _get_prompt_table()
        escaped_collection = escape_lancedb_string(collection)

        if template is not None:
            # Template content changed - create new version
            if not template.strip():
                raise ConfigurationError("Template content cannot be empty.")

            # Find the highest version number for this name to avoid version conflicts
            escaped_name = escape_lancedb_string(current_template.name)
            name_filter = (
                f"collection == '{escaped_collection}' AND name == '{escaped_name}'"
            )
            all_versions = table.search().where(name_filter).to_pandas()
            max_version = all_versions["version"].max() if not all_versions.empty else 0
            new_version = max_version + 1

            # Mark all previous versions as not latest
            table.update(where=name_filter, values={"is_latest": False})

            # Create new version
            # Serialize the new metadata if provided, otherwise use current template's metadata
            new_metadata = (
                _serialize_metadata(metadata)
                if metadata is not None
                else current_template.metadata
            )
            updated_template = PromptTemplate(
                name=current_template.name,
                template=template.strip(),
                version=new_version,
                is_latest=True,
                metadata=new_metadata,
            )

            # Insert new version, including collection
            template_dict = updated_template.model_dump()
            template_dict["collection"] = collection
            df = pd.DataFrame([template_dict])
            table.add(df)

            logger.info(
                f"Created new version {new_version} for prompt template '{current_template.name}'"
            )
            return updated_template

        else:
            # Only metadata changed - update current version
            metadata_json = _serialize_metadata(metadata)
            updated_template = PromptTemplate(
                id=current_template.id,
                name=current_template.name,
                template=current_template.template,
                version=current_template.version,
                is_latest=current_template.is_latest,
                metadata=metadata_json,
                created_at=current_template.created_at,
                updated_at=datetime.utcnow(),
            )

            # Update the existing record (using safe filter with collection)
            escaped_id = escape_lancedb_string(prompt_id)
            id_filter = f"collection == '{escaped_collection}' AND id == '{escaped_id}'"
            table.update(
                where=id_filter,
                values={
                    "metadata": metadata_json,
                    "updated_at": updated_template.updated_at,
                },
            )

            logger.info(
                f"Updated metadata for prompt template '{current_template.name}' version {current_template.version}"
            )
            return updated_template

    except (ConfigurationError, DocumentNotFoundError):
        raise
    except Exception as e:
        logger.error(f"Failed to update prompt template {prompt_id}: {str(e)}")
        raise DatabaseOperationError(
            f"Failed to update prompt template: {str(e)}"
        ) from e


def delete_prompt_template(
    collection: str,
    prompt_id: Optional[str] = None,
    name: Optional[str] = None,
    version: Optional[int] = None,
) -> bool:
    """Delete a prompt template or specific version.

    Args:
        collection: Collection name for data isolation.
        prompt_id: UUID of the prompt template to delete.
        name: Name of the prompt template to delete.
        version: Specific version to delete (if not provided, all versions are deleted).

    Returns:
        True if deletion was successful.

    Raises:
        ConfigurationError: If collection is empty or neither prompt_id nor name is provided.
        DocumentNotFoundError: If prompt template is not found.
        DatabaseOperationError: If database operation fails.
    """
    if not collection:
        raise ConfigurationError("Collection name cannot be empty.")
    if not prompt_id and not name:
        raise ConfigurationError("Either prompt_id or name must be provided.")

    try:
        table = _get_prompt_table()
        escaped_collection = escape_lancedb_string(collection)

        if prompt_id:
            # Delete specific template by ID and collection (using safe filter)
            escaped_id = escape_lancedb_string(prompt_id)
            id_filter = f"collection == '{escaped_collection}' AND id == '{escaped_id}'"
            result = table.search().where(id_filter).to_pandas()
            if result.empty:
                raise DocumentNotFoundError(
                    f"Prompt template with ID '{prompt_id}' not found."
                )

            # Check if this was the latest version and get the name
            was_latest = result.iloc[0]["is_latest"]
            template_name = result.iloc[0]["name"]

            table.delete(id_filter)

            # If we deleted the latest version, update the latest flag for the remaining versions
            if was_latest:
                escaped_name = escape_lancedb_string(template_name)
                name_filter = (
                    f"collection == '{escaped_collection}' AND name == '{escaped_name}'"
                )
                remaining_versions = table.search().where(name_filter).to_pandas()
                if not remaining_versions.empty:
                    max_version = remaining_versions["version"].max()
                    update_filter = (
                        f"collection == '{escaped_collection}' AND "
                        f"name == '{escaped_name}' AND version == {max_version}"
                    )
                    table.update(where=update_filter, values={"is_latest": True})

            logger.info(f"Deleted prompt template with ID '{prompt_id}'")
            return True

        else:
            # Normalize name
            name = name.strip() if name else name
            escaped_name = escape_lancedb_string(name)
            # Delete by name and collection
            if version is not None:
                # Delete specific version
                version_filter = (
                    f"collection == '{escaped_collection}' AND "
                    f"name == '{escaped_name}' AND version == {version}"
                )
                result = table.search().where(version_filter).to_pandas()
                if result.empty:
                    raise DocumentNotFoundError(
                        f"Prompt template '{name}' version {version} not found."
                    )

                # Check if this was the latest version
                was_latest = result.iloc[0]["is_latest"]

                table.delete(version_filter)

                # If we deleted the latest version, update the latest flag
                if was_latest:
                    name_filter = (
                        f"collection == '{escaped_collection}' AND "
                        f"name == '{escaped_name}'"
                    )
                    remaining_versions = table.search().where(name_filter).to_pandas()
                    if not remaining_versions.empty:
                        # Find the highest remaining version and mark it as latest
                        max_version = remaining_versions["version"].max()
                        update_filter = (
                            f"collection == '{escaped_collection}' AND "
                            f"name == '{escaped_name}' AND version == {max_version}"
                        )
                        table.update(where=update_filter, values={"is_latest": True})

                logger.info(f"Deleted prompt template '{name}' version {version}")
                return True
            else:
                # Delete all versions
                name_filter = (
                    f"collection == '{escaped_collection}' AND name == '{escaped_name}'"
                )
                result = table.search().where(name_filter).to_pandas()
                if result.empty:
                    raise DocumentNotFoundError(f"Prompt template '{name}' not found.")

                table.delete(name_filter)
                logger.info(f"Deleted all versions of prompt template '{name}'")
                return True

    except (ConfigurationError, DocumentNotFoundError):
        raise
    except Exception as e:
        logger.error(f"Failed to delete prompt template: {str(e)}")
        raise DatabaseOperationError(
            f"Failed to delete prompt template: {str(e)}"
        ) from e


def list_prompt_templates(
    collection: str,
    name_filter: Optional[str] = None,
    latest_only: bool = False,
    metadata_filter: Optional[Dict[str, Any]] = None,
    limit: int = 100,
) -> List[PromptTemplate]:
    """List prompt templates with optional filtering.

    Args:
        collection: Collection name for data isolation.
        name_filter: Filter by name (partial match).
        latest_only: If True, only return latest versions.
        metadata_filter: Filter by metadata fields (not yet implemented).
        limit: Maximum number of results to return (default: 100).

    Returns:
        List of PromptTemplate instances.

    Raises:
        ConfigurationError: If collection is empty.
        DatabaseOperationError: If database operation fails.
    """
    if not collection:
        raise ConfigurationError("Collection name cannot be empty.")

    try:
        table = _get_prompt_table()
        escaped_collection = escape_lancedb_string(collection)

        # Build filter conditions safely, always include collection filter
        filters = [f"collection == '{escaped_collection}'"]

        if name_filter:
            # Use safe escaping for partial match
            escaped_name = escape_lancedb_string(name_filter)
            filters.append(f"name LIKE '%{escaped_name}%'")

        if latest_only:
            filters.append("is_latest == true")

        # Note: metadata filtering would require more complex logic
        # For now, we'll implement basic filtering
        if metadata_filter:
            logger.warning("Metadata filtering is not yet implemented")

        # Combine filters
        where_clause = " AND ".join(filters)
        result = table.search().where(where_clause).limit(limit).to_pandas()

        # Convert to PromptTemplate objects
        templates = []
        for _, row in result.iterrows():
            # Note: metadata is stored as JSON string, keep it as is
            template = PromptTemplate(
                id=str(row["id"]),
                name=row["name"],
                template=row["template"],
                version=int(row["version"]),
                is_latest=bool(row["is_latest"]),
                metadata=row["metadata"] if pd.notna(row["metadata"]) else None,
                created_at=row["created_at"],
                updated_at=row["updated_at"],
            )
            templates.append(template)

        logger.info(f"Listed {len(templates)} prompt templates (limit: {limit})")
        return templates

    except (ConfigurationError, DatabaseOperationError):
        raise
    except Exception as e:
        logger.error(f"Failed to list prompt templates: {str(e)}")
        raise DatabaseOperationError(
            f"Failed to list prompt templates: {str(e)}"
        ) from e
