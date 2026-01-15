"""Functions for displaying parse results with pagination support.

This module provides functions to retrieve and format parse results
from the database for display purposes.
"""

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

from ...providers.vector_store.lancedb import get_connection_from_env
from ..core.exceptions import DatabaseOperationError, DocumentNotFoundError
from ..core.schemas import (
    ParsedFigureDisplay,
    ParsedTableDisplay,
    ParsedTextSegmentDisplay,
)
from ..LanceDB.schema_manager import ensure_parses_table
from ..utils.lancedb_query_utils import query_to_list
from ..utils.string_utils import build_lancedb_filter_expression

logger = logging.getLogger(__name__)


def reconstruct_parse_result_from_db(
    collection: str,
    doc_id: str,
    parse_hash: Optional[str] = None,
) -> Tuple[
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    Optional[str],
]:
    """Reconstruct ParseResult-like structure from database.

    Args:
        collection: Collection name
        doc_id: Document ID
        parse_hash: Optional parse hash to filter. If None, uses the latest parse.

    Returns:
        Tuple of (text_segments, tables, figures, parse_hash)
        Each list contains dictionaries with 'text'/'html' and 'metadata' keys.
    """
    try:
        conn = get_connection_from_env()
        ensure_parses_table(conn)
        table = conn.open_table("parses")

        # Build filter expression
        query_filters = {
            "collection": collection,
            "doc_id": doc_id,
        }
        if parse_hash:
            query_filters["parse_hash"] = parse_hash

        filter_expr = build_lancedb_filter_expression(query_filters)

        if table.count_rows(filter_expr) == 0:
            if parse_hash:
                raise DocumentNotFoundError(
                    f"Parse result not found: doc_id={doc_id}, parse_hash={parse_hash}"
                )
            raise DocumentNotFoundError(
                f"No parse results found for document: doc_id={doc_id}"
            )

        # OPTIMIZATION: Use unified query_to_list() with three-tier fallback
        records = query_to_list(table.search().where(filter_expr))
        if not records:
            raise DocumentNotFoundError(
                f"No parse results found for document: doc_id={doc_id}"
            )

        # If parse_hash not specified, use the latest parse (first record)
        # TODO: Consider sorting by timestamp if available
        record = records[0]
        actual_parse_hash = record.get("parse_hash")

        parsed_content = record.get("parsed_content")
        if not parsed_content:
            logger.warning(f"Empty parsed_content for doc_id={doc_id}")
            return ([], [], [], actual_parse_hash)

        # Parse JSON string with error handling for data corruption
        try:
            data = json.loads(parsed_content)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode parsed_content for doc_id={doc_id}: {e}")
            raise DatabaseOperationError(
                f"Document parse data is corrupted for doc_id={doc_id}"
            )

        # Reconstruct text_segments, tables, figures from ParsedParagraph list
        text_segments = []
        tables = []
        figures = []

        for item in data:
            text = item.get("text", "")
            metadata = item.get("metadata", {})
            layout_type = metadata.get("layout_type", "text")

            if layout_type == "text":
                text_segments.append({"text": text, "metadata": metadata})
            elif layout_type == "table":
                tables.append({"html": text, "metadata": metadata})
            elif layout_type == "figure":
                figures.append({"text": text, "metadata": metadata})
            else:
                # Unknown layout type, treat as text
                logger.debug(f"Unknown layout_type '{layout_type}', treating as text")
                text_segments.append({"text": text, "metadata": metadata})

        logger.info(
            f"Reconstructed parse result: {len(text_segments)} text segments, "
            f"{len(tables)} tables, {len(figures)} figures"
        )

        return (text_segments, tables, figures, actual_parse_hash)

    except DocumentNotFoundError:
        raise
    except Exception as e:
        logger.error(f"Failed to reconstruct parse result: {e}")
        raise DatabaseOperationError(f"Failed to read parse result: {e}") from e


def paginate_parse_results(
    text_segments: List[Dict[str, Any]],
    tables: List[Dict[str, Any]],
    figures: List[Dict[str, Any]],
    page: int = 1,
    page_size: int = 20,
) -> Tuple[
    List[ParsedTextSegmentDisplay],
    List[ParsedTableDisplay],
    List[ParsedFigureDisplay],
    Dict[str, Any],
]:
    """Paginate parse results across all element types.

    Args:
        text_segments: List of text segment dicts
        tables: List of table dicts
        figures: List of figure dicts
        page: Page number (1-indexed)
        page_size: Number of elements per page

    Returns:
        Tuple of (paginated_text_segments, paginated_tables, paginated_figures, pagination_info)
    """
    # Validate inputs
    if page < 1:
        page = 1
    if page_size < 1:
        page_size = 20

    # Combine all elements in order: text_segments, tables, figures
    all_elements = (
        [("text", seg) for seg in text_segments]
        + [("table", tbl) for tbl in tables]
        + [("figure", fig) for fig in figures]
    )

    total_count = len(all_elements)
    total_pages = (total_count + page_size - 1) // page_size if total_count > 0 else 1

    # Calculate pagination
    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size

    # Get paginated elements
    paginated_elements = all_elements[start_idx:end_idx]

    # Separate back into types
    paginated_text_segments = []
    paginated_tables = []
    paginated_figures = []

    for elem_type, elem_data in paginated_elements:
        if elem_type == "text":
            paginated_text_segments.append(
                ParsedTextSegmentDisplay(
                    text=elem_data["text"], metadata=elem_data["metadata"]
                )
            )
        elif elem_type == "table":
            paginated_tables.append(
                ParsedTableDisplay(html=elem_data["html"], metadata=elem_data["metadata"])
            )
        elif elem_type == "figure":
            paginated_figures.append(
                ParsedFigureDisplay(
                    text=elem_data["text"], metadata=elem_data["metadata"]
                )
            )

    pagination_info = {
        "page": page,
        "page_size": page_size,
        "total_elements": total_count,
        "total_pages": total_pages,
        "has_next": page < total_pages,
        "has_previous": page > 1,
        "text_segments_count": len(text_segments),
        "tables_count": len(tables),
        "figures_count": len(figures),
    }

    return (
        paginated_text_segments,
        paginated_tables,
        paginated_figures,
        pagination_info,
    )

