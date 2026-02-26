"""Main entry point for document parsing.

This module provides the main parse_document function that orchestrates
document parsing by calling the unified document parsing tool.
"""

import asyncio
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional

import pandas as pd

from ......core.tools.core.document_parser import (
    DocumentCapabilities,
    DocumentParseArgs,
)
from ......core.tools.core.document_parser import parse_document as core_parse_document
from ......providers.vector_store.lancedb import get_connection_from_env
from ..core.exceptions import (
    ConfigurationError,
    DatabaseOperationError,
    DocumentNotFoundError,
    DocumentValidationError,
)
from ..core.schemas import (
    ParseDocumentRequest,
    ParseDocumentResponse,
    ParsedParagraph,
    ParseMethod,
)
from ..LanceDB.schema_manager import ensure_documents_table, ensure_parses_table
from ..utils.hash_utils import compute_parse_hash, get_parse_params_whitelist
from ..utils.lancedb_query_utils import query_to_list
from ..utils.string_utils import build_lancedb_filter_expression
from ..utils.user_permissions import UserPermissions

logger = logging.getLogger(__name__)


def parse_document(
    collection: str,
    doc_id: str,
    parse_method: ParseMethod,
    params: Optional[Dict[str, Any]] = None,
    user_id: Optional[int] = None,
    is_admin: bool = False,
    progress_callback: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Parse a document using the specified method.

    Args:
        collection: Collection name for data isolation
        doc_id: Document ID to parse
        parse_method: Parsing method to use
        params: Optional parameters for parsing
        user_id: Optional user ID for ownership tracking
        is_admin: Whether the user has admin privileges
        progress_callback: Optional callback for progress updates

    Returns:
        Dictionary containing parse results
    """
    if params is None:
        params = {}

    request = ParseDocumentRequest(
        collection=collection,
        doc_id=doc_id,
        parse_method=parse_method,
        params=params,
        user_id=user_id,
        is_admin=is_admin,
    )

    response = asyncio.run(_parse_document_internal(request, progress_callback))

    return response.model_dump()


async def _parse_document_internal(
    request: ParseDocumentRequest,
    progress_callback: Optional[Any] = None,
) -> ParseDocumentResponse:
    """
    Internal document parsing logic.
    """
    # Enable detailed timing (controlled by environment variable)
    enable_timing = os.environ.get("PARSE_DETAILED_TIMING", "0").lower() in (
        "1",
        "true",
        "yes",
    )
    timing_data: Optional[Dict[str, float]] = {} if enable_timing else None

    if enable_timing:
        assert timing_data is not None  # Type guard for mypy
        timing_data["start"] = time.perf_counter()
        logger.debug("\n" + "=" * 60)
        logger.debug(
            f"[PARSE TIMING] Starting document parsing: doc_id={request.doc_id}"
        )
        logger.debug("=" * 60)

    collection = request.collection
    doc_id = request.doc_id
    parse_method = request.parse_method
    params = request.params or {}
    user_id = request.user_id
    is_admin = request.is_admin

    logger.info(f"Starting document parsing: doc_id={doc_id}, method={parse_method}")

    try:
        conn = get_connection_from_env()
        ensure_parses_table(conn)
        ensure_documents_table(conn)
    except Exception as e:
        raise DatabaseOperationError(f"Failed to connect to database: {e}") from e

    document = _get_document_from_db(collection, doc_id, user_id, is_admin)
    if not document:
        raise DocumentNotFoundError(f"Document not found: {doc_id}")

    source_path = document["source_path"]
    file_type = document["file_type"]
    logger.info(f"Found document: {source_path}")

    _validate_parse_params(parse_method, params)

    parse_hash = compute_parse_hash(str(parse_method), params)
    logger.info(f"Computed parse hash: {parse_hash}")

    if _parse_exists(collection, doc_id, parse_hash, user_id, is_admin):
        existing_paragraphs = _get_existing_parse_content(
            collection, doc_id, parse_hash, user_id, is_admin
        )
        logger.info(
            f"Parse record already exists for doc_id={doc_id}, parse_hash={parse_hash}"
        )
        return ParseDocumentResponse(
            doc_id=doc_id,
            parse_hash=parse_hash,
            paragraphs=existing_paragraphs,
            written=False,
        )

    # --- Refactored Parsing Logic ---
    try:
        # 1. Call the unified core document parser
        # If parse_method is DEFAULT, use None to let the parser auto-route based on file type
        if parse_method == ParseMethod.DEFAULT:
            parser_name = None  # Let auto-router decide based on file extension
        else:
            parser_name = str(parse_method)

        # Merge params with doc_id for parsers that need it (e.g., deepdoc for PDF images)
        parse_params = {**(params or {}), "doc_id": doc_id}
        tool_args = DocumentParseArgs(
            file_path=source_path,
            parser_name=parser_name,
            # This uses default capabilities, can be expanded to take from params
            capabilities=DocumentCapabilities(),
            parser_kwargs=parse_params,
        )

        if enable_timing:
            assert timing_data is not None  # Type guard for mypy
            timing_data["ocr_start"] = time.perf_counter()
            logger.debug("[PARSE TIMING] Starting OCR processing...")

        parse_result = await core_parse_document(tool_args, progress_callback)

        if enable_timing:
            assert timing_data is not None  # Type guard for mypy
            timing_data["ocr_end"] = time.perf_counter()
            ocr_time = timing_data["ocr_end"] - timing_data["ocr_start"]
            logger.debug(
                f"[PARSE TIMING] OCR processing completed: {ocr_time:.3f} seconds"
            )

        # 2. Convert the rich ParseResult back to the RAG pipeline's ParsedParagraph list
        if enable_timing:
            assert timing_data is not None  # Type guard for mypy
            timing_data["convert_start"] = time.perf_counter()
            logger.debug(
                "[PARSE TIMING] Starting conversion of ParseResult to Paragraphs..."
            )

        paragraphs = _convert_parse_result_to_paragraphs(parse_result)

        if enable_timing:
            assert timing_data is not None  # Type guard for mypy
            timing_data["convert_end"] = time.perf_counter()
            convert_time = timing_data["convert_end"] - timing_data["convert_start"]
            logger.debug(
                f"[PARSE TIMING] Conversion completed: {convert_time:.3f} seconds (paragraphs={len(paragraphs)})"
            )

    except Exception as e:
        logger.error(f"Document parsing failed: {e}")
        raise DocumentValidationError(f"Parsing failed: {e}") from e

    # --- End of Refactored Logic ---

    if enable_timing:
        assert timing_data is not None  # Type guard for mypy
        timing_data["enrich_start"] = time.perf_counter()
        logger.debug("[PARSE TIMING] Starting metadata enrichment...")

    enriched_paragraphs = []
    for paragraph in paragraphs:
        # Start with parser metadata, then override with authoritative values
        enriched_metadata = {
            **paragraph.metadata,
            "source": source_path,
            "file_type": file_type,  # Use file_type from database (without dot)
            "parse_method": str(parse_method),
            "parser": f"local:{parse_method}@v1.0.0",
        }
        enriched_paragraphs.append(
            ParsedParagraph(text=paragraph.text, metadata=enriched_metadata)
        )

    if enable_timing:
        assert timing_data is not None  # Type guard for mypy
        timing_data["enrich_end"] = time.perf_counter()
        enrich_time = timing_data["enrich_end"] - timing_data["enrich_start"]
        logger.debug(
            f"[PARSE TIMING] Metadata enrichment completed: {enrich_time:.3f} seconds (paragraphs={len(enriched_paragraphs)})"
        )

    if enable_timing:
        assert timing_data is not None  # Type guard for mypy
        timing_data["db_write_start"] = time.perf_counter()
        logger.debug("[PARSE TIMING] Starting database write...")

    try:
        written = _write_parse_to_db(
            collection,
            doc_id,
            parse_hash,
            str(parse_method),
            params,
            enriched_paragraphs,
            user_id,
        )
    except Exception as e:
        raise DatabaseOperationError(f"Database write failed: {e}") from e

    if enable_timing:
        assert timing_data is not None  # Type guard for mypy
        timing_data["db_write_end"] = time.perf_counter()
        db_write_time = timing_data["db_write_end"] - timing_data["db_write_start"]
        logger.debug(
            f"[PARSE TIMING] Database write completed: {db_write_time:.3f} seconds"
        )

    logger.info(
        f"Document parsing completed: doc_id={doc_id}, paragraphs={len(enriched_paragraphs)}"
    )

    if enable_timing:
        assert timing_data is not None  # Type guard for mypy
        timing_data["end"] = time.perf_counter()
        total_time = timing_data["end"] - timing_data["start"]

        # Calculate time spent in each stage
        ocr_time = timing_data.get("ocr_end", timing_data["end"]) - timing_data.get(
            "ocr_start", timing_data["start"]
        )
        convert_time = timing_data.get(
            "convert_end", timing_data.get("ocr_end", timing_data["end"])
        ) - timing_data.get(
            "convert_start", timing_data.get("ocr_end", timing_data["start"])
        )
        enrich_time = timing_data.get(
            "enrich_end", timing_data.get("convert_end", timing_data["end"])
        ) - timing_data.get(
            "enrich_start", timing_data.get("convert_end", timing_data["start"])
        )
        db_write_time = timing_data.get(
            "db_write_end", timing_data["end"]
        ) - timing_data.get(
            "db_write_start", timing_data.get("enrich_end", timing_data["end"])
        )

        logger.debug("\n" + "=" * 60)
        logger.debug("[PARSE TIMING] Document parsing time breakdown")
        logger.debug("=" * 60)
        logger.debug(f"  Total time: {total_time:.3f} seconds")
        logger.debug(
            f"  - OCR processing: {ocr_time:.3f} seconds ({ocr_time / total_time * 100:.1f}%)"
        )
        logger.debug(
            f"  - Data conversion: {convert_time:.3f} seconds ({convert_time / total_time * 100:.1f}%)"
        )
        logger.debug(
            f"  - Metadata enrichment: {enrich_time:.3f} seconds ({enrich_time / total_time * 100:.1f}%)"
        )
        logger.debug(
            f"  - Database write: {db_write_time:.3f} seconds ({db_write_time / total_time * 100:.1f}%)"
        )
        logger.debug("=" * 60 + "\n")

    return ParseDocumentResponse(
        doc_id=doc_id,
        parse_hash=parse_hash,
        paragraphs=enriched_paragraphs,
        written=written,
    )


def _convert_parse_result_to_paragraphs(result: Any) -> List[ParsedParagraph]:
    """Converts a ParseResult object into a list of ParsedParagraphs."""
    paragraphs = []
    if result.text_segments:
        for seg in result.text_segments:
            paragraphs.append(ParsedParagraph(text=seg.text, metadata=seg.metadata))
    if result.tables:
        for tbl in result.tables:
            # Use HTML content as text for tables
            text = tbl.html or ""
            paragraphs.append(ParsedParagraph(text=text, metadata=tbl.metadata))
    if result.figures:
        for fig in result.figures:
            # Use caption as text for figures
            paragraphs.append(ParsedParagraph(text=fig.text, metadata=fig.metadata))
    return paragraphs


def _get_document_from_db(
    collection: str, doc_id: str, user_id: Optional[int] = None, is_admin: bool = False
) -> Optional[Any]:
    """Get document from database by doc_id."""
    try:
        conn = get_connection_from_env()
        ensure_documents_table(conn)
        table = conn.open_table("documents")
        query_filters = {"collection": collection, "doc_id": doc_id}
        base_filter_expr = build_lancedb_filter_expression(query_filters)
        user_filter_expr = UserPermissions.get_user_filter(user_id, is_admin)

        if user_filter_expr and base_filter_expr:
            filter_expr = f"({base_filter_expr}) and ({user_filter_expr})"
        elif user_filter_expr:
            filter_expr = user_filter_expr
        else:
            filter_expr = base_filter_expr

        if table.count_rows(filter_expr) == 0:
            return None

        # OPTIMIZATION: Use unified query_to_list() with three-tier fallback
        records = query_to_list(table.search().where(filter_expr))
        if not records:
            return None
        return records[0]

    except Exception as e:
        logger.error(f"Failed to get document from database: {e}")
        raise DatabaseOperationError(f"Failed to get document: {e}") from e


def _validate_parse_params(parse_method: ParseMethod, params: Dict[str, Any]) -> None:
    """Validate parsing parameters against whitelist."""
    valid_methods = set(ParseMethod)
    if parse_method not in valid_methods:
        raise DocumentValidationError(f"Unsupported parse method: {parse_method}")
    try:
        whitelist = get_parse_params_whitelist(str(parse_method))
        for key in params:
            if key not in whitelist:
                raise DocumentValidationError(
                    f"Invalid parameter '{key}' for parse method '{parse_method}'"
                )
    except Exception as e:
        if isinstance(e, DocumentValidationError):
            raise
        raise ConfigurationError(f"Parameter validation failed: {e}") from e


def _parse_exists(
    collection: str,
    doc_id: str,
    parse_hash: str,
    user_id: Optional[int] = None,
    is_admin: bool = False,
) -> bool:
    """Check if parse record already exists.

    Args:
        collection: Collection name
        doc_id: Document ID
        parse_hash: Parse hash to check
        user_id: Optional user ID for filtering (for multi-tenancy)
        is_admin: Whether user has admin privileges

    Returns:
        True if parse record exists and is accessible to the user
    """
    try:
        conn = get_connection_from_env()
        table = conn.open_table("parses")
        query_filters = {
            "collection": collection,
            "doc_id": doc_id,
            "parse_hash": parse_hash,
        }
        base_filter_expr = build_lancedb_filter_expression(query_filters)
        user_filter_expr = UserPermissions.get_user_filter(user_id, is_admin)

        # Combine filters for multi-tenancy
        if user_filter_expr and base_filter_expr:
            filter_expr = f"({base_filter_expr}) and ({user_filter_expr})"
        elif user_filter_expr:
            filter_expr = user_filter_expr
        else:
            filter_expr = base_filter_expr

        return bool(table.count_rows(filter_expr) > 0)
    except Exception as e:
        raise DatabaseOperationError(f"Database query failed: {e}") from e


def _get_existing_parse_content(
    collection: str,
    doc_id: str,
    parse_hash: str,
    user_id: Optional[int] = None,
    is_admin: bool = False,
) -> List[ParsedParagraph]:
    """Get existing parse content from database.

    Args:
        collection: Collection name
        doc_id: Document ID
        parse_hash: Parse hash to retrieve
        user_id: Optional user ID for filtering (for multi-tenancy)
        is_admin: Whether user has admin privileges

    Returns:
        List of parsed paragraphs if found and accessible, empty list otherwise
    """
    try:
        conn = get_connection_from_env()
        table = conn.open_table("parses")
        query_filters = {
            "collection": collection,
            "doc_id": doc_id,
            "parse_hash": parse_hash,
        }
        base_filter_expr = build_lancedb_filter_expression(query_filters)
        user_filter_expr = UserPermissions.get_user_filter(user_id, is_admin)

        # Combine filters for multi-tenancy
        if user_filter_expr and base_filter_expr:
            filter_expr = f"({base_filter_expr}) and ({user_filter_expr})"
        elif user_filter_expr:
            filter_expr = user_filter_expr
        else:
            filter_expr = base_filter_expr

        if table.count_rows(filter_expr) == 0:
            return []

        # OPTIMIZATION: Use unified query_to_list() with three-tier fallback
        records = query_to_list(table.search().where(filter_expr))
        if not records:
            return []
        record = records[0]

        parsed_content = record.get("parsed_content")
        if not parsed_content:
            return []

        data = json.loads(parsed_content)
        paragraphs = []
        for item in data:
            paragraphs.append(
                ParsedParagraph(
                    text=item.get("text", ""),
                    metadata=item.get("metadata", {}),
                )
            )
        return paragraphs
    except Exception as e:
        logger.error(f"Failed to read parse content: {e}")
        raise DatabaseOperationError(f"Failed reading parse content: {e}") from e


def _write_parse_to_db(
    collection: str,
    doc_id: str,
    parse_hash: str,
    parse_method: str,
    params: Dict[str, Any],
    paragraphs: List[ParsedParagraph],
    user_id: Optional[int] = None,
) -> bool:
    """Write parse record to database."""
    enable_timing = os.environ.get("PARSE_DETAILED_TIMING", "0").lower() in (
        "1",
        "true",
        "yes",
    )

    try:
        conn = get_connection_from_env()
        table = conn.open_table("parses")

        if enable_timing:
            serialize_start = time.perf_counter()
            logger.debug(
                f"[PARSE TIMING]    - Starting serialization of paragraphs ({len(paragraphs)} items)..."
            )

        paragraphs_data = [para.model_dump() for para in paragraphs]

        if enable_timing:
            serialize_end = time.perf_counter()
            serialize_time = serialize_end - serialize_start
            logger.debug(
                f"[PARSE TIMING]    - Serialization completed: {serialize_time:.3f} seconds"
            )
            json_start = time.perf_counter()
            logger.debug("[PARSE TIMING]    - Starting JSON serialization...")

        parsed_content = json.dumps(paragraphs_data, ensure_ascii=False)

        if enable_timing:
            json_end = time.perf_counter()
            json_time = json_end - json_start
            json_size_mb = len(parsed_content.encode("utf-8")) / (1024 * 1024)
            logger.debug(
                f"[PARSE TIMING]    - JSON serialization completed: {json_time:.3f} seconds (size: {json_size_mb:.2f} MB)"
            )
            db_op_start = time.perf_counter()
            logger.debug(
                "[PARSE TIMING]    - Starting database operation (merge_insert)..."
            )

        parse_record = {
            "collection": collection,
            "doc_id": doc_id,
            "parse_hash": parse_hash,
            "parser": f"local:{parse_method}@v1.0.0",
            "created_at": pd.Timestamp.now(tz="UTC"),
            "params_json": json.dumps(params, ensure_ascii=False),
            "parsed_content": parsed_content,
            "user_id": user_id,  # Add user_id for multi-tenancy
        }
        table.merge_insert(
            ["collection", "doc_id", "parse_hash"]
        ).when_matched_update_all().when_not_matched_insert_all().execute(
            [parse_record]
        )

        if enable_timing:
            db_op_end = time.perf_counter()
            db_op_time = db_op_end - db_op_start
            logger.debug(
                f"[PARSE TIMING]    - Database operation completed: {db_op_time:.3f} seconds"
            )

        logger.info(
            f"Parse record written to database: doc_id={doc_id}, parse_hash={parse_hash}"
        )
        return True
    except Exception as e:
        raise DatabaseOperationError(f"Database write failed: {e}") from e
