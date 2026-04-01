"""Helpers for bridging KB document metadata and uploaded file records."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from ...core.tools.core.RAG_tools.LanceDB.schema_manager import ensure_documents_table
from ...core.tools.core.RAG_tools.utils.lancedb_query_utils import query_to_list
from ...core.tools.core.RAG_tools.utils.string_utils import (
    build_lancedb_filter_expression,
)
from ...core.tools.core.RAG_tools.utils.user_permissions import UserPermissions
from ...providers.vector_store.lancedb import get_connection_from_env
from ..config import UPLOADS_DIR
from ..models.uploaded_file import UploadedFile

logger = logging.getLogger(__name__)


def upsert_uploaded_file_record(
    db: Session,
    *,
    user_id: int,
    filename: str,
    storage_path: Path,
    mime_type: Optional[str],
    file_size: int,
) -> UploadedFile:
    """Create or refresh an ``UploadedFile`` row for a stored file."""
    storage_path_str = str(storage_path)
    existing = (
        db.query(UploadedFile)
        .filter(UploadedFile.storage_path == storage_path_str)
        .first()
    )
    if existing:
        existing.filename = filename  # type: ignore[assignment]
        existing.file_size = int(file_size)  # type: ignore[assignment]
        if mime_type is not None:
            existing.mime_type = mime_type  # type: ignore[assignment]
        db.flush()
        file_record = existing
    else:
        file_record = UploadedFile(
            user_id=user_id,
            filename=filename,
            storage_path=storage_path_str,
            mime_type=mime_type,
            file_size=int(file_size),
        )
        db.add(file_record)
        db.flush()
    db.commit()
    db.refresh(file_record)
    return file_record


def list_documents_for_user(
    *,
    user_id: int,
    is_admin: bool,
    collection_name: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Load KB document metadata rows for a user."""
    conn = get_connection_from_env()
    ensure_documents_table(conn)
    table = conn.open_table("documents")

    base_filter = ""
    if collection_name:
        base_filter = build_lancedb_filter_expression({"collection": collection_name})
    user_filter = UserPermissions.get_user_filter(user_id, is_admin=is_admin)
    combined_filter = (
        f"({base_filter}) and ({user_filter})"
        if user_filter and base_filter
        else (user_filter or base_filter)
    )
    query = table.search()
    if combined_filter:
        query = query.where(combined_filter)
    return query_to_list(query.limit(10000))


def build_uploaded_filename_map(
    db: Session, *, user_id: int, file_ids: List[str]
) -> Dict[str, str]:
    """Resolve ``file_id`` values to current uploaded filenames."""
    normalized_file_ids = sorted({file_id for file_id in file_ids if file_id})
    if not normalized_file_ids:
        return {}
    records = (
        db.query(UploadedFile)
        .filter(
            UploadedFile.user_id == user_id,
            UploadedFile.file_id.in_(normalized_file_ids),
        )
        .all()
    )
    return {str(record.file_id): str(record.filename) for record in records}


def get_document_record_file_id(record: Dict[str, Any]) -> Optional[str]:
    """Extract a normalized ``file_id`` from a KB document record."""
    raw_file_id = record.get("file_id")
    if raw_file_id is None:
        return None
    file_id = str(raw_file_id).strip()
    return file_id or None


def resolve_document_filename(
    record: Dict[str, Any], filename_map: Dict[str, str]
) -> Optional[str]:
    """Resolve a user-facing filename from ``file_id`` first, then legacy path."""
    file_id = get_document_record_file_id(record)
    if file_id and filename_map.get(file_id):
        return filename_map[file_id]
    source_path = record.get("source_path")
    if source_path:
        return os.path.basename(str(source_path))
    return None


def delete_uploaded_file_if_orphaned(
    db: Session,
    *,
    file_id: str,
    user_id: int,
    remaining_file_ids: set[str],
) -> bool:
    """Delete uploaded file row and local file when no documents still reference it."""
    if not file_id or file_id in remaining_file_ids:
        return False

    file_record = (
        db.query(UploadedFile)
        .filter(
            UploadedFile.user_id == user_id,
            UploadedFile.file_id == file_id,
        )
        .first()
    )
    if file_record is None:
        return False

    uploads_root = UPLOADS_DIR.resolve()
    file_path = Path(str(file_record.storage_path))
    try:
        resolved_path = file_path.resolve()
        resolved_path.relative_to(uploads_root)
    except ValueError:
        logger.warning(
            "Skipping physical delete for file outside uploads root: %s",
            file_path,
        )
    else:
        if resolved_path.exists() and resolved_path.is_file():
            resolved_path.unlink()

    db.delete(file_record)
    return True
