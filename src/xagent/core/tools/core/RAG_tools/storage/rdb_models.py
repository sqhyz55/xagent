"""SQLAlchemy ORM models for KB metadata storage.

Phase 1B: RDB migration with file_id integration, multi-user isolation, and staged upload.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import Boolean, DateTime, Index, Integer, String, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Base class for all RAG KB metadata models."""

    pass


class KBCollectionMetadata(Base):
    """Collection metadata stored in relational database.

    Phase 1B additions:
    - owner_user_id: Collection owner for multi-user isolation
    - external_file_id: Linkage to file system's file_id
    """

    __tablename__ = "kb_collection_metadata"

    # Primary identification
    name: Mapped[str] = mapped_column(String(255), primary_key=True)

    # Phase 1B: Owner (for multi-user isolation)
    owner_user_id: Mapped[int] = mapped_column(
        Integer, nullable=False, index=True, comment="User ID of the collection owner"
    )

    # Schema and embedding info
    schema_version: Mapped[str] = mapped_column(String(50), default="1.0.0")
    embedding_model_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    embedding_dimension: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Statistics
    documents: Mapped[int] = mapped_column(Integer, default=0)
    processed_documents: Mapped[int] = mapped_column(Integer, default=0)
    parses: Mapped[int] = mapped_column(Integer, default=0)
    chunks: Mapped[int] = mapped_column(Integer, default=0)
    embeddings: Mapped[int] = mapped_column(Integer, default=0)

    # Document tracking
    document_names: Mapped[dict[str, Any]] = mapped_column(JSONB, default=list)

    # Collection flags
    collection_locked: Mapped[bool] = mapped_column(Boolean, default=False)
    allow_mixed_parse_methods: Mapped[bool] = mapped_column(Boolean, default=True)
    skip_config_validation: Mapped[bool] = mapped_column(Boolean, default=False)

    # Configuration (JSON)
    ingestion_config: Mapped[dict[str, Any] | None] = mapped_column(
        JSONB, nullable=True
    )

    # Phase 1B: File ID linkage
    external_file_id: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
        index=True,
        comment="Link to file system file_id for cross-domain reference",
    )

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )
    last_accessed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    # Additional metadata
    extra_metadata: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)

    __table_args__ = (
        Index("idx_kb_collection_metadata_updated_at", "updated_at"),
        Index("idx_kb_collection_metadata_owner_user_id", "owner_user_id"),
        Index("idx_kb_collection_metadata_external_file_id", "external_file_id"),
    )


class KBCollectionShare(Base):
    """Collection read-only sharing (Phase 1B).

    Owner can grant read-only access to other users.
    Shared users can view and search, but cannot upload/delete/process.
    """

    __tablename__ = "kb_collection_shares"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    collection: Mapped[str] = mapped_column(String(255), nullable=False)
    shared_with_user_id: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    created_by: Mapped[int] = mapped_column(Integer, nullable=False)

    __table_args__ = (
        Index("idx_kb_collection_shares_collection", "collection"),
        Index("idx_kb_collection_shares_shared_with_user_id", "shared_with_user_id"),
        UniqueConstraint(
            "collection",
            "shared_with_user_id",
            name="uq_kb_collection_shares_collection_user",
        ),
    )


class KBDocumentStaging(Base):
    """Staged documents pending or in processing (Phase 1B).

    Supports decoupling file upload from processing:
    - Files are registered via file_id immediately
    - Processing happens later on demand (via Celery or manual trigger)
    - State machine: uploaded → queued → parsing → chunked → embedding → complete
    """

    __tablename__ = "kb_document_staging"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    collection: Mapped[str] = mapped_column(String(255), nullable=False)
    doc_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    file_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    uploaded_by_user_id: Mapped[int] = mapped_column(Integer, nullable=False)

    # Processing state
    status: Mapped[str] = mapped_column(
        String(50), nullable=False, default="uploaded", index=True
    )  # 'uploaded', 'queued', 'parsing', 'chunked', 'embedding', 'complete', 'failed'

    # Timestamps
    uploaded_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    processing_started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    # Error tracking
    error_message: Mapped[str | None] = mapped_column(String, nullable=True)
    retry_count: Mapped[int] = mapped_column(Integer, default=0)

    # Processing metadata
    parse_method: Mapped[str | None] = mapped_column(String(100), nullable=True)
    chunk_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    embedding_model: Mapped[str | None] = mapped_column(String(255), nullable=True)

    __table_args__ = (
        Index("idx_kb_document_staging_collection", "collection"),
        Index("idx_kb_document_staging_doc_id", "doc_id"),
        Index("idx_kb_document_staging_file_id", "file_id"),
        Index("idx_kb_document_staging_status", "status"),
        Index("idx_kb_document_staging_uploaded_by_user_id", "uploaded_by_user_id"),
        UniqueConstraint(
            "collection",
            "doc_id",
            name="uq_kb_document_staging_collection_doc_id",
        ),
    )


class KBCollectionConfig(Base):
    """Per-user collection configuration.

    Note: is_admin is NOT stored here - it's a runtime permission check
    determined by the user's role at query time.
    """

    __tablename__ = "kb_collection_config"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    collection: Mapped[str] = mapped_column(String(255), nullable=False)
    user_id: Mapped[int] = mapped_column(Integer, nullable=False)
    config_json: Mapped[dict[str, Any]] = mapped_column(
        JSONB, default=dict, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    __table_args__ = (
        Index("idx_kb_collection_config_collection", "collection"),
        Index("idx_kb_collection_config_user_id", "user_id"),
        UniqueConstraint(
            "collection", "user_id", name="uq_kb_collection_config_collection_user"
        ),
    )
