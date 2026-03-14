"""change_task_fk_to_cascade

Revision ID: 44a6d3a54c35
Revises: a0f42ff986b2
Create Date: 2026-03-11 00:47:06.197244

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "44a6d3a54c35"
down_revision: Union[str, None] = "a0f42ff986b2"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Change uploaded_files.task_id FK to ON DELETE CASCADE in a dialect-agnostic way."""

    # 1. Create a new table definition with the desired FK behavior.
    op.create_table(
        "uploaded_files_new",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("file_id", sa.String(length=36), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("task_id", sa.Integer(), nullable=True),
        sa.Column("filename", sa.String(length=512), nullable=False),
        sa.Column("storage_path", sa.String(length=2048), nullable=False),
        sa.Column("mime_type", sa.String(length=255), nullable=True),
        sa.Column("file_size", sa.Integer(), nullable=False, server_default="0"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
        ),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["task_id"], ["tasks.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("file_id"),
        sa.UniqueConstraint("storage_path"),
    )

    # 2. Migrate existing data into the new table.
    op.execute(
        """
        INSERT INTO uploaded_files_new (
            id,
            file_id,
            user_id,
            task_id,
            filename,
            storage_path,
            mime_type,
            file_size,
            created_at,
            updated_at
        )
        SELECT
            id,
            file_id,
            user_id,
            task_id,
            filename,
            storage_path,
            mime_type,
            file_size,
            created_at,
            updated_at
        FROM uploaded_files
        """
    )

    # 3. Drop indexes on the old table to avoid name collisions, then drop the table.
    op.drop_index(op.f("ix_uploaded_files_file_id"), table_name="uploaded_files")
    op.drop_index(op.f("ix_uploaded_files_id"), table_name="uploaded_files")
    op.drop_table("uploaded_files")

    # 4. Rename the new table back to the original name and recreate indexes.
    op.rename_table("uploaded_files_new", "uploaded_files")
    op.create_index(
        op.f("ix_uploaded_files_id"), "uploaded_files", ["id"], unique=False
    )
    op.create_index(
        op.f("ix_uploaded_files_file_id"),
        "uploaded_files",
        ["file_id"],
        unique=False,
    )


def downgrade() -> None:
    """Revert uploaded_files.task_id FK back to ON DELETE SET NULL."""

    # 1. Create a new table definition that matches the original schema.
    op.create_table(
        "uploaded_files_new",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("file_id", sa.String(length=36), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("task_id", sa.Integer(), nullable=True),
        sa.Column("filename", sa.String(length=512), nullable=False),
        sa.Column("storage_path", sa.String(length=2048), nullable=False),
        sa.Column("mime_type", sa.String(length=255), nullable=True),
        sa.Column("file_size", sa.Integer(), nullable=False, server_default="0"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
        ),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["task_id"], ["tasks.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("file_id"),
        sa.UniqueConstraint("storage_path"),
    )

    # 2. Copy data back into the "original" schema.
    op.execute(
        """
        INSERT INTO uploaded_files_new (
            id,
            file_id,
            user_id,
            task_id,
            filename,
            storage_path,
            mime_type,
            file_size,
            created_at,
            updated_at
        )
        SELECT
            id,
            file_id,
            user_id,
            task_id,
            filename,
            storage_path,
            mime_type,
            file_size,
            created_at,
            updated_at
        FROM uploaded_files
        """
    )

    # 3. Drop indexes and the current table, then rename and recreate indexes.
    op.drop_index(op.f("ix_uploaded_files_file_id"), table_name="uploaded_files")
    op.drop_index(op.f("ix_uploaded_files_id"), table_name="uploaded_files")
    op.drop_table("uploaded_files")

    op.rename_table("uploaded_files_new", "uploaded_files")
    op.create_index(
        op.f("ix_uploaded_files_id"), "uploaded_files", ["id"], unique=False
    )
    op.create_index(
        op.f("ix_uploaded_files_file_id"),
        "uploaded_files",
        ["file_id"],
        unique=False,
    )
