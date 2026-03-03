"""Tests for register_document functionality."""

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from xagent.core.tools.core.RAG_tools.core.exceptions import (
    ConfigurationError,
    DatabaseOperationError,
    DocumentValidationError,
    HashComputationError,
)
from xagent.core.tools.core.RAG_tools.file.register_document import register_document


class TestRegisterDocument:
    """Test cases for register_document function."""

    def test_register_document_new_document(self, tmp_path: Path, monkeypatch) -> None:
        """Test registering a new document successfully."""
        # Setup environment variable
        db_dir = tmp_path / "lancedb"
        monkeypatch.setenv("LANCEDB_DIR", str(db_dir))

        # Setup test file
        test_file = tmp_path / "test.txt"
        test_file.write_text("Test content")

        response = register_document(
            collection="test_collection",
            source_path=str(test_file),
        )

        assert response["doc_id"] is not None
        assert response["created"] is True
        assert response["content_hash"] is not None
        assert len(response["content_hash"]) == 64  # SHA256 hash length

    def test_register_document_auto_file_type_detection(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """Test automatic file type detection from extension."""
        # Setup environment variable
        db_dir = tmp_path / "lancedb"
        monkeypatch.setenv("LANCEDB_DIR", str(db_dir))

        # Setup test file with .md extension
        test_file = tmp_path / "test.md"
        test_file.write_text("# Test Markdown")

        response = register_document(
            collection="test_collection",
            source_path=str(test_file),
        )

        assert response["doc_id"] is not None
        assert response["created"] is True
        assert response["content_hash"] is not None

    def test_register_document_idempotency(self, tmp_path: Path, monkeypatch) -> None:
        """Test idempotent registration with same doc_id."""
        # Setup environment variable
        db_dir = tmp_path / "lancedb"
        monkeypatch.setenv("LANCEDB_DIR", str(db_dir))

        # Setup test file
        test_file = tmp_path / "test.txt"
        test_file.write_text("Test content")

        doc_id = "test-doc-123"

        # First registration
        response1 = register_document(
            collection="test_collection",
            source_path=str(test_file),
            doc_id=doc_id,
        )
        assert response1["doc_id"] == doc_id
        assert response1["created"] is True

        # Second registration with same doc_id
        response2 = register_document(
            collection="test_collection",
            source_path=str(test_file),
            doc_id=doc_id,
        )
        assert response2["doc_id"] == doc_id
        assert response2["created"] is False  # Should be update, not create

    def test_register_document_deterministic_doc_id_idempotency(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """Without doc_id, same (collection, source_path) yields same doc_id and second is update."""
        db_dir = tmp_path / "lancedb"
        monkeypatch.setenv("LANCEDB_DIR", str(db_dir))

        test_file = tmp_path / "report.docx"
        test_file.write_text("content")

        response1 = register_document(
            collection="my_kb",
            source_path=str(test_file),
        )
        response2 = register_document(
            collection="my_kb",
            source_path=str(test_file),
        )

        assert response1["doc_id"] == response2["doc_id"]
        assert response1["created"] is True
        assert response2["created"] is False

    def test_register_document_custom_uploaded_at(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """Test registration with custom uploaded_at timestamp."""
        # Setup environment variable
        db_dir = tmp_path / "lancedb"
        monkeypatch.setenv("LANCEDB_DIR", str(db_dir))

        # Setup test file
        test_file = tmp_path / "test.txt"
        test_file.write_text("Test content")

        custom_time = datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc)

        response = register_document(
            collection="test_collection",
            source_path=str(test_file),
            uploaded_at=custom_time.isoformat(),
        )

        assert response["doc_id"] is not None
        assert response["created"] is True
        assert response["content_hash"] is not None

    def test_register_document_empty_file(self, tmp_path: Path, monkeypatch) -> None:
        """Test registering an empty file."""
        # Setup environment variable
        db_dir = tmp_path / "lancedb"
        monkeypatch.setenv("LANCEDB_DIR", str(db_dir))

        # Setup empty test file
        test_file = tmp_path / "empty.txt"
        test_file.write_text("")

        response = register_document(
            collection="test_collection",
            source_path=str(test_file),
        )

        assert response["doc_id"] is not None
        assert response["created"] is True
        assert response["content_hash"] is not None
        # Empty file should have a hash (hash of empty string)
        assert response["content_hash"] == (
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        )

    def test_register_document_different_collection(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """Test registration in different collection."""
        # Setup environment variable
        db_dir = tmp_path / "lancedb"
        monkeypatch.setenv("LANCEDB_DIR", str(db_dir))

        # Setup test file
        test_file = tmp_path / "test.txt"
        test_file.write_text("Test content")

        response = register_document(
            collection="different_collection",
            source_path=str(test_file),
        )

        assert response["doc_id"] is not None
        assert response["created"] is True
        assert response["content_hash"] is not None

    def test_register_document_invalid_path(self) -> None:
        """Test registration with invalid file path."""
        with pytest.raises(DocumentValidationError, match="Source path does not exist"):
            register_document(
                collection="test_collection", source_path="/nonexistent/file.txt"
            )

    def test_register_document_empty_collection(self) -> None:
        """Test registration with empty collection name."""
        with pytest.raises(
            DocumentValidationError, match="Collection name cannot be empty"
        ):
            register_document(collection="", source_path="/tmp/test.txt")

    @patch("xagent.core.tools.core.RAG_tools.file.register_document.compute_file_hash")
    def test_register_document_hash_computation_error(
        self, mock_hash, tmp_path: Path, monkeypatch
    ) -> None:
        """Test handling hash computation errors."""
        # Setup environment variable
        db_dir = tmp_path / "lancedb"
        monkeypatch.setenv("LANCEDB_DIR", str(db_dir))

        # Setup test file
        test_file = tmp_path / "hash_error_test.txt"
        test_file.write_text("Test content")

        # Mock hash computation to raise error
        mock_hash.side_effect = HashComputationError("Hash computation failed")

        # Should propagate HashComputationError
        with pytest.raises(HashComputationError):
            register_document(collection="test_collection", source_path=str(test_file))

    @patch(
        "xagent.core.tools.core.RAG_tools.file.register_document.get_connection_from_env"
    )
    def test_register_document_configuration_error(
        self, mock_get_db, tmp_path: Path
    ) -> None:
        """Test handling configuration errors."""
        # Setup test file
        test_file = tmp_path / "config_error_test.txt"
        test_file.write_text("Test content")

        # Mock database connection to raise configuration error
        mock_get_db.side_effect = ConfigurationError("LANCEDB_DIR not configured")

        # Should propagate ConfigurationError
        with pytest.raises(ConfigurationError):
            register_document(collection="test_collection", source_path=str(test_file))

    def test_register_document_unsupported_file_type(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """Test registering document with unsupported file type."""
        # Setup environment variable
        db_dir = tmp_path / "lancedb"
        monkeypatch.setenv("LANCEDB_DIR", str(db_dir))

        # Create a file with unsupported extension
        unsupported_file = tmp_path / "test.unsupported"
        unsupported_file.write_text("Unsupported file content")

        collection = "test_collection"

        # Should raise DocumentValidationError for unsupported file type
        with pytest.raises(
            DocumentValidationError, match="Unsupported file type: \\.unsupported"
        ):
            register_document(collection=collection, source_path=str(unsupported_file))

    @patch(
        "xagent.core.tools.core.RAG_tools.file.register_document.get_connection_from_env"
    )
    def test_register_document_database_operation_error(
        self, mock_get_db, tmp_path: Path, monkeypatch
    ) -> None:
        """Test handling database operation errors."""
        # Setup environment variable
        db_dir = tmp_path / "lancedb"
        monkeypatch.setenv("LANCEDB_DIR", str(db_dir))

        # Setup test file
        test_file = tmp_path / "db_error_test.txt"
        test_file.write_text("Test content")

        # Mock database connection to succeed, but table operations to fail
        mock_db = MagicMock()
        mock_get_db.return_value = mock_db

        # Mock ensure_documents_table to succeed
        mock_db.ensure_documents_table = MagicMock()

        # Mock open_table to raise an error
        mock_db.open_table.side_effect = Exception("Table access failed")

        # Should propagate DatabaseOperationError
        with pytest.raises(DatabaseOperationError, match="Table access failed"):
            register_document(collection="test_collection", source_path=str(test_file))
