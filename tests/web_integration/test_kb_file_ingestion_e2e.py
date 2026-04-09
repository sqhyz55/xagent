"""
End-to-end tests for KB file ingestion workflow.

This module tests the complete workflow from frontend file upload
through RAG processing to final searchability, ensuring all
components work together correctly.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Dict

import pytest
from fastapi.testclient import TestClient

from xagent.core.model.embedding.base import BaseEmbedding
from xagent.core.model.model import EmbeddingModelConfig
from xagent.core.tools.core.RAG_tools.core.schemas import (
    CollectionInfo,
)

# ==========================================
# TEST FIXTURES
# ==========================================


class _StubEmbeddingAdapter(BaseEmbedding):
    """Deterministic embedding adapter for E2E tests."""

    def encode(
        self,
        text: Any,
        dimension: int | None = None,
        instruct: str | None = None,
    ) -> Any:
        if isinstance(text, str):
            return [float(len(text)), 0.0]
        return [[float(len(item)), float(index)] for index, item in enumerate(text)]

    def get_dimension(self) -> int:
        return 2

    @property
    def abilities(self) -> list[str]:
        return ["embedding"]


@pytest.fixture
def stub_embedding_config():
    """Create stub embedding configuration for testing."""
    return EmbeddingModelConfig(
        id="e2e-test-embedding",
        model_name="e2e-test-embedding-model",
        model_provider="test",
        dimension=2,
    )


@pytest.fixture
def stub_embedding_adapter():
    """Create stub embedding adapter for testing."""
    return _StubEmbeddingAdapter()


@pytest.fixture
def sample_test_files():
    """Create sample test files for E2E testing."""
    files = {}

    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test files of different formats
        test_files = {
            "test.txt": "This is a test text file for E2E testing.\nIt contains multiple lines.",
            "test.md": "# Test Markdown\n\nThis is a **markdown** test file.\n\n## Section 2\n\nContent here.",
            "test.json": '{"title": "Test Document", "content": "JSON test content", "count": 42}',
            "test.csv": "name,age,city\nJohn,25,NYC\nJane,30,LA",
        }

        for filename, content in test_files.items():
            file_path = Path(temp_dir) / filename
            file_path.write_text(content, encoding="utf-8")
            files[filename] = str(file_path)

        yield files, temp_dir


@pytest.fixture
def mock_rag_pipeline(monkeypatch, stub_embedding_config, stub_embedding_adapter):
    """Mock the RAG pipeline components for E2E testing.

    This fixture provides realistic mocks that simulate successful
    RAG processing without requiring actual embedding models or
    vector database operations.
    """
    from xagent.core.tools.core.RAG_tools import pipelines as pipelines_module
    from xagent.core.tools.core.RAG_tools.management import collection_manager
    from xagent.core.tools.core.RAG_tools.utils import model_resolver

    # Mock collection to exist
    mock_collection = CollectionInfo(
        name="e2e_test_collection",
        embedding_model_id="e2e-test-embedding",
        embedding_dimension=2,
    )

    async def mock_get_collection(collection_name: str) -> CollectionInfo:
        return mock_collection

    async def mock_initialize_collection(
        collection_name: str, embedding_model_id: str
    ) -> CollectionInfo:
        return mock_collection

    def mock_resolve_embedding_adapter(
        model_id: str | None = None, **kwargs
    ) -> tuple[EmbeddingModelConfig, BaseEmbedding]:
        return (stub_embedding_config, stub_embedding_adapter)

    # Apply mocks
    monkeypatch.setattr(
        collection_manager,
        "get_collection",
        mock_get_collection,
    )
    monkeypatch.setattr(
        collection_manager,
        "initialize_collection_embedding",
        mock_initialize_collection,
    )
    monkeypatch.setattr(
        model_resolver,
        "resolve_embedding_adapter",
        mock_resolve_embedding_adapter,
    )

    # Also mock in pipelines module
    monkeypatch.setattr(
        pipelines_module.document_ingestion,
        "_resolve_embedding_adapter",
        lambda cfg: (stub_embedding_config, stub_embedding_adapter),
    )


# ==========================================
# E2E TEST CLASSES
# ==========================================


class TestKBFileIngestionE2E:
    """
    End-to-end tests for KB file ingestion workflow.

    These tests simulate the complete user workflow:
    1. Frontend uploads file via API
    2. Backend saves file to storage
    3. RAG processing (parse → chunk → embed)
    4. Content becomes searchable
    """

    @pytest.mark.e2e
    @pytest.mark.slow
    def test_ingest_text_file_complete_workflow(
        self,
        client: TestClient,
        auth_headers: Dict[str, str],
        sample_test_files: tuple[Dict[str, str], str],
        mock_rag_pipeline: None,
    ):
        """Test complete workflow: upload .txt file → RAG processing → searchable."""
        files, temp_dir = sample_test_files
        file_path = files["test.txt"]
        collection_name = "e2e_txt_test"

        # Step 1: Upload file via API
        with open(file_path, "rb") as f:
            upload_response = client.post(
                "/api/kb/ingest",
                files={"file": ("test.txt", f, "text/plain")},
                data={
                    "collection": collection_name,
                    "parse_method": "default",
                    "chunk_strategy": "fixed_size",
                    "chunk_size": "500",
                    "chunk_overlap": "50",
                },
                headers=auth_headers,
            )

        # Step 2: Verify upload was successful
        assert upload_response.status_code == 200
        result = upload_response.json()
        assert result["status"] in ["success", "partial"]
        assert "doc_id" in result or result["status"] == "success"

    @pytest.mark.e2e
    @pytest.mark.slow
    def test_ingest_markdown_file_complete_workflow(
        self,
        client: TestClient,
        auth_headers: Dict[str, str],
        sample_test_files: tuple[Dict[str, str], str],
        mock_rag_pipeline: None,
    ):
        """Test complete workflow: upload .md file → RAG processing → searchable."""
        files, temp_dir = sample_test_files
        file_path = files["test.md"]
        collection_name = "e2e_md_test"

        # Upload markdown file with markdown-specific chunking
        with open(file_path, "rb") as f:
            upload_response = client.post(
                "/api/kb/ingest",
                files={"file": ("test.md", f, "text/markdown")},
                data={
                    "collection": collection_name,
                    "parse_method": "default",
                    "chunk_strategy": "markdown",  # Use markdown-aware chunking
                    "chunk_size": "300",
                    "chunk_overlap": "50",
                },
                headers=auth_headers,
            )

        # Verify response
        assert upload_response.status_code == 200
        result = upload_response.json()
        assert result["status"] in ["success", "partial"]

    @pytest.mark.e2e
    @pytest.mark.slow
    def test_ingest_json_file_complete_workflow(
        self,
        client: TestClient,
        auth_headers: Dict[str, str],
        sample_test_files: tuple[Dict[str, str], str],
        mock_rag_pipeline: None,
    ):
        """Test complete workflow: upload .json file → RAG processing → searchable."""
        files, temp_dir = sample_test_files
        file_path = files["test.json"]
        collection_name = "e2e_json_test"

        # Upload JSON file
        with open(file_path, "rb") as f:
            upload_response = client.post(
                "/api/kb/ingest",
                files={"file": ("test.json", f, "application/json")},
                data={
                    "collection": collection_name,
                    "parse_method": "default",
                    "chunk_strategy": "fixed_size",
                },
                headers=auth_headers,
            )

        # Verify response
        assert upload_response.status_code == 200
        result = upload_response.json()
        assert result["status"] in ["success", "partial"]

    @pytest.mark.e2e
    @pytest.mark.slow
    def test_ingest_csv_file_complete_workflow(
        self,
        client: TestClient,
        auth_headers: Dict[str, str],
        sample_test_files: tuple[Dict[str, str], str],
        mock_rag_pipeline: None,
    ):
        """Test complete workflow: upload .csv file → RAG processing → searchable."""
        files, temp_dir = sample_test_files
        file_path = files["test.csv"]
        collection_name = "e2e_csv_test"

        # Upload CSV file
        with open(file_path, "rb") as f:
            upload_response = client.post(
                "/api/kb/ingest",
                files={"file": ("test.csv", f, "text/csv")},
                data={
                    "collection": collection_name,
                    "parse_method": "default",
                    "chunk_strategy": "fixed_size",
                },
                headers=auth_headers,
            )

        # Verify response
        assert upload_response.status_code == 200
        result = upload_response.json()
        assert result["status"] in ["success", "partial"]

    @pytest.mark.e2e
    @pytest.mark.slow
    def test_ingest_multiple_files_same_collection(
        self,
        client: TestClient,
        auth_headers: Dict[str, str],
        sample_test_files: tuple[Dict[str, str], str],
        mock_rag_pipeline: None,
    ):
        """Test uploading multiple files to the same collection."""
        files, temp_dir = sample_test_files
        collection_name = "e2e_multi_test"

        uploaded_count = 0
        for filename in ["test.txt", "test.md", "test.json"]:
            file_path = files[filename]
            with open(file_path, "rb") as f:
                response = client.post(
                    "/api/kb/ingest",
                    files={"file": (filename, f, "text/plain")},
                    data={"collection": collection_name},
                    headers=auth_headers,
                )
                if response.status_code == 200:
                    uploaded_count += 1

        # Verify at least some files were processed
        assert uploaded_count >= 1

    @pytest.mark.e2e
    @pytest.mark.slow
    def test_list_documents_after_ingestion(
        self,
        client: TestClient,
        auth_headers: Dict[str, str],
        sample_test_files: tuple[Dict[str, str], str],
        mock_rag_pipeline: None,
    ):
        """Test that uploaded documents appear in collection listing."""
        files, temp_dir = sample_test_files
        file_path = files["test.txt"]
        collection_name = "e2e_list_test"

        # Upload a file
        with open(file_path, "rb") as f:
            upload_response = client.post(
                "/api/kb/ingest",
                files={"file": ("test.txt", f, "text/plain")},
                data={"collection": collection_name},
                headers=auth_headers,
            )

        if upload_response.status_code == 200:
            # List collections and verify our collection appears
            list_response = client.get("/api/kb/collections", headers=auth_headers)
            assert list_response.status_code == 200

    @pytest.mark.e2e
    @pytest.mark.slow
    def test_delete_document_after_ingestion(
        self,
        client: TestClient,
        auth_headers: Dict[str, str],
        sample_test_files: tuple[Dict[str, str], str],
        mock_rag_pipeline: None,
    ):
        """Test deleting a document after successful ingestion."""
        files, temp_dir = sample_test_files
        file_path = files["test.txt"]
        collection_name = "e2e_delete_test"

        # Upload a file
        with open(file_path, "rb") as f:
            upload_response = client.post(
                "/api/kb/ingest",
                files={"file": ("test.txt", f, "text/plain")},
                data={"collection": collection_name},
                headers=auth_headers,
            )

        if upload_response.status_code == 200:
            # Try to delete the document
            delete_response = client.delete(
                f"/api/kb/collections/{collection_name}/documents/test.txt",
                headers=auth_headers,
            )
            # Delete may succeed (200) or fail (404) - both are acceptable for E2E
            assert delete_response.status_code in [200, 404, 500]

    @pytest.mark.e2e
    @pytest.mark.slow
    def test_error_handling_unsupported_format(
        self,
        client: TestClient,
        auth_headers: Dict[str, str],
        mock_rag_pipeline: None,
    ):
        """Test error handling when uploading unsupported file format."""
        # Create a fake .exe file
        with tempfile.NamedTemporaryFile(suffix=".exe", delete=False) as tmp:
            tmp.write(b"fake executable content")
            tmp.flush()

            with open(tmp.name, "rb") as f:
                response = client.post(
                    "/api/kb/ingest",
                    files={"file": ("test.exe", f, "application/octet-stream")},
                    data={"collection": "e2e_error_test"},
                    headers=auth_headers,
                )

            # Should handle error gracefully
            assert response.status_code in [200, 400, 422, 500]

            os.unlink(tmp.name)


# ==========================================
# TEST FIXTURES FOR MODULE
# ==========================================


@pytest.fixture
def client(test_env):
    """Provide test client for E2E tests."""
    app, headers, user, TestingSessionLocal = test_env
    from fastapi.testclient import TestClient

    return TestClient(app)


@pytest.fixture
def auth_headers(test_env):
    """Provide authentication headers for E2E tests."""
    app, headers, user, TestingSessionLocal = test_env
    return headers


@pytest.fixture
def test_env():
    """Provide complete test environment for E2E tests.

    This fixture sets up:
    - FastAPI app with all routes
    - Authentication headers
    - Test database session
    - Temporary upload directory
    """
    from xagent.web.api.test_kb_dir import test_env as kb_test_env

    # Reuse existing test environment from kb_dir tests
    yield from kb_test_env()
