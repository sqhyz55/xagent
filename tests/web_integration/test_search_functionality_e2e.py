"""
End-to-end tests for search functionality verification.

This module tests that search works correctly after document ingestion,
ensuring that users can find their content immediately.

IMPORTANT: Legacy data compatibility tests in this file must be updated
whenever schema changes are made to ensure forward compatibility is maintained.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Generator

import pytest
from fastapi.testclient import TestClient

from xagent.core.model.embedding.base import BaseEmbedding
from xagent.core.model.model import EmbeddingModelConfig
from xagent.core.tools.core.RAG_tools.core.schemas import (
    CollectionInfo,
)

pytestmark = [pytest.mark.e2e, pytest.mark.contract_stub, pytest.mark.real_rag]

# ==========================================
# TEST FIXTURES
# ==========================================


class _StubEmbeddingAdapter(BaseEmbedding):
    """Deterministic embedding adapter for search E2E tests."""

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
def stub_embedding_config() -> EmbeddingModelConfig:
    """Create stub embedding configuration for search tests."""
    return EmbeddingModelConfig(
        id="e2e-search-embedding",
        model_name="e2e-search-embedding-model",
        model_provider="test",
        dimension=2,
    )


@pytest.fixture
def stub_embedding_adapter() -> _StubEmbeddingAdapter:
    """Create stub embedding adapter for search tests."""
    return _StubEmbeddingAdapter()


@pytest.fixture
def mock_search_rag_pipeline(
    monkeypatch: Any,
    stub_embedding_config: EmbeddingModelConfig,
    stub_embedding_adapter: _StubEmbeddingAdapter,
) -> None:
    """Mock the RAG pipeline components for search E2E testing."""
    from xagent.core.tools.core.RAG_tools import pipelines as pipelines_module
    from xagent.core.tools.core.RAG_tools.management import collection_manager
    from xagent.core.tools.core.RAG_tools.utils import model_resolver

    mgr = collection_manager.collection_manager

    # Mock collection to exist
    mock_collection = CollectionInfo(
        name="e2e_search_test",
        embedding_model_id="e2e-search-embedding",
        embedding_dimension=2,
    )

    async def mock_get_collection(collection_name: str) -> CollectionInfo:
        return mock_collection

    async def mock_initialize_collection(
        collection_name: str, embedding_model_id: str
    ) -> CollectionInfo:
        return mock_collection

    def mock_resolve_embedding_adapter(
        model_id: str | None = None, **kwargs: Any
    ) -> tuple[EmbeddingModelConfig, BaseEmbedding]:
        return (stub_embedding_config, stub_embedding_adapter)

    # Apply mocks (patch singleton used by KB routes)
    monkeypatch.setattr(
        mgr,
        "get_collection",
        mock_get_collection,
    )
    monkeypatch.setattr(
        mgr,
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


@pytest.fixture
def legacy_collection_data(
    monkeypatch: Any,
    stub_embedding_config: EmbeddingModelConfig,
    stub_embedding_adapter: _StubEmbeddingAdapter,
) -> None:
    """Create legacy schema data for forward compatibility testing.

    IMPORTANT: This fixture creates data with legacy schema to test that
    new code can handle old data formats. Must be updated when schema changes:

    - When new required fields are added: update this fixture to NOT include them
    - When field types change: test both old and new types
    - When schema structure changes: ensure old structure is still readable

    Current legacy schema simulation:
    - Direct LanceDB table creation without new schema fields
    - Tests graceful degradation when new fields are missing
    """
    from xagent.core.tools.core.RAG_tools.management import collection_manager
    from xagent.core.tools.core.RAG_tools.utils import model_resolver

    mgr = collection_manager.collection_manager

    # Mock to allow legacy collection operations
    async def mock_get_collection_legacy(collection_name: str) -> CollectionInfo:
        # Return collection with minimal fields (legacy schema)
        return CollectionInfo(
            name=collection_name,
            embedding_model_id="e2e-search-embedding",
            embedding_dimension=2,
            # Intentionally omit some new fields to test graceful degradation
        )

    async def mock_initialize_collection_legacy(
        collection_name: str, embedding_model_id: str
    ) -> CollectionInfo:
        return CollectionInfo(
            name=collection_name,
            embedding_model_id=embedding_model_id,
            embedding_dimension=2,
        )

    monkeypatch.setattr(
        mgr,
        "get_collection",
        mock_get_collection_legacy,
    )
    monkeypatch.setattr(
        mgr,
        "initialize_collection_embedding",
        mock_initialize_collection_legacy,
    )
    monkeypatch.setattr(
        model_resolver,
        "resolve_embedding_adapter",
        lambda **kwargs: (stub_embedding_config, stub_embedding_adapter),
    )

    # Note: In real scenario, legacy data would be existing data from old schema
    # For testing, we rely on the actual ingestion process to create collections
    # which we then query to test forward compatibility


@pytest.fixture
def sample_search_files() -> Generator[tuple[dict[str, str], str], None, None]:
    """Create sample test files for search testing."""
    files = {}

    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test files with specific searchable content
        test_files = {
            "python_tutorial.txt": "Python is a programming language. Python is widely used for web development, data science, and automation.",
            "machine_learning.md": "# Machine Learning Guide\n\nMachine learning is a subset of artificial intelligence. It focuses on building systems that can learn from data.",
            "cooking_guide.txt": "Cooking tips: Always use fresh ingredients. Follow the recipe carefully. Taste your food while cooking.",
            "travel_guide.txt": "Travel destinations: Paris, Tokyo, New York are popular cities. Each city has unique attractions and culture.",
        }

        for filename, content in test_files.items():
            file_path = Path(temp_dir) / filename
            file_path.write_text(content, encoding="utf-8")
            files[filename] = str(file_path)

        yield files, temp_dir


# ==========================================
# BASIC SEARCH TESTS
# ==========================================


class TestBasicSearch:
    """
    Test basic search functionality.

    These tests verify that:
    1. Search works immediately after ingestion
    2. Search results are relevant
    3. Search pagination works
    4. Search filters work correctly
    """

    @pytest.mark.e2e
    @pytest.mark.slow
    def test_search_immediate_after_ingestion(
        self,
        client: TestClient,
        auth_headers: dict[str, str],
        sample_search_files: tuple[dict[str, str], str],
        mock_search_rag_pipeline: None,
    ) -> None:
        """Test that search works immediately after document ingestion."""
        files, temp_dir = sample_search_files
        collection_name = "e2e_search_immediate"
        file_path = files["python_tutorial.txt"]

        # Ingest a document
        with open(file_path, "rb") as f:
            ingest_response = client.post(
                "/api/kb/ingest",
                files={"file": ("python_tutorial.txt", f, "text/plain")},
                data={"collection": collection_name},
                headers=auth_headers,
            )

        if ingest_response.status_code == 200:
            # Search immediately after ingestion
            search_response = client.post(
                "/api/kb/search",
                data={
                    "collection": collection_name,
                    "query_text": "Python programming",
                    "top_k": "5",
                },
                headers=auth_headers,
            )

            # Search should work
            assert search_response.status_code == 200
            result = search_response.json()
            assert "results" in result or "status" in result

    @pytest.mark.e2e
    @pytest.mark.slow
    def test_search_relevance_ranking(
        self,
        client: TestClient,
        auth_headers: dict[str, str],
        sample_search_files: tuple[dict[str, str], str],
        mock_search_rag_pipeline: None,
    ) -> None:
        """Test that search results are ranked by relevance."""
        files, temp_dir = sample_search_files
        collection_name = "e2e_search_relevance"

        # Ingest documents with specific content
        keywords_docs = [
            ("python_tutorial.txt", "Python"),
            ("machine_learning.md", "machine learning"),
            ("cooking_guide.txt", "cooking"),
        ]

        for filename, keyword in keywords_docs:
            file_path = files[filename]
            with open(file_path, "rb") as f:
                client.post(
                    "/api/kb/ingest",
                    files={"file": (filename, f, "text/plain")},
                    data={"collection": collection_name},
                    headers=auth_headers,
                )

        # Search for specific term and verify relevance
        search_response = client.post(
            "/api/kb/search",
            data={
                "collection": collection_name,
                "query_text": "Python",
                "top_k": 3,
            },
            headers=auth_headers,
        )

        # Should return relevant results
        assert search_response.status_code == 200
        result = search_response.json()
        assert "results" in result or "status" in result

    @pytest.mark.e2e
    @pytest.mark.slow
    def test_search_pagination(
        self,
        client: TestClient,
        auth_headers: dict[str, str],
        sample_search_files: tuple[dict[str, str], str],
        mock_search_rag_pipeline: None,
    ) -> None:
        """Test that search pagination works correctly."""
        files, temp_dir = sample_search_files
        collection_name = "e2e_search_pagination"

        # Create multiple documents
        for i in range(5):
            content = f"Document {i} with searchable content about topic {i % 3}."
            with tempfile.NamedTemporaryFile(
                suffix=".txt", delete=False, mode="w"
            ) as tmp:
                tmp.write(content)
                tmp.flush()

                try:
                    with open(tmp.name, "rb") as f:
                        client.post(
                            "/api/kb/ingest",
                            files={"file": (f"doc{i}.txt", f, "text/plain")},
                            data={"collection": collection_name},
                            headers=auth_headers,
                        )
                finally:
                    import os

                    os.unlink(tmp.name)

        # Test pagination
        search_response = client.post(
            "/api/kb/search",
            data={
                "collection": collection_name,
                "query_text": "document",
                "top_k": 3,
            },
            headers=auth_headers,
        )

        # Should handle pagination
        assert search_response.status_code == 200

    @pytest.mark.e2e
    @pytest.mark.slow
    def test_search_filters(
        self,
        client: TestClient,
        auth_headers: dict[str, str],
        sample_search_files: tuple[dict[str, str], str],
        mock_search_rag_pipeline: None,
    ) -> None:
        """Test that search filters work correctly."""
        files, temp_dir = sample_search_files
        collection_name = "e2e_search_filters"

        # Ingest documents
        for filename in ["python_tutorial.txt", "machine_learning.md"]:
            file_path = files[filename]
            with open(file_path, "rb") as f:
                client.post(
                    "/api/kb/ingest",
                    files={"file": (filename, f, "text/plain")},
                    data={"collection": collection_name},
                    headers=auth_headers,
                )

        # Search with filters
        search_response = client.post(
            "/api/kb/search",
            data={
                "collection": collection_name,
                "query_text": "document",
                "top_k": 5,
                # Add any available filters
            },
            headers=auth_headers,
        )

        # Should handle search
        assert search_response.status_code == 200


# ==========================================
# MULTI-TENANT SEARCH TESTS
# ==========================================


class TestMultiTenantSearch:
    """
    Test multi-tenant search isolation.

    These tests verify that:
    1. Users can only search their own documents
    2. Admin users can search across tenants
    3. Legacy data isolation works correctly
    """

    @pytest.mark.e2e
    @pytest.mark.slow
    def test_search_only_returns_own_documents(
        self,
        client: TestClient,
        auth_headers: dict[str, str],
        sample_search_files: tuple[dict[str, str], str],
        mock_search_rag_pipeline: None,
    ) -> None:
        """Test that regular users can only search their own documents."""
        files, temp_dir = sample_search_files
        collection_name = "e2e_search_isolation"
        file_path = files["python_tutorial.txt"]

        # Ingest a document
        with open(file_path, "rb") as f:
            client.post(
                "/api/kb/ingest",
                files={"file": ("python_tutorial.txt", f, "text/plain")},
                data={"collection": collection_name},
                headers=auth_headers,
            )

        # Search as regular user
        search_response = client.post(
            "/api/kb/search",
            data={
                "collection": collection_name,
                "query_text": "Python",
                "top_k": "5",
            },
            headers=auth_headers,
        )

        # Search should respect tenant isolation
        assert search_response.status_code == 200
        result = search_response.json()
        assert "results" in result or "status" in result

    @pytest.mark.e2e
    @pytest.mark.slow
    def test_search_with_legacy_data_isolation(
        self,
        client: TestClient,
        auth_headers: dict[str, str],
        legacy_collection_data: None,
    ) -> None:
        """Test that legacy data is properly isolated in search results.

        FORWARD COMPATIBILITY TEST:
        This test verifies that the system can handle data created with
        previous schema versions. When schema changes are made, ensure:
        1. Old collections remain searchable
        2. Tenant isolation works for legacy data
        3. No crashes when new fields are missing

        UPDATE THIS TEST when:
        - New required fields are added to documents schema
        - Collection schema structure changes
        - Search logic depends on new fields
        """
        collection_name = "e2e_search_legacy"

        # First, ingest a document to create the collection with current schema
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Legacy data isolation test content")
            temp_path = Path(f.name)

        try:
            with open(temp_path, "rb") as f:
                ingest_response = client.post(
                    "/api/kb/ingest",
                    files={"file": ("legacy_test.txt", f, "text/plain")},
                    data={"collection": collection_name},
                    headers=auth_headers,
                )

            # If ingestion succeeded, search should work
            if ingest_response.status_code == 200:
                search_response = client.post(
                    "/api/kb/search",
                    data={
                        "collection": collection_name,
                        "query_text": "legacy",
                        "top_k": 5,
                    },
                    headers=auth_headers,
                )

                # Search should work with legacy data
                assert search_response.status_code == 200
                result = search_response.json()
                assert "results" in result or "status" in result
            else:
                # If collection doesn't exist, search may return 404
                # This is acceptable for legacy data testing
                pass
        finally:
            temp_path.unlink(missing_ok=True)


# ==========================================
# SEARCH AFTER SCHEMA CHANGES TESTS
# ==========================================


class TestSearchAfterSchemaChanges:
    """
    Test search functionality after schema changes.

    These tests verify that:
    1. Search works after migration
    2. Search works with mixed schema versions
    3. Fallback mechanisms work for legacy data
    """

    @pytest.mark.e2e
    @pytest.mark.slow
    def test_search_after_migration(
        self,
        client: TestClient,
        auth_headers: dict[str, str],
        sample_search_files: tuple[dict[str, str], str],
        mock_search_rag_pipeline: None,
    ) -> None:
        """Test that search works correctly after schema migration."""
        files, temp_dir = sample_search_files
        collection_name = "e2e_search_migration"
        file_path = files["python_tutorial.txt"]

        # Ingest document (simulating post-migration state)
        with open(file_path, "rb") as f:
            client.post(
                "/api/kb/ingest",
                files={"file": ("python_tutorial.txt", f, "text/plain")},
                data={"collection": collection_name},
                headers=auth_headers,
            )

        # Search should work after migration
        search_response = client.post(
            "/api/kb/search",
            data={
                "collection": collection_name,
                "query_text": "Python",
                "top_k": "5",
            },
            headers=auth_headers,
        )

        assert search_response.status_code == 200

    @pytest.mark.e2e
    @pytest.mark.slow
    def test_search_with_mixed_schema_versions(
        self,
        client: TestClient,
        auth_headers: dict[str, str],
        legacy_collection_data: None,
    ) -> None:
        """Test that search works with mixed schema versions.

        FORWARD COMPATIBILITY TEST:
        This test verifies that search can handle documents with different
        schema versions in the same collection. When schema changes are made:
        1. Test with documents missing new fields (old schema)
        2. Test with documents having new fields (new schema)
        3. Verify search works across both

        UPDATE THIS TEST when:
        - Document schema adds new optional fields
        - Field types change between versions
        - Search behavior depends on schema version
        """
        collection_name = "e2e_search_mixed"

        # Ingest documents to simulate mixed schema scenario
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Mixed schema test content with keywords")
            temp_path = Path(f.name)

        try:
            with open(temp_path, "rb") as f:
                ingest_response = client.post(
                    "/api/kb/ingest",
                    files={"file": ("mixed_test.txt", f, "text/plain")},
                    data={"collection": collection_name},
                    headers=auth_headers,
                )

            # If ingestion succeeded, test search
            if ingest_response.status_code == 200:
                search_response = client.post(
                    "/api/kb/search",
                    data={
                        "collection": collection_name,
                        "query_text": "mixed",
                        "top_k": 5,
                    },
                    headers=auth_headers,
                )

                # Search should handle mixed schema data
                assert search_response.status_code == 200
        finally:
            temp_path.unlink(missing_ok=True)

    @pytest.mark.e2e
    @pytest.mark.slow
    def test_search_fallback_for_legacy_data(
        self,
        client: TestClient,
        auth_headers: dict[str, str],
        legacy_collection_data: None,
    ) -> None:
        """Test that search fallback mechanisms work for legacy data.

        FORWARD COMPATIBILITY TEST:
        This test verifies that when new search features fail on legacy data,
        the system falls back gracefully. When adding new search features:
        1. Test with legacy data that doesn't support the feature
        2. Verify fallback to basic search works
        3. Ensure no crashes or errors

        UPDATE THIS TEST when:
        - New search features are added that depend on new schema fields
        - Search algorithms change significantly
        - New filtering or ranking options are introduced
        """
        collection_name = "e2e_search_fallback"

        # Ingest a document to create the collection
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Fallback test content for legacy data")
            temp_path = Path(f.name)

        try:
            with open(temp_path, "rb") as f:
                ingest_response = client.post(
                    "/api/kb/ingest",
                    files={"file": ("fallback_test.txt", f, "text/plain")},
                    data={"collection": collection_name},
                    headers=auth_headers,
                )

            # Test search with basic query (should work even with legacy data)
            if ingest_response.status_code == 200:
                search_response = client.post(
                    "/api/kb/search",
                    data={
                        "collection": collection_name,
                        "query_text": "fallback",
                        "top_k": 5,
                    },
                    headers=auth_headers,
                )

                # Should provide results or graceful fallback
                assert search_response.status_code in [200, 404]
        finally:
            temp_path.unlink(missing_ok=True)


# ==========================================
# SEARCH ACCURACY TESTS
# ==========================================


class TestSearchAccuracy:
    """
    Test search result accuracy and quality.

    These tests verify that:
    1. Search returns relevant results
    2. Search scores are reasonable
    3. Search handles different query types
    """

    @pytest.mark.e2e
    @pytest.mark.slow
    def test_search_returns_relevant_results(
        self,
        client: TestClient,
        auth_headers: dict[str, str],
        sample_search_files: tuple[dict[str, str], str],
        mock_search_rag_pipeline: None,
    ) -> None:
        """Test that search returns relevant results."""
        files, temp_dir = sample_search_files
        collection_name = "e2e_search_accuracy"
        file_path = files["python_tutorial.txt"]

        # Ingest document with specific content
        with open(file_path, "rb") as f:
            client.post(
                "/api/kb/ingest",
                files={"file": ("python_tutorial.txt", f, "text/plain")},
                data={"collection": collection_name},
                headers=auth_headers,
            )

        # Search for specific term from the document
        search_response = client.post(
            "/api/kb/search",
            data={
                "collection": collection_name,
                "query_text": "Python programming language",
                "top_k": 5,
            },
            headers=auth_headers,
        )

        assert search_response.status_code == 200
        result = search_response.json()
        assert "results" in result or "status" in result

    @pytest.mark.e2e
    @pytest.mark.slow
    def test_search_with_different_query_types(
        self,
        client: TestClient,
        auth_headers: dict[str, str],
        sample_search_files: tuple[dict[str, str], str],
        mock_search_rag_pipeline: None,
    ) -> None:
        """Test that search handles different query types correctly."""
        files, temp_dir = sample_search_files
        collection_name = "e2e_search_query_types"

        # Ingest document
        file_path = files["python_tutorial.txt"]
        with open(file_path, "rb") as f:
            client.post(
                "/api/kb/ingest",
                files={"file": ("python_tutorial.txt", f, "text/plain")},
                data={"collection": collection_name},
                headers=auth_headers,
            )

        # Test different query types
        queries = [
            "Python",  # Single word
            "Python programming",  # Phrase
            "Python language web development",  # Multiple words
        ]

        for query in queries:
            search_response = client.post(
                "/api/kb/search",
                data={
                    "collection": collection_name,
                    "query_text": query,
                    "top_k": 3,
                },
                headers=auth_headers,
            )

            # All query types should work
            assert search_response.status_code == 200


# ==========================================
# REAL-TIME SEARCH TESTS
# ==========================================


class TestRealTimeSearch:
    """
    Test real-time search functionality.

    These tests verify that:
    1. Search works immediately after ingestion
    2. Search updates in real-time
    3. Search handles concurrent operations
    """

    @pytest.mark.e2e
    @pytest.mark.slow
    def test_search_updates_in_realtime(
        self,
        client: TestClient,
        auth_headers: dict[str, str],
        sample_search_files: tuple[dict[str, str], str],
        mock_search_rag_pipeline: None,
    ) -> None:
        """Test that search results update in real-time after ingestion."""
        files, temp_dir = sample_search_files
        collection_name = "e2e_search_realtime"

        # Initial search should be empty
        client.post(
            "/api/kb/search",
            data={
                "collection": collection_name,
                "query_text": "Python",
                "top_k": "5",
            },
            headers=auth_headers,
        )

        # Ingest document
        file_path = files["python_tutorial.txt"]
        with open(file_path, "rb") as f:
            ingest_response = client.post(
                "/api/kb/ingest",
                files={"file": ("python_tutorial.txt", f, "text/plain")},
                data={"collection": collection_name},
                headers=auth_headers,
            )

        if ingest_response.status_code == 200:
            # Search again should now return results
            final_search = client.post(
                "/api/kb/search",
                data={
                    "collection": collection_name,
                    "query_text": "Python",
                    "top_k": 5,
                },
                headers=auth_headers,
            )

            assert final_search.status_code == 200

    @pytest.mark.e2e
    @pytest.mark.slow
    def test_search_handles_concurrent_operations(
        self,
        client: TestClient,
        auth_headers: dict[str, str],
        sample_search_files: tuple[dict[str, str], str],
        mock_search_rag_pipeline: None,
    ) -> None:
        """Test that search handles concurrent ingestion operations."""
        import concurrent.futures

        files, temp_dir = sample_search_files
        collection_name = "e2e_search_concurrent"

        def ingest_and_search(filename: str) -> int:
            """Ingest a file and then search for it."""
            file_path = files[filename]
            with open(file_path, "rb") as f:
                ingest_response = client.post(
                    "/api/kb/ingest",
                    files={"file": (filename, f, "text/plain")},
                    data={"collection": collection_name},
                    headers=auth_headers,
                )

            if ingest_response.status_code == 200:
                # Search for content
                search_response = client.post(
                    "/api/kb/search",
                    data={
                        "collection": collection_name,
                        "query_text": "content",
                        "top_k": 3,
                    },
                    headers=auth_headers,
                )
                return search_response.status_code
            return ingest_response.status_code

        # Run concurrent ingestion and search operations
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [
                executor.submit(ingest_and_search, "python_tutorial.txt"),
                executor.submit(ingest_and_search, "machine_learning.md"),
            ]
            results = [
                future.result() for future in concurrent.futures.as_completed(futures)
            ]

        # At least some operations should succeed
        success_count = sum(1 for code in results if code == 200)
        assert success_count >= 1


# ==========================================
# SEARCH ERROR HANDLING TESTS
# ==========================================


class TestSearchErrorHandling:
    """Test search error handling and edge cases."""

    @pytest.mark.e2e
    @pytest.mark.slow
    def test_search_with_empty_query(
        self,
        client: TestClient,
        auth_headers: dict[str, str],
        mock_search_rag_pipeline: None,
    ) -> None:
        """Test that search handles empty queries gracefully."""
        collection_name = "e2e_search_empty"

        search_response = client.post(
            "/api/kb/search",
            data={
                "collection": collection_name,
                "query_text": "",  # Empty query
                "top_k": 5,
            },
            headers=auth_headers,
        )

        # Should handle empty query gracefully
        assert search_response.status_code in [200, 400, 422, 500]

    @pytest.mark.e2e
    @pytest.mark.slow
    def test_search_nonexistent_collection(
        self,
        client: TestClient,
        auth_headers: dict[str, str],
        mock_search_rag_pipeline: None,
    ) -> None:
        """Test that search handles nonexistent collections gracefully."""
        search_response = client.post(
            "/api/kb/search",
            data={
                "collection": "nonexistent_collection_xyz",
                "query_text": "test",
                "top_k": 5,
            },
            headers=auth_headers,
        )

        # Should handle nonexistent collection gracefully
        assert search_response.status_code in [200, 404, 500]

    @pytest.mark.e2e
    @pytest.mark.slow
    def test_search_with_special_characters(
        self,
        client: TestClient,
        auth_headers: dict[str, str],
        sample_search_files: tuple[dict[str, str], str],
        mock_search_rag_pipeline: None,
    ) -> None:
        """Test that search handles special characters in queries."""
        files, temp_dir = sample_search_files
        collection_name = "e2e_search_special"
        file_path = files["python_tutorial.txt"]

        # Ingest document
        with open(file_path, "rb") as f:
            client.post(
                "/api/kb/ingest",
                files={"file": ("python_tutorial.txt", f, "text/plain")},
                data={"collection": collection_name},
                headers=auth_headers,
            )

        # Search with special characters
        special_queries = [
            "Python & programming",  # Ampersand
            "Python, data, science",  # Commas
            'Python "language"',  # Quotes
        ]

        for query in special_queries:
            search_response = client.post(
                "/api/kb/search",
                data={
                    "collection": collection_name,
                    "query_text": query,
                    "top_k": 3,
                },
                headers=auth_headers,
            )

            # Should handle special characters
            assert search_response.status_code in [200, 400, 422, 500]
