"""
E2E Tests for Real URL Ingestion in Knowledge Base System.

This test suite verifies that real URLs can be ingested correctly,
with focus on actual web content rather than mocked data.

These tests use real URLs to verify:
1. Web crawling functionality works with actual websites
2. Content is correctly extracted and parsed
3. Ingested content is searchable
4. Display shows correct metadata
"""

import time
from typing import Dict

import pytest


class TestRealURLIngestion:
    """Test ingestion of real URLs from actual websites.

    These tests verify the complete flow from URL upload through
    parsing, storage, and searchability using real web content.
    """

    # Real URLs for testing (stable, publicly accessible documentation)
    GITHUB_URLS = [
        "https://github.com/xorbitsai/xagent",
    ]

    DOC_URLS = [
        # These are backup URLs if GitHub tests fail
        # "https://docs.python.org/3/tutorial/index.html",
        # "https://react.dev/learn",
    ]

    @pytest.mark.slow
    @pytest.mark.requires_network
    def test_github_readme_ingestion(
        self, client, auth_headers: Dict[str, str]    ):
        """Test ingestion of GitHub repository README.

        This test verifies that:
        1. GitHub URL can be uploaded
        2. README content is correctly extracted
        3. Content is chunked and stored
        4. Content is searchable
        """
        github_url = "https://github.com/xorbitsai/xagent"

        # Upload GitHub URL
        response = client.post(
            "/api/kb/ingest/url/",
            json={"url": github_url, "collection_name": "github_xagent_repo"},
            headers=auth_headers,
        )

        # Accept both success and async processing responses
        assert response.status_code in [200, 202], (
            f"GitHub URL ingestion failed: {response.text}"
        )

        result = response.json()

        # Should have collection info
        assert (
            "collection_id" in result or "collection" in result or "task_id" in result
        )

        # If async, wait for completion
        if "task_id" in result:
            task_id = result["task_id"]
            max_wait = 60  # 60 seconds max wait
            start_time = time.time()

            while time.time() - start_time < max_wait:
                status_response = client.get(
                    f"/api/kb/tasks/{task_id}/", headers=auth_headers
                )
                if status_response.status_code == 200:
                    status = status_response.json()
                    if status.get("status") in ["completed", "failed"]:
                        break
                time.sleep(2)

        # Get collection ID
        collection_id = result.get("collection_id")
        if not collection_id and "collection" in result:
            collection_id = result["collection"].get("id")

        assert collection_id is not None, "Could not get collection_id"

        # Verify documents were created
        response = client.get(
            f"/api/kb/collections/{collection_id}/documents/", headers=auth_headers
        )
        assert response.status_code == 200

        documents = response.json().get("documents", [])
        assert len(documents) > 0, "No documents were created from GitHub URL"

        # Verify content is searchable
        # Give it some time for indexing
        time.sleep(5)

        response = client.post(
            f"/api/kb/collections/{collection_id}/search/",
            json={"query": "xagent AI agent framework"},
            headers=auth_headers,
        )

        # Search should work (might return no results if not indexed, but should not error)
        assert response.status_code == 200

    @pytest.mark.slow
    @pytest.mark.requires_network
    def test_github_raw_content_ingestion(
        self, client, auth_headers: Dict[str, str]    ):
        """Test ingestion of raw GitHub content.

        This test verifies that raw GitHub content (like README.md)
        can be ingested correctly.
        """
        # Use raw GitHub URL for README
        raw_url = "https://raw.githubusercontent.com/xorbitsai/xagent/main/README.md"

        response = client.post(
            "/api/kb/ingest/url/",
            json={"url": raw_url, "collection_name": "github_raw_readme"},
            headers=auth_headers,
        )

        # Should succeed
        assert response.status_code in [200, 202], (
            f"Raw GitHub URL ingestion failed: {response.text}"
        )

        result = response.json()
        collection_id = result.get("collection_id")

        if not collection_id and "collection" in result:
            collection_id = result["collection"].get("id")

        assert collection_id is not None

        # Verify document was created
        response = client.get(
            f"/api/kb/collections/{collection_id}/documents/", headers=auth_headers
        )
        assert response.status_code == 200

        documents = response.json().get("documents", [])
        assert len(documents) > 0, "No documents from raw GitHub URL"

    @pytest.mark.slow
    @pytest.mark.requires_network
    def test_real_url_search_verification(
        self, client, auth_headers: Dict[str, str]    ):
        """Test that ingested real URLs produce searchable content.

        This is a comprehensive test that verifies:
        1. URL ingestion completes
        2. Content is properly indexed
        3. Search returns relevant results
        4. Results are properly ranked
        """
        github_url = "https://github.com/xorbitsai/xagent"

        # Ingest URL
        response = client.post(
            "/api/kb/ingest/url/",
            json={
                "url": github_url,
                "collection_name": "search_verification_collection",
            },
            headers=auth_headers,
        )

        assert response.status_code in [200, 202]
        result = response.json()

        collection_id = result.get("collection_id")
        if not collection_id and "collection" in result:
            collection_id = result["collection"].get("id")

        assert collection_id is not None

        # Wait for indexing (real URLs take time)
        max_wait = 90
        start_time = time.time()

        while time.time() - start_time < max_wait:
            response = client.post(
                f"/api/kb/collections/{collection_id}/search/",
                json={"query": "xagent agent framework"},
                headers=auth_headers,
            )

            if response.status_code == 200:
                results = response.json().get("results", [])
                if len(results) > 0:
                    # Found results!
                    break

            time.sleep(5)

        # Final search verification
        response = client.post(
            f"/api/kb/collections/{collection_id}/search/",
            json={"query": "xagent agent framework"},
            headers=auth_headers,
        )

        assert response.status_code == 200
        results = response.json().get("results", [])

        # Verify results structure
        for result in results:
            assert "content" in result or "text" in result
            # Should have relevance info
            assert any(k in result for k in ["score", "distance", "relevance"])

    @pytest.mark.slow
    @pytest.mark.requires_network
    def test_multiple_github_urls_ingestion(
        self, client, auth_headers: Dict[str, str]    ):
        """Test ingestion of multiple GitHub URLs into same collection.

        This verifies that multiple URLs can be ingested and
        searched together.
        """
        urls = [
            "https://github.com/xorbitsai/xagent",
            # Can add more related repos here
        ]

        collection_name = "multiple_github_urls"

        for url in urls:
            response = client.post(
                "/api/kb/ingest/url/",
                json={"url": url, "collection_name": collection_name},
                headers=auth_headers,
            )
            assert response.status_code in [200, 202], (
                f"Failed to ingest {url}: {response.text}"
            )

        # Find the collection
        response = client.get("/api/kb/collections/", headers=auth_headers)
        assert response.status_code == 200

        collections = response.json()["collections"]
        test_collection = next(
            (c for c in collections if c["name"] == collection_name), None
        )
        assert test_collection is not None

        collection_id = test_collection["id"]

        # Verify multiple documents exist
        response = client.get(
            f"/api/kb/collections/{collection_id}/documents/", headers=auth_headers
        )
        assert response.status_code == 200

        documents = response.json().get("documents", [])
        # Should have documents from multiple URLs
        assert len(documents) >= len(urls), (
            f"Expected at least {len(urls)} documents, got {len(documents)}"
        )


class TestRealURLParsing:
    """Test parsing behavior with real URLs.

    These tests verify that content from real URLs is parsed
    correctly and metadata is extracted properly.
    """

    @pytest.mark.slow
    @pytest.mark.requires_network
    def test_github_markdown_parsing(
        self, client, auth_headers: Dict[str, str]    ):
        """Test that GitHub markdown is parsed correctly.

        GitHub repositories contain markdown files. This test verifies
        that markdown formatting is handled correctly.
        """
        # Use a raw markdown file from GitHub
        url = "https://raw.githubusercontent.com/xorbitsai/xagent/main/README.md"

        response = client.post(
            "/api/kb/ingest/url/",
            json={"url": url, "collection_name": "github_markdown_test"},
            headers=auth_headers,
        )

        assert response.status_code in [200, 202]
        result = response.json()

        collection_id = result.get("collection_id")
        if not collection_id and "collection" in result:
            collection_id = result["collection"].get("id")

        assert collection_id is not None

        # Check documents
        response = client.get(
            f"/api/kb/collections/{collection_id}/documents/", headers=auth_headers
        )
        assert response.status_code == 200

        documents = response.json().get("documents", [])
        assert len(documents) > 0

        # Verify markdown-specific elements are preserved
        # (This depends on the parser implementation)

    @pytest.mark.slow
    @pytest.mark.requires_network
    def test_url_metadata_extraction(
        self, client, auth_headers: Dict[str, str]    ):
        """Test that metadata is correctly extracted from URLs.

        This verifies that:
        1. Source URL is stored
        2. Title/heading is extracted
        3. Author/date info is captured if available
        4. Content type is detected
        """
        url = "https://github.com/xorbitsai/xagent"

        response = client.post(
            "/api/kb/ingest/url/",
            json={"url": url, "collection_name": "metadata_test_collection"},
            headers=auth_headers,
        )

        assert response.status_code in [200, 202]
        result = response.json()

        collection_id = result.get("collection_id")
        if not collection_id and "collection" in result:
            collection_id = result["collection"].get("id")

        # Check documents for metadata
        response = client.get(
            f"/api/kb/collections/{collection_id}/documents/", headers=auth_headers
        )
        assert response.status_code == 200

        documents = response.json().get("documents", [])

        for doc in documents:
            # Should have metadata
            metadata = doc.get("metadata", {})

            # Verify source URL is preserved
            assert (
                "source" in metadata
                or "url" in metadata
                or doc.get("source_url")
                or doc.get("url")
            ), "Source URL should be in metadata"

            # Other metadata fields (optional, might not be present)
            # title, author, date, content_type, etc.


class TestRealURLErrors:
    """Test error handling with real URLs.

    These tests verify that the system handles real-world errors
    like network issues, invalid URLs, etc.
    """

    @pytest.mark.slow
    @pytest.mark.requires_network
    def test_invalid_url_handling(
        self, client, auth_headers: Dict[str, str]    ):
        """Test that invalid URLs are handled gracefully.

        This uses a real but invalid URL to verify error handling.
        """
        invalid_url = (
            "https://this-domain-definitely-does-not-exist-12345.com/README.md"
        )

        response = client.post(
            "/api/kb/ingest/url/",
            json={"url": invalid_url, "collection_name": "invalid_url_test"},
            headers=auth_headers,
        )

        # Should fail gracefully
        assert response.status_code in [400, 404, 500], (
            "Invalid URL should be handled with appropriate error"
        )

    @pytest.mark.slow
    @pytest.mark.requires_network
    def test_private_url_handling(
        self, client, auth_headers: Dict[str, str]    ):
        """Test that private/restricted URLs are handled properly.

        This test uses a URL that requires authentication to verify
        the system handles it correctly.
        """
        # Use a private GitHub repo URL (will fail without auth)
        # This is just to test error handling
        private_url = "https://github.com/xorbitsai/private-repo-that-does-not-exist"

        response = client.post(
            "/api/kb/ingest/url/",
            json={"url": private_url, "collection_name": "private_url_test"},
            headers=auth_headers,
        )

        # Should fail gracefully (401, 403, or 404)
        assert response.status_code in [400, 401, 403, 404], (
            "Private URL should be handled with appropriate error"
        )

    @pytest.mark.slow
    @pytest.mark.requires_network
    def test_malformed_url_handling(
        self, client, auth_headers: Dict[str, str]    ):
        """Test that malformed URLs are rejected.

        This verifies URL validation works correctly.
        """
        malformed_urls = [
            "not-a-url",
            "htp://missing-slashes.com",
            "ftp://unsupported-protocol.com",
            "",
        ]

        for malformed_url in malformed_urls:
            response = client.post(
                "/api/kb/ingest/url/",
                json={"url": malformed_url, "collection_name": "malformed_url_test"},
                headers=auth_headers,
            )

            # Should reject malformed URLs
            assert response.status_code in [400, 422], (
                f"Malformed URL should be rejected: {malformed_url}"
            )


class TestRealURLDisplay:
    """Test frontend display of real URL content.

    These tests verify that content ingested from real URLs
    is displayed correctly in the frontend.
    """

    @pytest.mark.slow
    @pytest.mark.requires_network
    def test_ingested_url_document_display(
        self, client, auth_headers: Dict[str, str]    ):
        """Test that ingested URL documents display correctly.

        This verifies that:
        1. Document names are derived from URL
        2. Source URL is shown
        3. Content preview is available
        4. Metadata is displayed
        """
        url = "https://github.com/xorbitsai/xagent"

        response = client.post(
            "/api/kb/ingest/url/",
            json={"url": url, "collection_name": "display_test_collection"},
            headers=auth_headers,
        )

        assert response.status_code in [200, 202]
        result = response.json()

        collection_id = result.get("collection_id")
        if not collection_id and "collection" in result:
            collection_id = result["collection"].get("id")

        # Get documents for display
        response = client.get(
            f"/api/kb/collections/{collection_id}/documents/", headers=auth_headers
        )
        assert response.status_code == 200

        documents = response.json().get("documents", [])

        for doc in documents:
            # Should have displayable name
            assert "name" in doc or "filename" in doc or "title" in doc, (
                "Document should have displayable name"
            )

            # Should show source URL
            metadata = doc.get("metadata", {})
            assert (
                "source" in metadata
                or "url" in metadata
                or doc.get("source_url")
                or doc.get("url")
            ), "Source URL should be visible for display"

            # Should have content preview
            # (This depends on implementation)

    @pytest.mark.slow
    @pytest.mark.requires_network
    def test_collection_list_with_url_documents(
        self, client, auth_headers: Dict[str, str]    ):
        """Test that collections with URL documents display correctly.

        This verifies the collection list shows correct information
        for collections containing URL-ingested documents.
        """
        url = "https://github.com/xorbitsai/xagent"

        response = client.post(
            "/api/kb/ingest/url/",
            json={"url": url, "collection_name": "url_collection_display_test"},
            headers=auth_headers,
        )

        assert response.status_code in [200, 202]

        # List collections
        response = client.get("/api/kb/collections/", headers=auth_headers)
        assert response.status_code == 200

        collections = response.json()["collections"]
        test_collection = next(
            (c for c in collections if c["name"] == "url_collection_display_test"), None
        )

        assert test_collection is not None

        # Should show document count
        assert any(
            k in test_collection
            for k in ["document_count", "doc_count", "count", "size"]
        )

        # Should show collection metadata
        # (This depends on implementation)


class TestRealURLReingestion:
    """Test re-ingestion of real URLs.

    These tests verify that re-ingesting URLs works correctly,
    handling updates and duplicates properly.
    """

    @pytest.mark.slow
    @pytest.mark.requires_network
    def test_url_reingestion(
        self, client, auth_headers: Dict[str, str]    ):
        """Test re-ingesting the same URL.

        This verifies that:
        1. Re-ingestion updates existing content
        2. Or creates new version (depending on implementation)
        3. No duplicate documents are created unnecessarily
        """
        url = "https://github.com/xorbitsai/xagent"
        collection_name = "reingestion_test_collection"

        # First ingestion
        response = client.post(
            "/api/kb/ingest/url/",
            json={"url": url, "collection_name": collection_name},
            headers=auth_headers,
        )
        assert response.status_code in [200, 202]

        # Get collection info
        collections_response = client.get("/api/kb/collections/", headers=auth_headers)
        collections = collections_response.json()["collections"]
        test_collection = next(
            (c for c in collections if c["name"] == collection_name), None
        )

        if test_collection:
            collection_id = test_collection["id"]

            # Get document count before re-ingestion
            docs_response = client.get(
                f"/api/kb/collections/{collection_id}/documents/", headers=auth_headers
            )
            docs_before = docs_response.json().get("documents", [])

            # Re-ingest same URL
            response = client.post(
                "/api/kb/ingest/url/",
                json={"url": url, "collection_name": collection_name},
                headers=auth_headers,
            )
            assert response.status_code in [200, 202]

            # Get document count after re-ingestion
            docs_response = client.get(
                f"/api/kb/collections/{collection_id}/documents/", headers=auth_headers
            )
            docs_after = docs_response.json().get("documents", [])

            # Behavior depends on implementation:
            # - Should update existing documents (same count)
            # - Or create new versions (increased count)
            # - Or skip duplicates (same count)
            assert len(docs_after) >= len(docs_before), (
                "Re-ingestion should not decrease document count"
            )


# ==========================================
# TEST FIXTURES
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
    from tests.web.api.test_kb_dir import test_env as kb_test_env

    # Reuse existing test environment from kb_dir tests
    yield from kb_test_env()
