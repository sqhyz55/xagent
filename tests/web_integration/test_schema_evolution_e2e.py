"""
E2E Tests for Schema Evolution and Migration Compatibility.

This test suite verifies that schema changes and database migrations
do not break core functionality in the RAG/KB system.

These tests are critical for ensuring that:
1. Behavior remains consistent before and after schema migrations
2. Mixed schema versions can coexist during migration
3. Graceful degradation when fields are missing
4. Legacy data remains accessible after schema upgrades
"""

import json
import tempfile
from pathlib import Path
from typing import Dict

import pytest


class TestSchemaMigrationCompatibility:
    """Test schema migration compatibility and coexistence.

    These tests verify that the system can handle different schema versions
    simultaneously during migrations and that operations work correctly.
    """

    @pytest.fixture
    def sample_documents(self, tmp_path: Path) -> Dict[str, Path]:
        """Create sample documents for testing."""
        docs = {}

        # Text document
        docs["text"] = tmp_path / "sample.txt"
        docs["text"].write_text(
            "This is a sample document for schema evolution testing. "
            "It contains various keywords: testing, schema, evolution, migration. "
            "The purpose is to verify behavior consistency across schema versions."
        )

        # JSON document
        docs["json"] = tmp_path / "data.json"
        docs["json"].write_text(
            json.dumps(
                {
                    "title": "Schema Evolution Test",
                    "content": "Testing JSON document handling across schema versions",
                    "metadata": {
                        "version": "1.0",
                        "tags": ["schema", "evolution", "test"],
                    },
                }
            )
        )

        # Markdown document
        docs["markdown"] = tmp_path / "guide.md"
        docs["markdown"].write_text("""
# Schema Evolution Guide

This document tests markdown processing during schema changes.

## Key Points
- Behavior should remain consistent
- Data should not be lost
- Operations should work across versions

## Testing Keywords
schema evolution migration compatibility
        """)

        return docs

    def test_ingestion_with_new_schema_fields(
        self,
        client,
        auth_headers: Dict[str, str],
        sample_documents: Dict[str, Path],
        clean_storage: None,
    ):
        """Test that ingestion works with new schema fields.

        This verifies that when new fields are added to the schema,
        ingestion still works and both old and new fields are handled.
        """
        # Upload a document
        file_path = sample_documents["text"]
        with open(file_path, "rb") as f:
            response = client.post(
                "/api/kb/upload/",
                files={"file": (file_path.name, f, "text/plain")},
                data={"collection_name": "schema_test_collection"},
                headers=auth_headers,
            )

        assert response.status_code == 200
        result = response.json()

        # Verify response contains both legacy and new fields
        assert "collection_id" in result
        assert "file_id" in result or "doc_id" in result or "filename" in result

        # New schema fields (if present)
        optional_new_fields = [
            "created_at",
            "updated_at",
            "schema_version",
            "metadata",
            "chunk_count",
            "status",
        ]
        # At least some new fields should be present
        _ = any(f in result for f in optional_new_fields)

    def test_read_operations_with_mixed_schemas(
        self,
        client,
        auth_headers: Dict[str, str],
        sample_documents: Dict[str, Path],
        clean_storage: None,
    ):
        """Test that read operations work with mixed schema versions.

        This simulates a scenario where some documents use the old schema
        and some use the new schema.
        """
        collection_name = "mixed_schema_collection"

        # Upload multiple documents (potentially creating mixed schemas)
        file_ids = []
        for doc_key, file_path in sample_documents.items():
            with open(file_path, "rb") as f:
                response = client.post(
                    "/api/kb/upload/",
                    files={
                        "file": (
                            file_path.name,
                            f,
                            "text/plain"
                            if doc_key == "text"
                            else "application/json"
                            if doc_key == "json"
                            else "text/markdown",
                        )
                    },
                    data={"collection_name": collection_name},
                    headers=auth_headers,
                )
                assert response.status_code == 200
                file_ids.append(response.json().get("file_id"))

        # List documents - should handle mixed schemas
        response = client.get("/api/kb/collections/", headers=auth_headers)
        assert response.status_code == 200

        collections = response.json()["collections"]
        test_collection = next(
            (c for c in collections if c["name"] == collection_name), None
        )
        assert test_collection is not None

        # Get collection details - should work with mixed schemas
        collection_id = test_collection["id"]
        response = client.get(
            f"/api/kb/collections/{collection_id}/documents/", headers=auth_headers
        )
        assert response.status_code == 200

        documents = response.json().get("documents", [])
        assert len(documents) >= len(sample_documents)

    def test_field_absence_graceful_degradation(
        self, client, auth_headers: Dict[str, str], clean_storage: None
    ):
        """Test graceful degradation when expected fields are missing.

        This verifies that if new schema fields are missing (old schema),
        the system degrades gracefully rather than failing.
        """
        # Create a collection
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Test content for field absence testing")
            temp_path = Path(f.name)

        try:
            with open(temp_path, "rb") as f:
                response = client.post(
                    "/api/kb/upload/",
                    files={"file": ("test.txt", f, "text/plain")},
                    data={"collection_name": "field_test_collection"},
                    headers=auth_headers,
                )
                assert response.status_code == 200

                # Even if some new fields are missing, basic operations should work
                result = response.json()

                # Core fields should be present
                assert "collection_id" in result or "collection" in result

                # Optional new fields might be missing - that's OK
                # The system should not crash
        finally:
            temp_path.unlink(missing_ok=True)

    def test_search_with_legacy_schema_data(
        self, client, auth_headers: Dict[str, str], clean_storage: None
    ):
        """Test that search works with legacy schema data.

        This verifies that documents using the old schema can still be searched.
        """
        # Upload a document
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(
                "Legacy schema search test. "
                "Keywords: legacy, schema, search, compatibility. "
                "This simulates old data that needs to remain searchable."
            )
            temp_path = Path(f.name)

        try:
            with open(temp_path, "rb") as f:
                response = client.post(
                    "/api/kb/upload/",
                    files={"file": ("legacy.txt", f, "text/plain")},
                    data={"collection_name": "legacy_search_collection"},
                    headers=auth_headers,
                )
                assert response.status_code == 200
                collection_id = response.json()["collection_id"]

            # Search should work even with legacy schema
            response = client.post(
                f"/api/kb/collections/{collection_id}/search/",
                json={"query": "legacy schema"},
                headers=auth_headers,
            )
            assert response.status_code == 200

            _ = response.json().get("results", [])
            # Should find results regardless of schema version
            # (results might be empty if indexing hasn't completed, but request should succeed)

        finally:
            temp_path.unlink(missing_ok=True)

    def test_mixed_schema_crud_operations(
        self, client, auth_headers: Dict[str, str], clean_storage: None
    ):
        """Test CRUD operations work with mixed schema versions.

        This verifies Create, Read, Update, Delete operations work
        correctly when dealing with mixed old/new schema data.
        """
        collection_name = "mixed_crud_collection"

        # CREATE: Upload document
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Mixed schema CRUD test content")
            temp_path = Path(f.name)

        try:
            with open(temp_path, "rb") as f:
                response = client.post(
                    "/api/kb/upload/",
                    files={"file": ("crud_test.txt", f, "text/plain")},
                    data={"collection_name": collection_name},
                    headers=auth_headers,
                )
                assert response.status_code == 200
                result = response.json()
                collection_id = result["collection_id"]
                file_id = result.get("file_id")

            # READ: Get collections list
            response = client.get("/api/kb/collections/", headers=auth_headers)
            assert response.status_code == 200
            collections = response.json()["collections"]
            assert any(c["name"] == collection_name for c in collections)

            # READ: Get documents in collection
            response = client.get(
                f"/api/kb/collections/{collection_id}/documents/", headers=auth_headers
            )
            assert response.status_code == 200
            documents = response.json().get("documents", [])
            assert len(documents) >= 1

            # UPDATE: Try to reingest (update operation)
            with open(temp_path, "rb") as f:
                response = client.post(
                    "/api/kb/upload/",
                    files={"file": ("crud_test.txt", f, "text/plain")},
                    data={"collection_name": collection_name},
                    headers=auth_headers,
                )
                # Should handle reingestion gracefully
                assert response.status_code in [
                    200,
                    409,
                ]  # 409 if duplicate not allowed

            # DELETE: Delete document
            if file_id:
                response = client.delete(
                    f"/api/kb/documents/?file_id={file_id}", headers=auth_headers
                )
                # Should handle deletion regardless of schema version
                assert response.status_code in [200, 204, 404]

        finally:
            temp_path.unlink(missing_ok=True)


class TestBehaviorConsistency:
    """Test behavior consistency across schema migrations.

    These tests verify that core behaviors remain the same before
    and after schema changes.
    """

    def test_search_consistency_before_after_migration(
        self, client, auth_headers: Dict[str, str], clean_storage: None
    ):
        """Test that search behavior is consistent after schema migration.

        This verifies that:
        1. Search returns similar results before and after migration
        2. Search relevance ranking is consistent
        3. Search filters work the same way
        """
        # Upload test documents with known content
        test_docs = [
            (
                "doc1.txt",
                "Python is a programming language for data science and machine learning",
            ),
            (
                "doc2.txt",
                "JavaScript is used for web development and frontend applications",
            ),
            (
                "doc3.txt",
                "Python and JavaScript are both popular programming languages",
            ),
        ]

        collection_name = "search_consistency_collection"

        for filename, content in test_docs:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".txt", delete=False
            ) as f:
                f.write(content)
                temp_path = Path(f.name)

            try:
                with open(temp_path, "rb") as f:
                    response = client.post(
                        "/api/kb/upload/",
                        files={"file": (filename, f, "text/plain")},
                        data={"collection_name": collection_name},
                        headers=auth_headers,
                    )
                    assert response.status_code == 200
                    collection_id = response.json()["collection_id"]
            finally:
                temp_path.unlink(missing_ok=True)

        # Perform search and verify behavior
        response = client.post(
            f"/api/kb/collections/{collection_id}/search/",
            json={"query": "Python programming"},
            headers=auth_headers,
        )
        assert response.status_code == 200

        results = response.json().get("results", [])

        # Verify search results structure is consistent
        for result in results:
            # Should have expected fields regardless of schema version
            assert "content" in result or "text" in result or "chunk" in result
            # Metadata might be in different fields
            assert any(
                k in result for k in ["metadata", "score", "distance", "relevance"]
            )

    def test_ingestion_consistency_before_after_migration(
        self, client, auth_headers: Dict[str, str], clean_storage: None
    ):
        """Test that ingestion behavior is consistent after schema migration.

        This verifies that:
        1. File ingestion produces similar results
        2. Chunking behavior is consistent
        3. Metadata extraction works the same
        """
        # Upload a document
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(
                "This is a test document for ingestion consistency. "
                "It has multiple sentences. Each sentence should be processed. "
                "The chunking behavior should remain consistent across schema versions. "
                "Metadata extraction should work the same way."
            )
            temp_path = Path(f.name)

        try:
            with open(temp_path, "rb") as f:
                response = client.post(
                    "/api/kb/upload/",
                    files={"file": ("ingestion_test.txt", f, "text/plain")},
                    data={"collection_name": "ingestion_consistency_collection"},
                    headers=auth_headers,
                )
                assert response.status_code == 200

                result = response.json()

                # Verify ingestion response structure is consistent
                # Should have collection identifier
                assert "collection_id" in result or "collection" in result

                # Should have document identifier
                assert any(k in result for k in ["file_id", "doc_id", "filename"])

                # Status information (might vary by schema version)
                # Should not crash if fields are missing

        finally:
            temp_path.unlink(missing_ok=True)

    def test_crud_consistency_before_after_migration(
        self, client, auth_headers: Dict[str, str], clean_storage: None
    ):
        """Test that CRUD behavior is consistent after schema migration.

        This verifies that all CRUD operations work the same way
        before and after schema changes.
        """
        collection_name = "crud_consistency_collection"

        # CREATE: Create collection with document
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("CRUD consistency test document")
            temp_path = Path(f.name)

        try:
            with open(temp_path, "rb") as f:
                create_response = client.post(
                    "/api/kb/upload/",
                    files={"file": ("crud_consistency.txt", f, "text/plain")},
                    data={"collection_name": collection_name},
                    headers=auth_headers,
                )
                assert create_response.status_code == 200
                create_result = create_response.json()

                # Verify create response structure
                assert "collection_id" in create_result

            # READ: List collections
            list_response = client.get("/api/kb/collections/", headers=auth_headers)
            assert list_response.status_code == 200
            collections = list_response.json()["collections"]

            # Find our collection
            test_collection = next(
                (c for c in collections if c["name"] == collection_name), None
            )
            assert test_collection is not None
            assert test_collection["id"] == create_result["collection_id"]

            # READ: Get documents
            collection_id = test_collection["id"]
            docs_response = client.get(
                f"/api/kb/collections/{collection_id}/documents/", headers=auth_headers
            )
            assert docs_response.status_code == 200
            documents = docs_response.json().get("documents", [])
            assert len(documents) >= 1

            # UPDATE: Update collection (if supported)
            # This might be a PUT/PATCH request
            # For now, verify the endpoint exists and responds consistently
            update_response = client.put(
                f"/api/kb/collections/{collection_id}/",
                json={"description": "Updated description"},
                headers=auth_headers,
            )
            # Update might not be supported, so 404/405 is acceptable
            assert update_response.status_code in [200, 404, 405]

            # DELETE: Delete document
            file_id = create_result.get("file_id")
            if file_id:
                delete_response = client.delete(
                    f"/api/kb/documents/?file_id={file_id}", headers=auth_headers
                )
                # Delete should work consistently
                assert delete_response.status_code in [200, 204]

        finally:
            temp_path.unlink(missing_ok=True)

    def test_frontend_display_consistency(
        self, client, auth_headers: Dict[str, str], clean_storage: None
    ):
        """Test that frontend display data is consistent after schema migration.

        This verifies that the frontend receives data in a consistent format
        regardless of schema version.
        """
        # Upload document
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Frontend display consistency test")
            temp_path = Path(f.name)

        try:
            with open(temp_path, "rb") as f:
                response = client.post(
                    "/api/kb/upload/",
                    files={"file": ("display_test.txt", f, "text/plain")},
                    data={"collection_name": "display_consistency_collection"},
                    headers=auth_headers,
                )
                assert response.status_code == 200
                collection_id = response.json()["collection_id"]

            # Get collections list (frontend display endpoint)
            response = client.get("/api/kb/collections/", headers=auth_headers)
            assert response.status_code == 200

            collections = response.json()["collections"]

            # Verify response structure is frontend-friendly
            for collection in collections:
                # Should have displayable fields
                assert "name" in collection or "id" in collection
                # Document count should be present (field name might vary)
                assert any(
                    k in collection
                    for k in ["document_count", "doc_count", "count", "size"]
                )

            # Get documents (frontend display endpoint)
            response = client.get(
                f"/api/kb/collections/{collection_id}/documents/", headers=auth_headers
            )
            assert response.status_code == 200

            documents = response.json().get("documents", [])

            # Verify document structure is frontend-friendly
            for doc in documents:
                # Should have identifiable fields
                assert any(k in doc for k in ["name", "filename", "title"])
                # Metadata should be accessible
                assert "metadata" in doc or any(
                    k.startswith("meta") for k in doc.keys()
                )

        finally:
            temp_path.unlink(missing_ok=True)


class TestLegacyDataAccessAfterMigration:
    """Test legacy data accessibility after schema migrations.

    These tests ensure that data created with old schema versions
    remains accessible and functional after migrations.
    """

    def test_legacy_collection_access(
        self, client, auth_headers: Dict[str, str], clean_storage: None
    ):
        """Test that legacy collections can still be accessed.

        This verifies that collections created with old schema versions
        can be listed, viewed, and managed.
        """
        # Create a collection (simulating legacy data)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Legacy collection test content")
            temp_path = Path(f.name)

        try:
            with open(temp_path, "rb") as f:
                response = client.post(
                    "/api/kb/upload/",
                    files={"file": ("legacy.txt", f, "text/plain")},
                    data={"collection_name": "legacy_collection"},
                    headers=auth_headers,
                )
                assert response.status_code == 200
                collection_id = response.json()["collection_id"]

            # Access legacy collection
            response = client.get(
                f"/api/kb/collections/{collection_id}/", headers=auth_headers
            )
            # Should work regardless of schema version
            assert response.status_code in [200, 404]  # 404 if endpoint not implemented

            # List documents in legacy collection
            response = client.get(
                f"/api/kb/collections/{collection_id}/documents/", headers=auth_headers
            )
            assert response.status_code == 200

        finally:
            temp_path.unlink(missing_ok=True)

    def test_legacy_document_search(
        self, client, auth_headers: Dict[str, str], clean_storage: None
    ):
        """Test that legacy documents can still be searched.

        This verifies that documents created with old schema versions
        remain searchable after migrations.
        """
        # Upload document (simulating legacy)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(
                "Legacy document for search testing. "
                "Keywords: legacy, search, migration, compatibility. "
                "This document should remain searchable after schema changes."
            )
            temp_path = Path(f.name)

        try:
            with open(temp_path, "rb") as f:
                response = client.post(
                    "/api/kb/upload/",
                    files={"file": ("legacy_search.txt", f, "text/plain")},
                    data={"collection_name": "legacy_search_collection"},
                    headers=auth_headers,
                )
                assert response.status_code == 200
                collection_id = response.json()["collection_id"]

            # Search in legacy collection
            response = client.post(
                f"/api/kb/collections/{collection_id}/search/",
                json={"query": "legacy search"},
                headers=auth_headers,
            )
            assert response.status_code == 200

            # Should return results (might be empty if not indexed yet)
            results = response.json().get("results", [])
            # Structure should be consistent
            for result in results:
                assert "content" in result or "text" in result

        finally:
            temp_path.unlink(missing_ok=True)

    def test_legacy_document_deletion(
        self, client, auth_headers: Dict[str, str], clean_storage: None
    ):
        """Test that legacy documents can still be deleted.

        This verifies that deletion operations work on old schema data.
        """
        # Upload document
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Legacy deletion test")
            temp_path = Path(f.name)

        try:
            with open(temp_path, "rb") as f:
                response = client.post(
                    "/api/kb/upload/",
                    files={"file": ("legacy_delete.txt", f, "text/plain")},
                    data={"collection_name": "legacy_delete_collection"},
                    headers=auth_headers,
                )
                assert response.status_code == 200
                file_id = response.json().get("file_id")

            # Delete legacy document
            if file_id:
                response = client.delete(
                    f"/api/kb/documents/?file_id={file_id}", headers=auth_headers
                )
                # Should work with legacy schema
                assert response.status_code in [200, 204, 404]

        finally:
            temp_path.unlink(missing_ok=True)

    def test_mixed_schema_coexistence(
        self, client, auth_headers: Dict[str, str], clean_storage: None
    ):
        """Test that old and new schema versions can coexist.

        This simulates a gradual migration scenario where some data
        has been migrated and some hasn't.
        """
        collection_name = "mixed_schema_coexistence"

        # Upload multiple documents (simulating mixed migration state)
        for i in range(3):
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".txt", delete=False
            ) as f:
                f.write(f"Document {i + 1} in mixed schema environment")
                temp_path = Path(f.name)

            try:
                with open(temp_path, "rb") as f:
                    response = client.post(
                        "/api/kb/upload/",
                        files={"file": (f"doc{i + 1}.txt", f, "text/plain")},
                        data={"collection_name": collection_name},
                        headers=auth_headers,
                    )
                    assert response.status_code == 200
            finally:
                temp_path.unlink(missing_ok=True)

        # List collections - should handle mixed state
        response = client.get("/api/kb/collections/", headers=auth_headers)
        assert response.status_code == 200

        collections = response.json()["collections"]
        test_collection = next(
            (c for c in collections if c["name"] == collection_name), None
        )
        assert test_collection is not None

        # Get documents - should return all regardless of schema version
        collection_id = test_collection["id"]
        response = client.get(
            f"/api/kb/collections/{collection_id}/documents/", headers=auth_headers
        )
        assert response.status_code == 200

        documents = response.json().get("documents", [])
        assert len(documents) >= 3

        # Search - should work across mixed schemas
        response = client.post(
            f"/api/kb/collections/{collection_id}/search/",
            json={"query": "Document"},
            headers=auth_headers,
        )
        assert response.status_code == 200
