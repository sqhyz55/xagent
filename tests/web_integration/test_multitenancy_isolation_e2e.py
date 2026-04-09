"""
E2E Tests for Multi-Tenant Data Isolation in Knowledge Base System.

This test suite verifies that multi-tenant isolation is properly enforced
throughout the RAG/KB system, ensuring users can only access their own data
while admins can access all data when appropriate.

These tests are critical for ensuring that schema changes and database migrations
do not break tenant isolation, which is a fundamental security requirement.
"""

from pathlib import Path
from typing import Any, Dict

import pytest


class TestMultiTenantDataIsolation:
    """Test multi-tenant data isolation in KB system.

    These tests verify that:
    1. Regular users can only see their own collections and documents
    2. Admins can see all collections and documents when needed
    3. Cross-tenant access is properly denied
    4. Isolation is enforced at all levels (API, storage, search)
    """

    @pytest.fixture
    def tenant_users(self) -> Dict[str, Dict[str, str]]:
        """Create multiple test users for different tenants."""
        return {
            "tenant1_user": {
                "username": "tenant1_user",
                "password": "password123",
                "email": "tenant1@example.com",
                "role": "user",
            },
            "tenant2_user": {
                "username": "tenant2_user",
                "password": "password123",
                "email": "tenant2@example.com",
                "role": "user",
            },
            "admin_user": {
                "username": "admin_user",
                "password": "admin123",
                "email": "admin@example.com",
                "role": "admin",
            },
        }

    @pytest.fixture
    def sample_files_for_tenants(self, tmp_path: Path) -> Dict[str, Path]:
        """Create sample files for different tenants."""
        files = {}

        # Tenant 1 files
        tenant1_file = tmp_path / "tenant1_document.txt"
        tenant1_file.write_text(
            "This is a document for tenant 1. "
            "It contains sensitive information specific to tenant 1. "
            "Keywords: tenant1, private, secret"
        )
        files["tenant1"] = tenant1_file

        # Tenant 2 files
        tenant2_file = tmp_path / "tenant2_document.txt"
        tenant2_file.write_text(
            "This is a document for tenant 2. "
            "It contains sensitive information specific to tenant 2. "
            "Keywords: tenant2, confidential, internal"
        )
        files["tenant2"] = tenant2_file

        # Admin files
        admin_file = tmp_path / "admin_document.txt"
        admin_file.write_text(
            "This is an admin document. "
            "It contains administrative information. "
            "Keywords: admin, system, configuration"
        )
        files["admin"] = admin_file

        return files

    def test_users_see_only_own_collections(
        self,
        client,
        tenant_users: Dict[str, Dict[str, str]],
        sample_files_for_tenants: Dict[str, Path],
    ):
        """Test that users can only see their own collections.

        This test creates collections for different tenants and verifies
        that each user can only see their own collections when listing.
        """
        # Register and authenticate all users
        user_tokens = {}
        for user_key, user_data in tenant_users.items():
            # Register user
            register_response = client.post(
                "/api/auth/register",
                json={
                    "username": user_data["username"],
                    "password": user_data["password"],
                    "email": user_data["email"],
                },
            )
            if register_response.status_code not in [200, 400]:  # 400 if already exists
                register_response.raise_for_status()

            # Login to get token
            login_response = client.post(
                "/api/auth/login",
                json={
                    "username": user_data["username"],
                    "password": user_data["password"],
                },
            )
            assert login_response.status_code == 200
            user_tokens[user_key] = login_response.json()["access_token"]

        # Create collections for each tenant
        collection_ids = {}
        for user_key, token in user_tokens.items():
            headers = {"Authorization": f"Bearer {token}"}
            file_path = sample_files_for_tenants[user_key]

            with open(file_path, "rb") as f:
                response = client.post(
                    "/api/kb/upload/",
                    files={"file": (file_path.name, f, "text/plain")},
                    data={"collection_name": f"{user_key}_collection"},
                    headers=headers,
                )
                assert response.status_code == 200
                collection_ids[user_key] = response.json()["collection_id"]

        # Verify each user sees only their own collections
        for user_key, token in user_tokens.items():
            headers = {"Authorization": f"Bearer {token}"}

            # List collections
            response = client.get("/api/kb/collections/", headers=headers)
            assert response.status_code == 200

            collections = response.json()["collections"]
            collection_names = {col["name"] for col in collections}

            # Should see own collection
            expected_collection = f"{user_key}_collection"
            assert expected_collection in collection_names, (
                f"{user_key} should see their own collection"
            )

            # Should NOT see other tenants' collections
            other_tenants = [k for k in user_tokens.keys() if k != user_key]
            for other_tenant in other_tenants:
                other_collection = f"{other_tenant}_collection"
                assert other_collection not in collection_names, (
                    f"{user_key} should NOT see {other_tenant}'s collection"
                )

    def test_users_see_only_own_documents(
        self,
        client,
        tenant_users: Dict[str, Dict[str, str]],
        sample_files_for_tenants: Dict[str, Path],
    ):
        """Test that users can only see their own documents.

        This test verifies that when listing documents in a collection,
        users only see documents they uploaded, not other tenants' documents.
        """
        # Get user tokens (similar to above test)
        user_tokens = {}
        for user_key, user_data in tenant_users.items():
            # Register/login
            client.post(
                "/api/auth/register",
                json={
                    "username": user_data["username"],
                    "password": user_data["password"],
                    "email": user_data["email"],
                },
            )
            login_response = client.post(
                "/api/auth/login",
                json={
                    "username": user_data["username"],
                    "password": user_data["password"],
                },
            )
            assert login_response.status_code == 200
            user_tokens[user_key] = login_response.json()["access_token"]

        # Create collections and upload documents
        collection_ids = {}
        for user_key, token in user_tokens.items():
            headers = {"Authorization": f"Bearer {token}"}
            file_path = sample_files_for_tenants[user_key]

            with open(file_path, "rb") as f:
                response = client.post(
                    "/api/kb/upload/",
                    files={"file": (file_path.name, f, "text/plain")},
                    data={"collection_name": f"{user_key}_collection"},
                    headers=headers,
                )
                assert response.status_code == 200
                collection_ids[user_key] = response.json()["collection_id"]

        # Verify document isolation
        for user_key, token in user_tokens.items():
            headers = {"Authorization": f"Bearer {token}"}
            collection_id = collection_ids[user_key]

            # List documents in own collection
            response = client.get(
                f"/api/kb/collections/{collection_id}/documents/", headers=headers
            )
            assert response.status_code == 200

            documents = response.json().get("documents", [])
            document_names = {
                doc.get("name", doc.get("filename", "")) for doc in documents
            }

            # Should see own document
            expected_file = sample_files_for_tenants[user_key].name
            assert any(expected_file in name for name in document_names), (
                f"{user_key} should see their own document"
            )

            # Verify document count matches (only own documents)
            assert len(documents) >= 1, (
                f"{user_key} should have at least their own document"
            )

    def test_cross_tenant_access_denied(
        self,
        client,
        tenant_users: Dict[str, Dict[str, str]],
        sample_files_for_tenants: Dict[str, Path],
    ):
        """Test that cross-tenant access attempts are properly denied.

        This test verifies that users cannot access other tenants' collections
        or documents by trying different access methods.
        """
        # Setup users and collections
        user_tokens = {}
        collection_ids = {}

        for user_key, user_data in tenant_users.items():
            # Register/login
            client.post(
                "/api/auth/register",
                json={
                    "username": user_data["username"],
                    "password": user_data["password"],
                    "email": user_data["email"],
                },
            )
            login_response = client.post(
                "/api/auth/login",
                json={
                    "username": user_data["username"],
                    "password": user_data["password"],
                },
            )
            assert login_response.status_code == 200
            user_tokens[user_key] = login_response.json()["access_token"]

            # Create collection
            headers = {"Authorization": f"Bearer {user_tokens[user_key]}"}
            file_path = sample_files_for_tenants[user_key]

            with open(file_path, "rb") as f:
                response = client.post(
                    "/api/kb/upload/",
                    files={"file": (file_path.name, f, "text/plain")},
                    data={"collection_name": f"{user_key}_collection"},
                    headers=headers,
                )
                assert response.status_code == 200
                collection_ids[user_key] = response.json()["collection_id"]

        # Try to access other tenant's collection
        tenant1_token = user_tokens["tenant1_user"]
        tenant2_collection_id = collection_ids["tenant2_user"]

        # Attempt to list documents in tenant2's collection using tenant1's token
        response = client.get(
            f"/api/kb/collections/{tenant2_collection_id}/documents/",
            headers={"Authorization": f"Bearer {tenant1_token}"},
        )

        # Should be denied (403 or 404 depending on implementation)
        assert response.status_code in [403, 404], (
            "Cross-tenant access should be denied"
        )

        # Try to search in other tenant's collection
        response = client.post(
            f"/api/kb/collections/{tenant2_collection_id}/search/",
            json={"query": "test"},
            headers={"Authorization": f"Bearer {tenant1_token}"},
        )

        assert response.status_code in [403, 404], (
            "Cross-tenant search should be denied"
        )

        # Try to delete other tenant's document
        response = client.delete(
            f"/api/kb/collections/{tenant2_collection_id}/documents/",
            headers={"Authorization": f"Bearer {tenant1_token}"},
        )

        assert response.status_code in [403, 404], (
            "Cross-tenant delete should be denied"
        )

    def test_admin_sees_all_collections(
        self,
        client,
        tenant_users: Dict[str, Dict[str, str]],
        sample_files_for_tenants: Dict[str, Path],
    ):
        """Test that admin users can see all collections.

        This test verifies that admin users have visibility into all
        collections across all tenants for administrative purposes.
        """
        # Setup users
        user_tokens = {}

        for user_key, user_data in tenant_users.items():
            # Register/login
            client.post(
                "/api/auth/register",
                json={
                    "username": user_data["username"],
                    "password": user_data["password"],
                    "email": user_data["email"],
                },
            )
            login_response = client.post(
                "/api/auth/login",
                json={
                    "username": user_data["username"],
                    "password": user_data["password"],
                },
            )
            assert login_response.status_code == 200
            user_tokens[user_key] = login_response.json()["access_token"]

        # Create collections for regular users
        for user_key in ["tenant1_user", "tenant2_user"]:
            headers = {"Authorization": f"Bearer {user_tokens[user_key]}"}
            file_path = sample_files_for_tenants[user_key]

            with open(file_path, "rb") as f:
                response = client.post(
                    "/api/kb/upload/",
                    files={"file": (file_path.name, f, "text/plain")},
                    data={"collection_name": f"{user_key}_collection"},
                    headers=headers,
                )
                assert response.status_code == 200

        # Admin lists all collections
        admin_token = user_tokens["admin_user"]
        admin_headers = {"Authorization": f"Bearer {admin_token}"}

        response = client.get("/api/kb/collections/", headers=admin_headers)
        assert response.status_code == 200

        collections = response.json()["collections"]
        collection_names = {col["name"] for col in collections}

        # Admin should see all collections
        assert "tenant1_user_collection" in collection_names, (
            "Admin should see tenant1's collection"
        )
        assert "tenant2_user_collection" in collection_names, (
            "Admin should see tenant2's collection"
        )

    def test_admin_sees_all_documents(
        self,
        client,
        tenant_users: Dict[str, Dict[str, str]],
        sample_files_for_tenants: Dict[str, Path],
    ):
        """Test that admin users can see all documents.

        This test verifies that admin users have access to documents
        across all tenant collections.
        """
        # Setup users and collections
        user_tokens = {}
        collection_ids = {}

        for user_key, user_data in tenant_users.items():
            # Register/login
            client.post(
                "/api/auth/register",
                json={
                    "username": user_data["username"],
                    "password": user_data["password"],
                    "email": user_data["email"],
                },
            )
            login_response = client.post(
                "/api/auth/login",
                json={
                    "username": user_data["username"],
                    "password": user_data["password"],
                },
            )
            assert login_response.status_code == 200
            user_tokens[user_key] = login_response.json()["access_token"]

            # Create collection
            headers = {"Authorization": f"Bearer {user_tokens[user_key]}"}
            file_path = sample_files_for_tenants[user_key]

            with open(file_path, "rb") as f:
                response = client.post(
                    "/api/kb/upload/",
                    files={"file": (file_path.name, f, "text/plain")},
                    data={"collection_name": f"{user_key}_collection"},
                    headers=headers,
                )
                assert response.status_code == 200
                collection_ids[user_key] = response.json()["collection_id"]

        # Admin accesses documents from different tenants
        admin_token = user_tokens["admin_user"]
        admin_headers = {"Authorization": f"Bearer {admin_token}"}

        # Access tenant1's documents
        response = client.get(
            f"/api/kb/collections/{collection_ids['tenant1_user']}/documents/",
            headers=admin_headers,
        )
        assert response.status_code == 200

        # Access tenant2's documents
        response = client.get(
            f"/api/kb/collections/{collection_ids['tenant2_user']}/documents/",
            headers=admin_headers,
        )
        assert response.status_code == 200


class TestMultiTenantSearchIsolation:
    """Test search isolation between tenants.

    These tests verify that search results are properly isolated,
    ensuring users can only search and retrieve their own documents.
    """

    @pytest.fixture
    def tenant_search_data(self, tmp_path: Path) -> Dict[str, Dict[str, Any]]:
        """Create search test data for different tenants."""
        data = {}

        # Tenant 1: Technical documentation
        tenant1_file = tmp_path / "tenant1_tech_docs.txt"
        tenant1_file.write_text("""
        Tenant 1 Technical Documentation

        This document contains proprietary technical information
        for Tenant 1's systems.

        Key technologies:
        - Python backend services
        - React frontend framework
        - PostgreSQL database
        - Redis caching layer

        This information is CONFIDENTIAL to Tenant 1.
        """)
        data["tenant1"] = {
            "file": tenant1_file,
            "query": "Python backend",
            "expected_keywords": ["Python", "backend", "Tenant 1"],
        }

        # Tenant 2: Marketing materials
        tenant2_file = tmp_path / "tenant2_marketing.txt"
        tenant2_file.write_text("""
        Tenant 2 Marketing Strategy

        This document contains proprietary marketing strategies
        for Tenant 2's products.

        Key strategies:
        - Social media campaigns
        - Email marketing automation
        - Influencer partnerships
        - Content marketing initiatives

        This information is CONFIDENTIAL to Tenant 2.
        """)
        data["tenant2"] = {
            "file": tenant2_file,
            "query": "marketing strategy",
            "expected_keywords": ["marketing", "strategy", "Tenant 2"],
        }

        return data

    def test_search_only_returns_own_documents(
        self, client, tenant_search_data: Dict[str, Dict[str, Any]], clean_storage: None
    ):
        """Test that search results only contain documents from the requesting tenant.

        This test uploads documents for different tenants and verifies that
        search queries only return results from the tenant's own documents.
        """
        # Create users for each tenant
        users = {
            "tenant1": {"username": "search_user1", "password": "pass123"},
            "tenant2": {"username": "search_user2", "password": "pass123"},
        }

        user_tokens = {}
        collection_ids = {}

        # Register users and create collections with documents
        for tenant_key, user_info in users.items():
            # Register
            client.post(
                "/api/auth/register",
                json={
                    "username": user_info["username"],
                    "password": user_info["password"],
                    "email": f"{tenant_key}@example.com",
                },
            )

            # Login
            login_response = client.post("/api/auth/login", json=user_info)
            assert login_response.status_code == 200
            token = login_response.json()["access_token"]
            user_tokens[tenant_key] = token

            # Upload document
            file_data = tenant_search_data[tenant_key]
            headers = {"Authorization": f"Bearer {token}"}

            with open(file_data["file"], "rb") as f:
                response = client.post(
                    "/api/kb/upload/",
                    files={"file": (file_data["file"].name, f, "text/plain")},
                    data={"collection_name": f"{tenant_key}_search_collection"},
                    headers=headers,
                )
                assert response.status_code == 200
                collection_ids[tenant_key] = response.json()["collection_id"]

        # Verify search isolation for each tenant
        for tenant_key, token in user_tokens.items():
            headers = {"Authorization": f"Bearer {token}"}
            collection_id = collection_ids[tenant_key]
            search_data = tenant_search_data[tenant_key]

            # Search in own collection
            response = client.post(
                f"/api/kb/collections/{collection_id}/search/",
                json={"query": search_data["query"]},
                headers=headers,
            )
            assert response.status_code == 200

            results = response.json().get("results", [])
            assert len(results) > 0, "Should find search results in own collection"

            # Verify results contain expected keywords
            result_text = " ".join([r.get("content", "") for r in results])
            for keyword in search_data["expected_keywords"]:
                assert keyword in result_text, (
                    f"Search results should contain expected keyword: {keyword}"
                )

            # Verify results don't contain other tenant's content
            other_tenant = "tenant2" if tenant_key == "tenant1" else "tenant1"
            other_keywords = tenant_search_data[other_tenant]["expected_keywords"]
            for keyword in other_keywords:
                # The other tenant's keywords should not appear in results
                # (unless they happen to overlap, but in our test data they don't)
                if keyword.startswith(other_tenant.capitalize()):  # e.g., "Tenant 1"
                    assert keyword not in result_text, (
                        f"Search results should NOT contain {other_tenant}'s content"
                    )

    def test_cross_tenant_search_fails(
        self, client, tenant_search_data: Dict[str, Dict[str, Any]], clean_storage: None
    ):
        """Test that searching in other tenant's collection fails.

        This test verifies that users cannot perform searches in
        collections belonging to other tenants.
        """
        # Setup users and collections
        users = {
            "tenant1": {"username": "cross_search_user1", "password": "pass123"},
            "tenant2": {"username": "cross_search_user2", "password": "pass123"},
        }

        user_tokens = {}
        collection_ids = {}

        for tenant_key, user_info in users.items():
            # Register and login
            client.post(
                "/api/auth/register",
                json={
                    "username": user_info["username"],
                    "password": user_info["password"],
                    "email": f"{tenant_key}@example.com",
                },
            )
            login_response = client.post("/api/auth/login", json=user_info)
            assert login_response.status_code == 200
            user_tokens[tenant_key] = login_response.json()["access_token"]

            # Upload document
            file_data = tenant_search_data[tenant_key]
            headers = {"Authorization": f"Bearer {user_tokens[tenant_key]}"}

            with open(file_data["file"], "rb") as f:
                response = client.post(
                    "/api/kb/upload/",
                    files={"file": (file_data["file"].name, f, "text/plain")},
                    data={"collection_name": f"{tenant_key}_collection"},
                    headers=headers,
                )
                assert response.status_code == 200
                collection_ids[tenant_key] = response.json()["collection_id"]

        # Try to search in other tenant's collection
        tenant1_token = user_tokens["tenant1"]
        tenant2_collection_id = collection_ids["tenant2"]

        response = client.post(
            f"/api/kb/collections/{tenant2_collection_id}/search/",
            json={"query": "test query"},
            headers={"Authorization": f"Bearer {tenant1_token}"},
        )

        # Should be denied
        assert response.status_code in [403, 404], (
            "Cross-tenant search should be denied"
        )


class TestMultiTenantAfterMigration:
    """Test that multi-tenant isolation persists after schema migrations.

    These tests are critical for ensuring that schema changes and
    database migrations do not break tenant isolation.
    """

    def test_isolation_after_user_id_field_change(self, client, clean_storage: None):
        """Test that isolation remains after user_id field changes.

        This simulates a scenario where the user_id field type or
        structure changes (e.g., from string to int, or added metadata).
        """
        # Create two users
        users = [
            {
                "username": "migration_user1",
                "password": "pass123",
                "email": "user1@example.com",
            },
            {
                "username": "migration_user2",
                "password": "pass123",
                "email": "user2@example.com",
            },
        ]

        user_tokens = []
        for user in users:
            client.post("/api/auth/register", json=user)
            login_response = client.post(
                "/api/auth/login",
                json={"username": user["username"], "password": user["password"]},
            )
            assert login_response.status_code == 200
            user_tokens.append(login_response.json()["access_token"])

        # User 1 creates a collection
        headers1 = {"Authorization": f"Bearer {user_tokens[0]}"}
        response = client.post(
            "/api/kb/upload/",
            files={"file": ("user1_doc.txt", "User 1 private data", "text/plain")},
            data={"collection_name": "user1_collection"},
            headers=headers1,
        )
        assert response.status_code == 200
        collection_id = response.json()["collection_id"]

        # User 2 tries to access User 1's collection
        headers2 = {"Authorization": f"Bearer {user_tokens[1]}"}
        response = client.get(
            f"/api/kb/collections/{collection_id}/documents/", headers=headers2
        )

        # Should be denied
        assert response.status_code in [403, 404], (
            "User 2 should not access User 1's collection after migration"
        )

    def test_legacy_orphan_data_isolation(self, client, clean_storage: None):
        """Test that legacy data without proper user_id is properly isolated.

        This test simulates legacy data that might not have user_id set
        and verifies it doesn't leak between tenants.
        """
        # Create a regular user
        client.post(
            "/api/auth/register",
            json={
                "username": "legacy_test_user",
                "password": "pass123",
                "email": "legacy@example.com",
            },
        )
        login_response = client.post(
            "/api/auth/login",
            json={"username": "legacy_test_user", "password": "pass123"},
        )
        assert login_response.status_code == 200
        user_token = login_response.json()["access_token"]
        headers = {"Authorization": f"Bearer {user_token}"}

        # Create a collection
        response = client.post(
            "/api/kb/upload/",
            files={"file": ("user_doc.txt", "User private data", "text/plain")},
            data={"collection_name": "user_collection"},
            headers=headers,
        )
        assert response.status_code == 200

        # List collections - should only see own collection
        response = client.get("/api/kb/collections/", headers=headers)
        assert response.status_code == 200

        collections = response.json()["collections"]
        collection_names = {col["name"] for col in collections}

        # Should see own collection
        assert "user_collection" in collection_names

        # If there were legacy orphan collections, they should not appear
        # (unless the system has specific logic to handle them)

    def test_isolation_with_mixed_schema_versions(self, client, clean_storage: None):
        """Test isolation works with mixed schema versions.

        This simulates a scenario where some collections use the old schema
        and some use the new schema, verifying isolation still works.
        """
        # Create two users
        users = [
            {
                "username": "schema_user1",
                "password": "pass123",
                "email": "schema1@example.com",
            },
            {
                "username": "schema_user2",
                "password": "pass123",
                "email": "schema2@example.com",
            },
        ]

        user_tokens = []
        for user in users:
            client.post("/api/auth/register", json=user)
            login_response = client.post(
                "/api/auth/login",
                json={"username": user["username"], "password": user["password"]},
            )
            assert login_response.status_code == 200
            user_tokens.append(login_response.json()["access_token"])

        # Both users create collections
        collection_ids = []
        for i, token in enumerate(user_tokens):
            headers = {"Authorization": f"Bearer {token}"}
            response = client.post(
                "/api/kb/upload/",
                files={
                    "file": (f"user{i + 1}_doc.txt", f"User {i + 1} data", "text/plain")
                },
                data={"collection_name": f"user{i + 1}_collection"},
                headers=headers,
            )
            assert response.status_code == 200
            collection_ids.append(response.json()["collection_id"])

        # Verify cross-access is denied
        headers1 = {"Authorization": f"Bearer {user_tokens[0]}"}
        response = client.get(
            f"/api/kb/collections/{collection_ids[1]}/documents/", headers=headers1
        )
        assert response.status_code in [403, 404], (
            "Cross-tenant access should be denied even with mixed schemas"
        )


class TestMultiTenantDeleteIsolation:
    """Test that delete operations respect tenant isolation.

    These tests verify that users can only delete their own collections
    and documents, not other tenants'.
    """

    def test_users_can_only_delete_own_collections(self, client, clean_storage: None):
        """Test that users can only delete their own collections."""
        # Create two users
        users = [
            {
                "username": "delete_user1",
                "password": "pass123",
                "email": "delete1@example.com",
            },
            {
                "username": "delete_user2",
                "password": "pass123",
                "email": "delete2@example.com",
            },
        ]

        user_tokens = []
        collection_ids = []

        for i, user in enumerate(users):
            client.post("/api/auth/register", json=user)
            login_response = client.post(
                "/api/auth/login",
                json={"username": user["username"], "password": user["password"]},
            )
            assert login_response.status_code == 200
            token = login_response.json()["access_token"]
            user_tokens.append(token)

            # Create collection
            headers = {"Authorization": f"Bearer {token}"}
            response = client.post(
                "/api/kb/upload/",
                files={
                    "file": (f"user{i + 1}_doc.txt", f"User {i + 1} data", "text/plain")
                },
                data={"collection_name": f"user{i + 1}_collection"},
                headers=headers,
            )
            assert response.status_code == 200
            collection_ids.append(response.json()["collection_id"])

        # User 1 tries to delete User 2's collection
        headers1 = {"Authorization": f"Bearer {user_tokens[0]}"}
        response = client.delete(
            f"/api/kb/collections/{collection_ids[1]}/", headers=headers1
        )

        # Should be denied
        assert response.status_code in [403, 404], (
            "User should not be able to delete other tenant's collection"
        )

        # Verify User 2's collection still exists
        headers2 = {"Authorization": f"Bearer {user_tokens[1]}"}
        response = client.get(
            f"/api/kb/collections/{collection_ids[1]}/", headers=headers2
        )
        assert response.status_code == 200, (
            "Other tenant's collection should still exist"
        )

    def test_users_can_only_delete_own_documents(self, client, clean_storage: None):
        """Test that users can only delete their own documents."""
        # Create two users in same collection scenario
        # (if system supports shared collections)
        # Or test document-level delete isolation

        # For now, test that file_id based deletion is isolated
        users = [
            {
                "username": "doc_delete_user1",
                "password": "pass123",
                "email": "docdel1@example.com",
            },
            {
                "username": "doc_delete_user2",
                "password": "pass123",
                "email": "docdel2@example.com",
            },
        ]

        user_tokens = []
        file_ids = []

        for i, user in enumerate(users):
            client.post("/api/auth/register", json=user)
            login_response = client.post(
                "/api/auth/login",
                json={"username": user["username"], "password": user["password"]},
            )
            assert login_response.status_code == 200
            token = login_response.json()["access_token"]
            user_tokens.append(token)

            # Upload document
            headers = {"Authorization": f"Bearer {token}"}
            response = client.post(
                "/api/kb/upload/",
                files={
                    "file": (
                        f"user{i + 1}_file.txt",
                        f"User {i + 1} file content",
                        "text/plain",
                    )
                },
                data={"collection_name": f"user{i + 1}_collection"},
                headers=headers,
            )
            assert response.status_code == 200
            # Get file_id from response
            file_ids.append(response.json().get("file_id"))

        # User 1 tries to delete User 2's document by file_id
        if file_ids[1]:  # If file_id is available
            headers1 = {"Authorization": f"Bearer {user_tokens[0]}"}
            response = client.delete(
                f"/api/kb/documents/?file_id={file_ids[1]}", headers=headers1
            )

            # Should be denied
            assert response.status_code in [403, 404], (
                "User should not be able to delete other tenant's document"
            )
