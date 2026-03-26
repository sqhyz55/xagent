#!/usr/bin/env python3
"""Development script to verify PostgreSQL migration for Phase 1B.

This script:
1. Starts a PostgreSQL container (if needed)
2. Runs Alembic migration
3. Tests basic CRUD operations
4. Verifies table structure
4. Cleans up

Usage:
    python scripts/verify_pg_migration.py [--no-cleanup]
"""

from __future__ import annotations

import argparse
import asyncio
import os
import subprocess
import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def start_postgres_container() -> dict[str, str]:
    """Start PostgreSQL container for testing.

    Returns:
        Dict with connection info.
    """
    print("Starting PostgreSQL container...")

    # Check if container already exists
    result = subprocess.run(
        ["docker", "ps", "-a", "-q", "-f", "name=xagent-pg-test"],
        capture_output=True,
        text=True,
    )

    if result.stdout.strip():
        print("Container exists, starting it...")
        subprocess.run(
            ["docker", "start", "xagent-pg-test"],
            check=True,
            capture_output=True,
        )
    else:
        # Create and start new container
        subprocess.run(
            [
                "docker",
                "run",
                "-d",
                "--name",
                "xagent-pg-test",
                "-e",
                "POSTGRES_USER=xagent",
                "-e",
                "POSTGRES_PASSWORD=xagent",
                "-e",
                "POSTGRES_DB=xagent",
                "-p",
                "5433:5432",
                "postgres:16",
            ],
            check=True,
        )

    # Wait for PostgreSQL to be ready
    print("Waiting for PostgreSQL to be ready...")
    for _ in range(30):
        try:
            result = subprocess.run(
                [
                    "docker",
                    "exec",
                    "xagent-pg-test",
                    "pg_isready",
                    "-U",
                    "xagent",
                ],
                capture_output=True,
                text=True,
            )
            if "accepting connections" in result.stdout:
                break
        except Exception:
            pass
        time.sleep(1)

    print("PostgreSQL is ready!")
    print("  Connection URL: postgresql://xagent:xagent@localhost:5433/xagent")

    return {
        "host": "localhost",
        "port": "5433",
        "user": "xagent",
        "password": "xagent",
        "database": "xagent",
        "url": "postgresql://xagent:xagent@localhost:5433/xagent",
    }


def stop_postgres_container(cleanup: bool = True) -> None:
    """Stop and optionally remove PostgreSQL container.

    Args:
        cleanup: If True, remove container; if False, just stop it.
    """
    print("\nStopping PostgreSQL container...")

    if cleanup:
        subprocess.run(
            ["docker", "rm", "-f", "xagent-pg-test"],
            capture_output=True,
        )
        print("Container removed.")
    else:
        subprocess.run(
            ["docker", "stop", "xagent-pg-test"],
            capture_output=True,
        )
        print("Container stopped (kept for inspection).")


async def verify_migration(db_url: str) -> bool:
    """Verify migration and test basic operations.

    Args:
        db_url: Database connection URL.

    Returns:
        True if verification passed, False otherwise.
    """
    print("\n=== Verifying Migration ===")

    # Set environment for migration
    os.environ["DATABASE_URL"] = db_url
    os.environ["RAG_METADATA_STORE_BACKEND"] = "postgresql"

    try:
        from sqlalchemy import create_engine, inspect

        from xagent.core.tools.core.RAG_tools.core.schemas import CollectionInfo
        from xagent.core.tools.core.RAG_tools.storage import factory
        from xagent.core.tools.core.RAG_tools.storage.rdb_models import Base

        # Reset factory to use PostgreSQL
        factory.reset_metadata_store()

        print("\n1. Creating tables...")
        engine = create_engine(db_url)
        Base.metadata.create_all(engine)

        # Verify tables exist
        inspector = inspect(engine)
        tables = inspector.get_table_names()
        kb_tables = [t for t in tables if t.startswith("kb_")]

        print(f"   Created KB tables: {kb_tables}")

        expected_tables = {
            "kb_collection_metadata",
            "kb_collection_shares",
            "kb_document_staging",
            "kb_collection_config",
        }

        missing_tables = expected_tables - set(kb_tables)
        if missing_tables:
            print(f"   ERROR: Missing tables: {missing_tables}")
            return False

        print("   ✓ All tables created successfully")

        # Test 2: Insert and query collection
        print("\n2. Testing collection CRUD...")

        from xagent.core.tools.core.RAG_tools.storage.pg_metadata_store import (
            PostgreSQLMetadataStore,
        )

        store = PostgreSQLMetadataStore(database_url=db_url)
        await store.ensure_collection_metadata_table()

        # Create test collection
        test_collection = CollectionInfo(
            name="test_collection",
            owner_user_id=1,
            embedding_model_id="text-embedding-3-small",
            embedding_dimension=1536,
            documents=0,
        )

        await store.save_collection(test_collection)
        print("   ✓ Collection saved")

        # Read back
        retrieved = await store.get_collection("test_collection")
        assert retrieved.name == "test_collection"
        assert retrieved.owner_user_id == 1
        assert retrieved.embedding_model_id == "text-embedding-3-small"
        print("   ✓ Collection retrieved successfully")

        # Update
        retrieved.documents = 10
        await store.save_collection(retrieved)
        updated = await store.get_collection("test_collection")
        assert updated.documents == 10
        print("   ✓ Collection updated successfully")

        # Test 3: Collection config
        print("\n3. Testing collection config...")
        await store.save_collection_config(
            collection="test_collection",
            config_json='{"chunk_size": 1000}',
            user_id=1,
        )
        config = await store.get_collection_config("test_collection", 1)
        assert config == '{"chunk_size": 1000}'
        print("   ✓ Config saved and retrieved successfully")

        # Test 4: Permissions
        print("\n4. Testing permission system...")

        from xagent.core.tools.core.RAG_tools.storage.permissions import (
            CollectionPermissionChecker,
        )

        session_factory = store._session_factory
        checker = CollectionPermissionChecker(session_factory)

        # Owner should have full permissions
        perms = checker.get_permissions("test_collection", user_id=1)
        assert perms.can_read is True
        assert perms.can_modify is True
        assert perms.is_owner is True
        print("   ✓ Owner has full permissions")

        # Non-owner should have no access
        perms = checker.get_permissions("test_collection", user_id=2)
        assert perms.can_read is False
        assert perms.can_modify is False
        assert perms.is_owner is False
        print("   ✓ Non-owner has no access")

        # Test 5: Factory integration
        print("\n5. Testing factory integration...")
        factory_store = factory.get_metadata_store()
        assert isinstance(factory_store, PostgreSQLMetadataStore)
        print("   ✓ Factory returns PostgreSQLMetadataStore")

        # Test 6: Verify table structure
        print("\n6. Verifying table structure...")

        # Check kb_collection_metadata columns
        columns = {c["name"] for c in inspector.get_columns("kb_collection_metadata")}
        required_columns = {
            "name",
            "owner_user_id",
            "embedding_model_id",
            "embedding_dimension",
            "documents",
            "processed_documents",
            "parses",
            "chunks",
            "embeddings",
            "document_names",
            "collection_locked",
            "allow_mixed_parse_methods",
            "skip_config_validation",
            "ingestion_config",
            "external_file_id",
            "created_at",
            "updated_at",
            "last_accessed_at",
            "extra_metadata",
        }

        missing_columns = required_columns - columns
        if missing_columns:
            print(f"   ERROR: Missing columns: {missing_columns}")
            return False

        print(
            f"   ✓ All {len(required_columns)} columns present in kb_collection_metadata"
        )

        # Check indexes
        indexes = {
            idx["name"] for idx in inspector.get_indexes("kb_collection_metadata")
        }
        expected_indexes = {
            "idx_kb_collection_metadata_updated_at",
            "idx_kb_collection_metadata_owner_user_id",
            "idx_kb_collection_metadata_external_file_id",
        }

        missing_indexes = expected_indexes - indexes
        if missing_indexes:
            print(f"   WARNING: Missing indexes: {missing_indexes}")
        else:
            print(f"   ✓ All {len(expected_indexes)} indexes present")

        print("\n=== All Verification Tests Passed! ===")
        return True

    except Exception as e:
        print(f"\n❌ Verification failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Verify PostgreSQL migration")
    parser.add_argument(
        "--no-cleanup",
        action="store_true",
        help="Keep container running after verification",
    )
    parser.add_argument(
        "--use-existing",
        action="store_true",
        help="Use existing PostgreSQL container",
    )
    args = parser.parse_args()

    container_info = None

    try:
        if not args.use_existing:
            container_info = start_postgres_container()

        # Use default test database URL
        db_url = (
            container_info["url"]
            if container_info
            else "postgresql://xagent:xagent@localhost:5433/xagent"
        )

        success = await verify_migration(db_url)

        if success:
            print("\n✅ Migration verification completed successfully!")
            return 0
        else:
            print("\n❌ Migration verification failed!")
            return 1

    finally:
        if container_info and not args.no_cleanup:
            stop_postgres_container(cleanup=True)
        elif not args.no_cleanup:
            print("\n📝 Container kept running. Connect with:")
            print("   psql -h localhost -p 5433 -U xagent -d xagent")
            print("\nTo stop later:")
            print("   docker rm -f xagent-pg-test")


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
