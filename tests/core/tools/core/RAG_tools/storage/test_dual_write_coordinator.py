"""Tests for DualWriteCoordinator (Phase 1B.5).

Tests cover:
- Dual-write coordinator initialization
- Backfill operations
- Reconcile operations
- Statistics tracking
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from xagent.core.tools.core.RAG_tools.core.schemas import CollectionInfo
from xagent.core.tools.core.RAG_tools.storage.dual_write_coordinator import (
    DualWriteCoordinator,
    DualWriteMetadataStore,
    DualWriteStats,
    MetadataBackend,
    ReconcileResult,
)
from xagent.core.tools.core.RAG_tools.storage.lancedb_stores import LanceDBMetadataStore


class TestDualWriteStats:
    """Test DualWriteStats dataclass."""

    def test_default_stats(self) -> None:
        """Test default statistics values."""
        stats = DualWriteStats()
        assert stats.writes_to_primary == 0
        assert stats.writes_to_secondary == 0
        assert stats.write_failures == 0
        assert stats.last_write_time is None
        assert stats.reconcile_checks == 0
        assert stats.reconcile_mismatches == 0

    def test_stats_mutation(self) -> None:
        """Test statistics can be mutated."""
        stats = DualWriteStats()
        stats.writes_to_primary = 10
        stats.writes_to_secondary = 10
        stats.reconcile_checks = 5
        stats.reconcile_mismatches = 2
        stats.last_write_time = datetime.now(timezone.utc)

        assert stats.writes_to_primary == 10
        assert stats.writes_to_secondary == 10
        assert stats.reconcile_checks == 5
        assert stats.reconcile_mismatches == 2
        assert stats.last_write_time is not None


class TestReconcileResult:
    """Test ReconcileResult dataclass."""

    def test_reconcile_result_success(self) -> None:
        """Test reconcile result with no mismatches."""
        result = ReconcileResult(
            collection_name="test_collection",
            primary_backend=MetadataBackend.LANCEDB,
            secondary_backend=MetadataBackend.POSTGRESQL,
            records_checked=1,
            mismatches=[],
            is_consistent=True,
        )
        assert result.collection_name == "test_collection"
        assert result.is_consistent is True
        assert len(result.mismatches) == 0

    def test_reconcile_result_with_mismatches(self) -> None:
        """Test reconcile result with mismatches."""
        mismatches = [
            {"field": "documents", "primary_value": "5", "secondary_value": "3"}
        ]
        result = ReconcileResult(
            collection_name="test_collection",
            primary_backend=MetadataBackend.LANCEDB,
            secondary_backend=MetadataBackend.POSTGRESQL,
            records_checked=1,
            mismatches=mismatches,
            is_consistent=False,
        )
        assert result.collection_name == "test_collection"
        assert result.is_consistent is False
        assert len(result.mismatches) == 1
        assert result.mismatches[0]["field"] == "documents"


class TestDualWriteCoordinator:
    """Test DualWriteCoordinator functionality."""

    @pytest.fixture
    def mock_lancedb_store(self) -> MagicMock:
        """Create mock LanceDB metadata store."""
        store = MagicMock(spec=LanceDBMetadataStore)
        store.get_collection = AsyncMock()
        store.save_collection = AsyncMock()
        return store

    @pytest.fixture
    def mock_postgres_store(self) -> MagicMock:
        """Create mock PostgreSQL metadata store."""
        store = MagicMock()
        store.get_collection = AsyncMock()
        store.save_collection = AsyncMock()
        return store

    @pytest.fixture
    def dual_write_coordinator(
        self,
        mock_lancedb_store: MagicMock,
        mock_postgres_store: MagicMock,
    ) -> DualWriteCoordinator:
        """Create dual-write coordinator with mocked stores."""
        return DualWriteCoordinator(
            read_backend=MetadataBackend.LANCEDB,
            write_mode="both",
            metadata_store_lancedb=mock_lancedb_store,
            metadata_store_pg=mock_postgres_store,
        )

    def test_initialization(self, dual_write_coordinator: DualWriteCoordinator) -> None:
        """Test coordinator initialization."""
        assert dual_write_coordinator._write_mode == "both"
        assert dual_write_coordinator._read_backend == MetadataBackend.LANCEDB
        assert dual_write_coordinator.get_stats().writes_to_primary == 0

    def test_invalid_write_mode(self) -> None:
        """Test that invalid write mode raises ValueError."""
        with pytest.raises(ValueError, match="Invalid write_mode"):
            DualWriteCoordinator(write_mode="invalid")

    def test_invalid_read_backend(self) -> None:
        """Test that invalid read backend raises ValueError."""
        with pytest.raises(ValueError, match="Invalid read_backend"):
            DualWriteCoordinator(read_backend="invalid")  # type: ignore[arg-type]

    @pytest.mark.asyncio
    async def test_reconcile_collection_consistent(
        self,
        dual_write_coordinator: DualWriteCoordinator,
        mock_lancedb_store: MagicMock,
        mock_postgres_store: MagicMock,
    ) -> None:
        """Test reconcile when collections are consistent."""
        # Create consistent collection data
        collection = CollectionInfo(
            name="test_collection",
            owner_user_id=1,
            documents=5,
            chunks=100,
        )

        mock_lancedb_store.get_collection.return_value = collection
        mock_postgres_store.get_collection.return_value = collection

        result = await dual_write_coordinator.reconcile_collection("test_collection")

        assert result.is_consistent is True
        assert result.collection_name == "test_collection"
        assert len(result.mismatches) == 0
        assert dual_write_coordinator.get_stats().reconcile_checks == 1

    @pytest.mark.asyncio
    async def test_reconcile_collection_with_mismatch(
        self,
        dual_write_coordinator: DualWriteCoordinator,
        mock_lancedb_store: MagicMock,
        mock_postgres_store: MagicMock,
    ) -> None:
        """Test reconcile when collections have mismatches."""
        # Create inconsistent collection data
        lancedb_collection = CollectionInfo(
            name="test_collection",
            owner_user_id=1,
            documents=5,
        )
        postgres_collection = CollectionInfo(
            name="test_collection",
            owner_user_id=1,
            documents=3,  # Mismatch!
        )

        mock_lancedb_store.get_collection.return_value = lancedb_collection
        mock_postgres_store.get_collection.return_value = postgres_collection

        result = await dual_write_coordinator.reconcile_collection("test_collection")

        assert result.is_consistent is False
        assert len(result.mismatches) == 1
        assert result.mismatches[0]["field"] == "documents"
        assert dual_write_coordinator.get_stats().reconcile_mismatches == 1

    @pytest.mark.asyncio
    async def test_backfill_collection(
        self,
        dual_write_coordinator: DualWriteCoordinator,
        mock_lancedb_store: MagicMock,
        mock_postgres_store: MagicMock,
    ) -> None:
        """Test backfill from LanceDB to PostgreSQL."""
        collection = CollectionInfo(
            name="test_collection",
            owner_user_id=1,
        )

        mock_lancedb_store.get_collection.return_value = collection
        mock_postgres_store.save_collection = AsyncMock()

        result = await dual_write_coordinator.backfill_collection("test_collection")

        assert result["status"] == "success"
        assert result["collection"] == "test_collection"
        mock_postgres_store.save_collection.assert_called_once_with(collection)

    @pytest.mark.asyncio
    async def test_backfill_collection_failure(
        self,
        dual_write_coordinator: DualWriteCoordinator,
        mock_lancedb_store: MagicMock,
    ) -> None:
        """Test backfill handles failures gracefully."""
        mock_lancedb_store.get_collection.side_effect = Exception(
            "Collection not found"
        )

        result = await dual_write_coordinator.backfill_collection("nonexistent")

        assert result["status"] == "error"
        assert "Collection not found" in result["error"]

    def test_set_write_mode(self, dual_write_coordinator: DualWriteCoordinator) -> None:
        """Test changing write mode dynamically."""
        assert dual_write_coordinator._write_mode == "both"
        dual_write_coordinator.set_write_mode("postgresql")
        assert dual_write_coordinator._write_mode == "postgresql"

    def test_set_read_backend(
        self, dual_write_coordinator: DualWriteCoordinator
    ) -> None:
        """Test changing read backend dynamically."""
        assert dual_write_coordinator._read_backend == MetadataBackend.LANCEDB
        dual_write_coordinator.set_read_backend(MetadataBackend.POSTGRESQL)
        assert dual_write_coordinator._read_backend == MetadataBackend.POSTGRESQL


class TestDualWriteMetadataStore:
    """Test DualWriteMetadataStore functionality."""

    @pytest.fixture
    def mock_primary_store(self) -> MagicMock:
        """Create mock primary metadata store."""
        store = MagicMock()
        store.get_collection = AsyncMock()
        store.save_collection = AsyncMock()
        store.get_collection_config = AsyncMock()
        store.save_collection_config = AsyncMock()
        store.ensure_collection_metadata_table = AsyncMock()
        return store

    @pytest.fixture
    def mock_secondary_store(self) -> MagicMock:
        """Create mock secondary metadata store."""
        store = MagicMock()
        store.get_collection = AsyncMock()
        store.save_collection = AsyncMock()
        store.get_collection_config = AsyncMock()
        store.save_collection_config = AsyncMock()
        store.ensure_collection_metadata_table = AsyncMock()
        return store

    @pytest.fixture
    def stats(self) -> DualWriteStats:
        """Create fresh stats for each test."""
        return DualWriteStats()

    @pytest.fixture
    def dual_write_store(
        self,
        mock_primary_store: MagicMock,
        mock_secondary_store: MagicMock,
        stats: DualWriteStats,
    ) -> DualWriteMetadataStore:
        """Create dual-write metadata store with mocked backends."""
        return DualWriteMetadataStore(
            lancedb_store=mock_primary_store,
            pg_store=mock_secondary_store,
            stats=stats,
            read_backend=MetadataBackend.LANCEDB,
        )

    @pytest.mark.asyncio
    async def test_get_collection_reads_from_lancedb(
        self,
        dual_write_store: DualWriteMetadataStore,
        mock_primary_store: MagicMock,
        mock_secondary_store: MagicMock,
    ) -> None:
        """Test that get_collection reads from LanceDB backend."""
        collection = CollectionInfo(name="test", owner_user_id=1)
        mock_primary_store.get_collection.return_value = collection

        result = await dual_write_store.get_collection("test")

        assert result.name == "test"
        mock_primary_store.get_collection.assert_called_once_with("test")
        mock_secondary_store.get_collection.assert_not_called()

    @pytest.mark.asyncio
    async def test_save_collection_writes_to_both(
        self,
        dual_write_store: DualWriteMetadataStore,
        mock_primary_store: MagicMock,
        mock_secondary_store: MagicMock,
    ) -> None:
        """Test that save_collection writes to both backends."""
        collection = CollectionInfo(name="test", owner_user_id=1)

        await dual_write_store.save_collection(collection)

        mock_primary_store.save_collection.assert_called_once_with(collection)
        mock_secondary_store.save_collection.assert_called_once_with(collection)
        assert dual_write_store._stats.writes_to_primary == 1
        assert dual_write_store._stats.writes_to_secondary == 1

    @pytest.mark.asyncio
    async def test_save_collection_secondary_failure_does_not_affect_primary(
        self,
        dual_write_store: DualWriteMetadataStore,
        mock_primary_store: MagicMock,
        mock_secondary_store: MagicMock,
    ) -> None:
        """Test that secondary write failure doesn't prevent primary write."""
        collection = CollectionInfo(name="test", owner_user_id=1)
        mock_secondary_store.save_collection.side_effect = Exception("Secondary down")

        # Should not raise despite secondary failure
        await dual_write_store.save_collection(collection)

        mock_primary_store.save_collection.assert_called_once()
        assert dual_write_store._stats.write_failures == 1
        assert dual_write_store._stats.writes_to_primary == 1
        # Secondary write was attempted but failed
        assert dual_write_store._stats.writes_to_secondary == 0

    @pytest.mark.asyncio
    async def test_save_collection_config_writes_to_both(
        self,
        dual_write_store: DualWriteMetadataStore,
        mock_primary_store: MagicMock,
        mock_secondary_store: MagicMock,
    ) -> None:
        """Test that save_collection_config writes to both backends."""
        await dual_write_store.save_collection_config(
            collection="test",
            config_json='{"chunk_size": 1000}',
            user_id=1,
        )

        mock_primary_store.save_collection_config.assert_called_once()
        mock_secondary_store.save_collection_config.assert_called_once()
        assert dual_write_store._stats.writes_to_primary == 1
        assert dual_write_store._stats.writes_to_secondary == 1

    @pytest.mark.asyncio
    async def test_ensure_collection_metadata_table_both_backends(
        self,
        dual_write_store: DualWriteMetadataStore,
        mock_primary_store: MagicMock,
        mock_secondary_store: MagicMock,
    ) -> None:
        """Test that ensure_collection_metadata_table calls both backends."""
        await dual_write_store.ensure_collection_metadata_table()

        mock_primary_store.ensure_collection_metadata_table.assert_called_once()
        mock_secondary_store.ensure_collection_metadata_table.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_collection_config_reads_from_lancedb(
        self,
        dual_write_store: DualWriteMetadataStore,
        mock_primary_store: MagicMock,
        mock_secondary_store: MagicMock,
    ) -> None:
        """Test that get_collection_config reads from LanceDB backend."""
        mock_primary_store.get_collection_config.return_value = '{"chunk_size": 1000}'

        result = await dual_write_store.get_collection_config("test", 1)

        assert result == '{"chunk_size": 1000}'
        mock_primary_store.get_collection_config.assert_called_once_with("test", 1)
        mock_secondary_store.get_collection_config.assert_not_called()

    def test_get_raw_connection_returns_lancedb(
        self,
        dual_write_store: DualWriteMetadataStore,
        mock_primary_store: MagicMock,
    ) -> None:
        """Test that get_raw_connection returns LanceDB connection."""
        mock_conn = MagicMock()
        mock_primary_store.get_raw_connection.return_value = mock_conn

        result = dual_write_store.get_raw_connection()

        assert result is mock_conn
        mock_primary_store.get_raw_connection.assert_called_once()
