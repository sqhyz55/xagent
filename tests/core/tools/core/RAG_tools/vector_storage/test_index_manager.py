"""Tests for index_manager functionality.

This module tests the IndexManager class and related index management functions:
- Index creation and status checking
- Automatic index type selection (IVF_HNSW_SQ vs IVF_PQ)
- Configuration-driven indexing behavior
- Error handling and edge cases
"""

import os
import tempfile
import uuid
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from xagent.core.tools.core.RAG_tools.core.config import IndexPolicy
from xagent.core.tools.core.RAG_tools.core.schemas import IndexMetric
from xagent.core.tools.core.RAG_tools.vector_storage.index_manager import (
    IndexManager,
    get_index_manager,
)


class TestIndexManager:
    """Test IndexManager class functionality."""

    def test_init_with_default_policy(self):
        """Test IndexManager initialization with default policy."""
        manager = IndexManager()

        assert isinstance(manager.policy, IndexPolicy)
        assert manager.policy.enable_threshold_rows == 50_000
        assert manager.policy.ivfpq_threshold_rows == 10_000_000
        assert manager.policy.hnsw_params == {}
        assert manager.policy.ivfpq_params == {}

    def test_init_with_custom_policy(self):
        """Test IndexManager initialization with custom policy."""
        custom_policy = IndexPolicy(
            enable_threshold_rows=100_000,
            ivfpq_threshold_rows=5_000_000,
            hnsw_params={"ef_construction": 200},
            ivfpq_params={"nlist": 1024},
        )
        manager = IndexManager(custom_policy)

        assert manager.policy.enable_threshold_rows == 100_000
        assert manager.policy.ivfpq_threshold_rows == 5_000_000
        assert manager.policy.hnsw_params == {"ef_construction": 200}
        assert manager.policy.ivfpq_params == {"nlist": 1024}

    def test_readonly_mode(self):
        """Test readonly mode behavior."""
        manager = IndexManager()
        mock_table = Mock()

        status, advice = manager.check_and_create_index(
            mock_table, "test_table", readonly=True
        )

        assert status == "readonly"
        assert "Readonly mode" in advice
        # Should not call any table methods
        mock_table.to_pandas.assert_not_called()
        mock_table.list_indices.assert_not_called()

    @patch("xagent.core.tools.core.RAG_tools.vector_storage.index_manager.logger")
    def test_below_threshold_no_index(self, mock_logger):
        """Test behavior when row count is below threshold."""
        manager = IndexManager()
        mock_table = Mock()
        mock_table.count_rows.return_value = 0  # Empty table

        status, advice = manager.check_and_create_index(mock_table, "test_table")

        assert status == "below_threshold"
        assert "below threshold" in advice
        assert "50000" in advice

    def test_hnsw_index_creation(self):
        """Test HNSW index creation for medium-sized datasets."""
        manager = IndexManager()
        mock_table = Mock()

        # Mock table with 100,000 rows (between thresholds)
        mock_table.count_rows.return_value = 100_000
        mock_table.list_indices.return_value = []  # No existing indexes

        status, advice = manager.check_and_create_index(mock_table, "test_table")

        assert status == "index_building"
        assert "HNSW index created" in advice
        assert "using HNSW strategy for medium-scale data" in advice
        mock_table.create_index.assert_called_once_with(
            metric="l2", index_type="IVF_HNSW_SQ"
        )

    def test_ivfpq_index_creation(self):
        """Test IVFPQ index creation for large datasets."""
        manager = IndexManager()
        mock_table = Mock()

        # Mock table with 15M rows (above IVFPQ threshold)
        mock_table.count_rows.return_value = 15_000_000
        mock_table.list_indices.return_value = []  # No existing indexes

        status, advice = manager.check_and_create_index(mock_table, "test_table")

        assert status == "index_building"
        assert "IVFPQ index created" in advice
        assert "using IVFPQ strategy for large-scale data" in advice
        mock_table.create_index.assert_called_once_with(
            metric="l2", index_type="IVF_PQ"
        )

    def test_existing_index_skip(self):
        """Test skipping when index already exists."""
        manager = IndexManager()
        mock_table = Mock()

        # Mock table with enough rows and existing index
        mock_table.count_rows.return_value = 100_000
        mock_table.list_indices.return_value = [SimpleNamespace(name="vector")]

        status, advice = manager.check_and_create_index(mock_table, "test_table")

        assert status == "index_ready"
        assert "Index ready" in advice
        mock_table.create_index.assert_not_called()

    def test_index_creation_with_custom_params(self):
        """Test index creation with custom parameters."""
        custom_policy = IndexPolicy(
            hnsw_params={"ef_construction": 200, "M": 32},
            ivfpq_params={"nlist": 1024, "nprobe": 10},
        )
        manager = IndexManager(custom_policy)
        mock_table = Mock()

        # Test HNSW with custom params
        mock_table.count_rows.return_value = 100_000
        mock_table.list_indices.return_value = []

        status, advice = manager.check_and_create_index(mock_table, "test_table")

        # Check that create_index was called with correct parameters
        mock_table.create_index.assert_called_once()
        call_args = mock_table.create_index.call_args

        # Check keyword arguments
        kwargs = call_args[1]
        assert kwargs["metric"] == "l2"
        assert kwargs["index_type"] == "IVF_HNSW_SQ"
        assert kwargs["ef_construction"] == 200
        assert kwargs["M"] == 32

    def test_index_creation_error_handling(self):
        """Test error handling during index creation."""
        manager = IndexManager()
        mock_table = Mock()

        # Mock table operations
        mock_table.count_rows.return_value = 100_000
        mock_table.list_indices.return_value = []
        mock_table.create_index.side_effect = Exception("Index creation failed")

        status, advice = manager.check_and_create_index(mock_table, "test_table")

        assert status == "index_corrupted"
        assert "Vector index check failed" in advice
        assert "Index creation failed" in advice

    def test_get_index_status_ready(self):
        """Test getting index status when index exists."""
        manager = IndexManager()
        mock_table = Mock()
        mock_table.list_indices.return_value = [SimpleNamespace(name="vector")]

        status = manager.get_index_status(mock_table)
        assert status == "index_ready"

    def test_get_index_status_no_index(self):
        """Test getting index status when no index exists but above threshold."""
        manager = IndexManager()
        mock_table = Mock()
        mock_table.list_indices.return_value = []

        # Mock enough rows for indexing
        mock_table.count_rows.return_value = 100_000

        status = manager.get_index_status(mock_table)
        assert status == "no_index"

    def test_get_index_status_below_threshold(self):
        """Test getting index status when below threshold."""
        manager = IndexManager()
        mock_table = Mock()
        mock_table.list_indices.return_value = []

        # Mock few rows
        mock_table.count_rows.return_value = 1000

        status = manager.get_index_status(mock_table)
        assert status == "below_threshold"

    def test_get_index_status_error(self):
        """Test error handling in get_index_status."""
        manager = IndexManager()
        mock_table = Mock()
        mock_table.list_indices.side_effect = Exception("Database error")

        status = manager.get_index_status(mock_table)
        assert status == "index_corrupted"


class TestIndexManagerIntegration:
    """Integration tests for IndexManager with real LanceDB operations."""

    @pytest.fixture
    def temp_lancedb_dir(self):
        """Create a temporary directory for LanceDB."""
        with tempfile.TemporaryDirectory() as temp_dir:
            original_env = os.environ.get("LANCEDB_DIR")
            os.environ["LANCEDB_DIR"] = temp_dir
            yield temp_dir
            if original_env is not None:
                os.environ["LANCEDB_DIR"] = original_env
            else:
                os.environ.pop("LANCEDB_DIR", None)

    @pytest.fixture
    def test_collection(self):
        """Test collection name."""
        return f"test_collection_{uuid.uuid4().hex[:8]}"

    def test_end_to_end_index_creation(self, temp_lancedb_dir, test_collection):
        """Test end-to-end index creation workflow."""
        from xagent.core.tools.core.RAG_tools.LanceDB.schema_manager import (
            ensure_embeddings_table,
        )
        from xagent.providers.vector_store.lancedb import get_connection_from_env

        conn = get_connection_from_env()
        model_tag = "test_model"

        # Create embeddings table
        ensure_embeddings_table(conn, model_tag, vector_dim=3)
        table = conn.open_table(f"embeddings_{model_tag}")

        # Add some test data
        import pandas as pd

        test_records = [
            {
                "collection": test_collection,
                "doc_id": f"doc_{i}",
                "chunk_id": f"chunk_{i}",
                "parse_hash": "test_parse",
                "model": model_tag,
                "vector": [1.0, 2.0, 3.0],
                "vector_dimension": 3,
                "text": f"test text {i}",
                "chunk_hash": f"hash_{i}",
                "created_at": pd.Timestamp.now(tz="UTC"),
            }
            for i in range(60000)  # Add 60,000 records to trigger indexing
        ]
        table.add(test_records)

        # Test index manager
        manager = IndexManager()
        status, advice = manager.check_and_create_index(
            table, f"embeddings_{model_tag}"
        )

        assert status in ["index_building", "index_ready"]
        if status == "index_building":
            assert "HNSW index created" in advice

    def test_custom_policy_integration(self, temp_lancedb_dir, test_collection):
        """Test custom policy integration."""
        from xagent.core.tools.core.RAG_tools.LanceDB.schema_manager import (
            ensure_embeddings_table,
        )
        from xagent.providers.vector_store.lancedb import get_connection_from_env

        # Create custom policy with lower threshold
        custom_policy = IndexPolicy(
            enable_threshold_rows=50,  # Very low threshold for testing
            hnsw_params={"ef_construction": 100},
        )

        conn = get_connection_from_env()
        model_tag = "custom_policy_model"

        # Create table and add minimal data
        ensure_embeddings_table(conn, model_tag, vector_dim=3)
        table = conn.open_table(f"embeddings_{model_tag}")

        import pandas as pd

        # Add enough records to exceed the custom threshold of 50
        test_records = [
            {
                "collection": test_collection,
                "doc_id": f"doc_{i}",
                "chunk_id": f"chunk_{i}",
                "parse_hash": "test_parse",
                "model": model_tag,
                "vector": [1.0, 2.0, 3.0],
                "vector_dimension": 3,
                "text": f"test text {i}",
                "chunk_hash": f"hash_{i}",
                "created_at": pd.Timestamp.now(tz="UTC"),
            }
            for i in range(100)  # Add 100 records to exceed threshold of 50
        ]
        table.add(test_records)

        # Test with custom policy
        manager = IndexManager(custom_policy)
        status, advice = manager.check_and_create_index(
            table, f"embeddings_{model_tag}"
        )

        # Should trigger indexing due to low threshold
        assert status == "index_building"
        assert "HNSW index created" in advice


class TestGetIndexManager:
    """Test get_index_manager function."""

    def test_get_default_manager(self):
        """Test getting default index manager."""
        manager = get_index_manager()
        assert isinstance(manager, IndexManager)
        assert isinstance(manager.policy, IndexPolicy)

    def test_get_manager_with_custom_policy(self):
        """Test getting manager with custom policy."""
        custom_policy = IndexPolicy(enable_threshold_rows=100_000)
        manager = get_index_manager(custom_policy)

        assert manager.policy.enable_threshold_rows == 100_000

    def test_manager_singleton_behavior(self):
        """Test that get_index_manager returns the same instance when no policy is provided."""
        # Test default singleton behavior (no policy provided)
        manager1 = get_index_manager()
        manager2 = get_index_manager()

        # Should return the same instance when no policy is provided
        assert manager1 is manager2

    def test_manager_creates_new_instance_with_policy(self):
        """Test that get_index_manager creates new instances when policy is provided."""
        policy1 = IndexPolicy(enable_threshold_rows=50000)
        policy2 = IndexPolicy(enable_threshold_rows=50000)

        manager1 = get_index_manager(policy1)
        manager2 = get_index_manager(policy2)  # Same policy values

        # Should return different instances when policy is provided (current design)
        assert manager1 is not manager2
        # But they should have the same policy values
        assert (
            manager1.policy.enable_threshold_rows
            == manager2.policy.enable_threshold_rows
        )

    def test_manager_different_instances(self):
        """Test that different policies create different managers."""
        policy1 = IndexPolicy(enable_threshold_rows=50000)
        policy2 = IndexPolicy(enable_threshold_rows=100000)

        manager1 = get_index_manager(policy1)
        manager2 = get_index_manager(policy2)

        # Should be different instances for different policies
        assert manager1 is not manager2
        assert manager1.policy.enable_threshold_rows == 50000
        assert manager2.policy.enable_threshold_rows == 100000


class TestIndexMetricSupport:
    """Test IndexMetric parameter support in IndexManager."""

    def test_default_metric_l2(self):
        """Test that default metric is L2."""
        manager = IndexManager()
        assert manager.policy.metric == IndexMetric.L2
        assert manager.policy.metric.value == "l2"

    def test_custom_metric_cosine(self):
        """Test index creation with COSINE metric."""
        custom_policy = IndexPolicy(metric=IndexMetric.COSINE)
        manager = IndexManager(custom_policy)
        mock_table = Mock()

        # Mock table with enough rows for indexing
        mock_table.count_rows.return_value = 100_000
        mock_table.list_indices.return_value = []

        status, advice = manager.check_and_create_index(mock_table, "test_table")

        assert status == "index_building"
        assert "cosine" in advice
        # Verify that create_index was called with the correct metric
        call_args = mock_table.create_index.call_args
        assert call_args[1]["metric"] == "cosine"

    def test_custom_metric_dot(self):
        """Test index creation with DOT metric."""
        custom_policy = IndexPolicy(metric=IndexMetric.DOT)
        manager = IndexManager(custom_policy)
        mock_table = Mock()

        # Mock table with enough rows for indexing
        mock_table.count_rows.return_value = 100_000
        mock_table.list_indices.return_value = []

        status, advice = manager.check_and_create_index(mock_table, "test_table")

        assert status == "index_building"
        assert "dot" in advice
        # Verify that create_index was called with the correct metric
        call_args = mock_table.create_index.call_args
        assert call_args[1]["metric"] == "dot"


class TestFTSIndexSupport:
    """Test FTS index functionality in IndexManager."""

    def test_get_fts_index_status_no_index(self):
        """Test FTS index status when no FTS index exists."""
        manager = IndexManager()
        mock_table = Mock()
        mock_table.list_indices.return_value = []

        status = manager.get_fts_index_status(mock_table)
        assert status is False

    def test_get_fts_index_status_with_vector_index_only(self):
        """Test FTS index status when only vector index exists."""
        manager = IndexManager()
        mock_table = Mock()
        mock_table.list_indices.return_value = [
            SimpleNamespace(name="vector", index_type="IvfHnswSq", columns=["vector"])
        ]

        status = manager.get_fts_index_status(mock_table)
        assert status is False

    def test_get_fts_index_status_with_fts_index(self):
        """Test FTS index status when FTS index exists."""
        manager = IndexManager()
        mock_table = Mock()
        mock_table.list_indices.return_value = [
            SimpleNamespace(index_type="FTS", columns=["text"])
        ]

        status = manager.get_fts_index_status(mock_table)
        assert status is True

    def test_get_fts_index_status_error_handling(self):
        """Test FTS index status error handling."""
        manager = IndexManager()
        mock_table = Mock()
        mock_table.list_indices.side_effect = Exception("Database error")
        mock_table.name = "test_table"

        status = manager.get_fts_index_status(mock_table)
        assert status is False

    def test_create_fts_index_success(self):
        """Test successful FTS index creation."""
        manager = IndexManager()
        mock_table = Mock()
        mock_table.list_indices.return_value = []  # No existing FTS index

        success, message = manager.create_fts_index(mock_table, "test_table")

        assert success is True
        assert "FTS index created" in message
        assert "with_position" in message
        # Verify create_index was called with correct parameters
        mock_table.create_fts_index.assert_called_once_with(
            "text", replace=True, with_position=True
        )

    def test_create_fts_index_already_exists(self):
        """Test FTS index creation when index already exists."""
        manager = IndexManager()
        mock_table = Mock()
        mock_table.list_indices.return_value = [
            SimpleNamespace(index_type="FTS", columns=["text"])
        ]

        success, message = manager.create_fts_index(mock_table, "test_table")

        assert success is True
        assert "already exists" in message
        mock_table.create_fts_index.assert_not_called()

    def test_create_fts_index_with_custom_params(self):
        """Test FTS index creation with custom parameters."""
        manager = IndexManager()
        mock_table = Mock()
        mock_table.list_indices.return_value = []

        custom_params = {
            "language": "english",
            "stem": True,
            "ascii_folding": True,
        }
        success, message = manager.create_fts_index(
            mock_table, "test_table", fts_params=custom_params
        )

        assert success is True
        assert "FTS index created" in message
        # Verify create_index was called with merged parameters
        expected_params = {"with_position": True, **custom_params}
        mock_table.create_fts_index.assert_called_once_with(
            "text", replace=True, **expected_params
        )

    def test_create_fts_index_error_handling(self):
        """Test FTS index creation error handling."""
        manager = IndexManager()
        mock_table = Mock()
        mock_table.list_indices.return_value = []
        mock_table.create_fts_index.side_effect = Exception("FTS creation failed")

        success, message = manager.create_fts_index(mock_table, "test_table")

        assert success is False
        assert "Failed to create FTS index" in message
        assert "FTS creation failed" in message

    def test_check_and_create_index_with_fts_enabled(self):
        """Test that check_and_create_index attempts FTS creation when enabled."""
        custom_policy = IndexPolicy(
            fts_enabled=True, fts_params={"language": "english"}
        )
        manager = IndexManager(custom_policy)
        mock_table = Mock()

        # Mock table with enough rows for vector indexing
        mock_table.count_rows.return_value = 100_000
        mock_table.list_indices.return_value = []  # No existing indexes

        status, advice = manager.check_and_create_index(mock_table, "test_table")

        # Should create both vector and FTS indexes
        assert status == "index_building"
        # Verify vector index creation
        assert mock_table.create_index.call_count == 1
        vector_call = mock_table.create_index.call_args_list[0]
        assert vector_call[1]["index_type"] == "IVF_HNSW_SQ"
        mock_table.create_fts_index.assert_called_once()
        fts_call = mock_table.create_fts_index.call_args
        assert fts_call[0][0] == "text"
        assert fts_call[1]["replace"] is True


class TestReindexingIntegration:
    """Test reindexing functionality integration with IndexManager."""

    def test_reindex_trigger_conditions(self):
        """Test various conditions that should trigger reindexing."""
        from unittest.mock import MagicMock

        # Test with different policy configurations
        policies = [
            # Immediate reindex
            IndexPolicy(enable_immediate_reindex=True),
            # Batch size threshold
            IndexPolicy(reindex_batch_size=100),
            # Smart reindex with ratio threshold
            IndexPolicy(
                enable_smart_reindex=True, reindex_unindexed_ratio_threshold=0.05
            ),
        ]

        for policy in policies:
            manager = IndexManager(policy)
            mock_table = MagicMock()

            # Mock table with existing index
            mock_table.count_rows.return_value = 100_000
            mock_table.list_indices.return_value = [SimpleNamespace(name="vector")]

            # Test that existing index is detected
            status, advice = manager.check_and_create_index(mock_table, "test_table")
            assert status == "index_ready"
            assert "Index ready" in advice

    @pytest.mark.skip(
        "Phase 1A: Reindex functionality moved to VectorIndexStore.should_reindex() "
        "and VectorIndexStore.trigger_reindex(). Tested in test_lancedb_stores.py:"
        "test_should_reindex_immediate_reindex_enabled, test_trigger_reindex_success."
    )
    def test_reindex_with_optimize_call(self):
        """Test that reindexing calls table.optimize()."""
        from unittest.mock import MagicMock

        # Import the reindex functions from vector_manager
        from xagent.core.tools.core.RAG_tools.vector_storage.vector_manager import (
            _should_reindex,
            _trigger_reindex,
        )

        mock_table = MagicMock()
        policy = IndexPolicy(enable_immediate_reindex=True)

        # Test _should_reindex returns True for immediate mode
        should_reindex = _should_reindex(mock_table, "test_table", 1, policy)
        assert should_reindex is True

        # Test _trigger_reindex calls optimize
        mock_table.optimize.return_value = None
        reindex_success = _trigger_reindex(mock_table, "test_table")

        assert reindex_success is True
        mock_table.optimize.assert_called_once()

    @pytest.mark.skip(
        "Phase 1A: Reindex error handling moved to VectorIndexStore.trigger_reindex(). "
        "Tested in test_lancedb_stores.py::test_trigger_reindex_failure."
    )
    def test_reindex_error_handling(self):
        """Test reindex error handling."""
        from unittest.mock import MagicMock

        from xagent.core.tools.core.RAG_tools.vector_storage.vector_manager import (
            _trigger_reindex,
        )

        mock_table = MagicMock()
        mock_table.optimize.side_effect = Exception("Optimize failed")

        reindex_success = _trigger_reindex(mock_table, "test_table")

        assert reindex_success is False
        mock_table.optimize.assert_called_once()

    @pytest.mark.skip(
        "Phase 1A: Smart reindex moved to VectorIndexStore.should_reindex(). "
        "Tested in test_lancedb_stores.py::test_should_reindex_smart_reindex."
    )
    def test_smart_reindex_with_index_stats(self):
        """Test smart reindex based on index statistics."""
        from unittest.mock import MagicMock

        from xagent.core.tools.core.RAG_tools.vector_storage.vector_manager import (
            _should_reindex,
        )

        mock_table = MagicMock()
        policy = IndexPolicy(
            enable_smart_reindex=True, reindex_unindexed_ratio_threshold=0.05
        )

        # Mock index stats showing high unindexed ratio
        mock_stats = MagicMock()
        mock_stats.num_indexed_rows = 1000
        mock_stats.num_unindexed_rows = 60  # 6% > 5% threshold
        mock_table.index_stats.return_value = mock_stats

        should_reindex = _should_reindex(mock_table, "test_table", 10, policy)
        assert should_reindex is True

        # Test below threshold
        mock_stats.num_unindexed_rows = 30  # 3% < 5% threshold
        should_reindex = _should_reindex(mock_table, "test_table", 10, policy)
        assert should_reindex is False

    @pytest.mark.skip(
        "Phase 1A: Batch size reindex threshold moved to VectorIndexStore.should_reindex(). "
        "Tested in test_lancedb_stores.py::test_should_reindex_batch_threshold."
    )
    def test_batch_size_reindex_threshold(self):
        """Test batch size threshold for reindexing."""
        from unittest.mock import MagicMock

        from xagent.core.tools.core.RAG_tools.vector_storage.vector_manager import (
            _should_reindex,
        )

        mock_table = MagicMock()
        policy = IndexPolicy(reindex_batch_size=100)

        # Test above batch threshold
        should_reindex = _should_reindex(mock_table, "test_table", 150, policy)
        assert should_reindex is True

        # Test below batch threshold
        should_reindex = _should_reindex(mock_table, "test_table", 50, policy)
        assert should_reindex is False

    @pytest.mark.skip(
        "Phase 1A: Index stats error handling moved to VectorIndexStore.should_reindex(). "
        "Tested in test_lancedb_stores.py::test_should_reindex_smart_reindex (logs error)."
    )
    def test_reindex_with_index_stats_error(self):
        """Test reindex behavior when index stats fail."""
        from unittest.mock import MagicMock

        from xagent.core.tools.core.RAG_tools.vector_storage.vector_manager import (
            _should_reindex,
        )

        mock_table = MagicMock()
        policy = IndexPolicy(enable_smart_reindex=True)

        # Mock index_stats to raise exception
        mock_table.index_stats.side_effect = Exception("Stats failed")

        # Should not trigger reindex when stats fail
        should_reindex = _should_reindex(mock_table, "test_table", 10, policy)
        assert should_reindex is False
