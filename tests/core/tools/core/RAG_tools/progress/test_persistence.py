"""Unit tests for ProgressPersistence."""

from __future__ import annotations

import tempfile
from pathlib import Path

from xagent.core.tools.core.RAG_tools.core.schemas import (
    DocumentProcessingStatus,
    TaskProgress,
)
from xagent.core.tools.core.RAG_tools.progress.persistence import ProgressPersistence


class TestProgressPersistence:
    """Test ProgressPersistence functionality."""

    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.persistence = ProgressPersistence(storage_dir=self.temp_dir)

    def teardown_method(self):
        """Clean up test environment."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_initialization(self):
        """Test persistence initialization."""
        assert self.persistence.storage_dir == Path(self.temp_dir)
        assert self.persistence.storage_dir.exists()

    def test_save_and_load_task(self):
        """Test saving and loading a single task."""
        # Create a test task
        task = TaskProgress(
            task_id="test_task_123",
            task_type="ingestion",
            status=DocumentProcessingStatus.RUNNING,
            current_step="parse_document",
            overall_progress=0.5,
            start_time=1234567890.0,
            end_time=None,
            metadata={"collection": "docs", "pages": 10},
        )

        # Save task
        self.persistence.save_task_progress(task)

        # Check file was created
        task_file = self.persistence.storage_dir / "test_task_123.json"
        assert task_file.exists()

        # Load task
        loaded_task = self.persistence.load_task_progress("test_task_123")

        assert loaded_task is not None
        assert loaded_task.task_id == task.task_id
        assert loaded_task.task_type == task.task_type
        assert loaded_task.status == task.status
        assert loaded_task.current_step == task.current_step
        assert loaded_task.overall_progress == task.overall_progress
        assert loaded_task.start_time == task.start_time
        assert loaded_task.end_time == task.end_time
        assert loaded_task.metadata == task.metadata

    def test_save_multiple_tasks(self):
        """Test saving multiple tasks."""
        tasks = []

        for i in range(3):
            task = TaskProgress(
                task_id=f"multi_task_{i}",
                task_type="ingestion" if i % 2 == 0 else "search",
                status=DocumentProcessingStatus.RUNNING,
                current_step=f"step_{i}",
                overall_progress=i * 0.2,
                start_time=1234567890.0 + i,
                metadata={"index": i},
            )
            tasks.append(task)
            self.persistence.save_task_progress(task)

        # Check all files were created
        for i in range(3):
            task_file = self.persistence.storage_dir / f"multi_task_{i}.json"
            assert task_file.exists()

        # Load and verify all tasks
        for i, original_task in enumerate(tasks):
            loaded_task = self.persistence.load_task_progress(f"multi_task_{i}")
            assert loaded_task is not None
            assert loaded_task.task_id == original_task.task_id
            assert loaded_task.task_type == original_task.task_type

    def test_load_nonexistent_task(self):
        """Test loading a task that doesn't exist."""
        loaded_task = self.persistence.load_task_progress("nonexistent_task")
        assert loaded_task is None

    def test_delete_task(self):
        """Test deleting a task."""
        # Create and save a task
        task = TaskProgress(
            task_id="delete_test",
            task_type="ingestion",
            status=DocumentProcessingStatus.RUNNING,
        )
        self.persistence.save_task_progress(task)

        # Verify it exists
        task_file = self.persistence.storage_dir / "delete_test.json"
        assert task_file.exists()

        # Delete the task
        self.persistence.delete_task_progress("delete_test")

        # Verify it's gone
        assert not task_file.exists()
        assert self.persistence.load_task_progress("delete_test") is None

    def test_list_all_tasks(self):
        """Test listing all tasks."""
        # Create multiple tasks
        task_ids = ["list_test_1", "list_test_2", "list_test_3"]

        for task_id in task_ids:
            task = TaskProgress(
                task_id=task_id,
                task_type="ingestion",
                status=DocumentProcessingStatus.RUNNING,
            )
            self.persistence.save_task_progress(task)

        # List all tasks
        all_tasks = self.persistence.list_all_tasks()

        assert len(all_tasks) == 3
        returned_ids = [task.task_id for task in all_tasks]
        for task_id in task_ids:
            assert task_id in returned_ids

    def test_list_active_tasks(self):
        """Test listing only active tasks."""
        # Create tasks with different statuses
        tasks_data = [
            ("active_task", DocumentProcessingStatus.RUNNING),
            ("completed_task", DocumentProcessingStatus.SUCCESS),
            ("failed_task", DocumentProcessingStatus.FAILED),
            ("cancelled_task", DocumentProcessingStatus.CANCELLED),
            ("pending_task", DocumentProcessingStatus.PENDING),
        ]

        for task_id, status in tasks_data:
            task = TaskProgress(task_id=task_id, task_type="ingestion", status=status)
            self.persistence.save_task_progress(task)

        # List active tasks (should exclude SUCCESS, FAILED, CANCELLED)
        active_tasks = self.persistence.list_active_tasks()

        active_ids = [task.task_id for task in active_tasks]
        assert "active_task" in active_ids
        assert "pending_task" in active_ids
        assert "completed_task" not in active_ids
        assert "failed_task" not in active_ids
        assert "cancelled_task" not in active_ids

    def test_cleanup_old_tasks(self):
        """Test cleaning up old completed tasks."""
        import time

        # Create tasks with different completion times
        base_time = time.time()

        # Create a completed task (old)
        old_task = TaskProgress(
            task_id="old_completed",
            task_type="ingestion",
            status=DocumentProcessingStatus.SUCCESS,
            end_time=base_time - (25 * 3600),  # 25 hours ago
        )
        self.persistence.save_task_progress(old_task)

        # Create a recently completed task
        recent_task = TaskProgress(
            task_id="recent_completed",
            task_type="ingestion",
            status=DocumentProcessingStatus.SUCCESS,
            end_time=base_time - (1 * 3600),  # 1 hour ago
        )
        self.persistence.save_task_progress(recent_task)

        # Clean up tasks older than 24 hours
        deleted_count = self.persistence.cleanup_old_tasks(max_age_hours=24)

        # Should have deleted 1 old task
        assert deleted_count == 1

        # Check that old task is gone
        assert self.persistence.load_task_progress("old_completed") is None
        assert self.persistence.load_task_progress("recent_completed") is not None
