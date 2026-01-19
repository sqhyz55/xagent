"""Unit tests for ProgressTracker and StepTracker."""

from __future__ import annotations

from unittest.mock import MagicMock

from fenixaos.core.tools.core.RAG_tools.progress.manager import ProgressManager
from fenixaos.core.tools.core.RAG_tools.progress.tracker import (
    ProgressTracker,
    StepTracker,
)


class TestStepTracker:
    """Test StepTracker functionality."""

    def test_step_creation(self):
        """Test step tracker initialization."""
        task_tracker = MagicMock(spec=ProgressTracker)
        tracker = StepTracker(task_tracker, "parse_document")

        assert tracker.step.step_name == "parse_document"
        assert tracker.step.completed is False
        assert tracker.step.current_count == 0

    def test_step_update(self):
        """Test step status updates."""
        task_tracker = MagicMock(spec=ProgressTracker)
        tracker = StepTracker(task_tracker, "chunk_document")

        # Update with message
        tracker.update(
            message="Processing chunks...", current_count=50, total_count=100
        )

        assert tracker.step.message == "Processing chunks..."
        assert tracker.step.current_count == 50
        assert tracker.step.total_count == 100
        assert tracker.step.step_progress == 0.5
        task_tracker.update_overall_progress.assert_called()

    def test_step_complete(self):
        """Test step completion."""
        task_tracker = MagicMock(spec=ProgressTracker)
        tracker = StepTracker(task_tracker, "embed_vectors")

        # Mark as completed
        tracker.complete("Embedding completed successfully")

        assert tracker.step.completed is True
        assert tracker.step.step_progress == 1.0
        assert tracker.step.message == "Embedding completed successfully"
        assert tracker.step.end_time is not None

    def test_step_fail(self):
        """Test step failure."""
        task_tracker = MagicMock(spec=ProgressTracker)
        tracker = StepTracker(task_tracker, "write_vectors")

        # Mark as failed
        tracker.fail("Database connection failed")

        assert tracker.step.completed is False
        assert tracker.step.message == "Failed: Database connection failed"
        assert tracker.step.metadata["error"] == "Database connection failed"


class TestProgressTracker:
    """Test ProgressTracker functionality."""

    def test_tracker_creation(self):
        """Test progress tracker initialization."""
        manager = ProgressManager()
        manager._reset()
        tracker = ProgressTracker(manager, "test_task")

        assert tracker.manager == manager
        assert tracker.task_id == "test_task"
        assert tracker.step_trackers == {}

    def test_track_step_context_manager(self):
        """Test track_step context manager."""
        manager = ProgressManager()
        manager._reset()
        manager.create_task("ingestion", "test_task")
        tracker = ProgressTracker(manager, "test_task")

        with tracker.track_step("step1") as step_tracker:
            assert isinstance(step_tracker, StepTracker)
            assert "step1" in tracker.step_trackers
            assert manager.get_task_progress("test_task").current_step == "step1"

        assert tracker.step_trackers["step1"].step.completed is True

    def test_multiple_steps(self):
        """Test tracking multiple steps."""
        manager = ProgressManager()
        manager._reset()
        manager.create_task("ingestion", "test_task")
        tracker = ProgressTracker(manager, "test_task")

        with tracker.track_step("step1"):
            pass

        with tracker.track_step("step2"):
            pass

        assert len(tracker.step_trackers) == 2
        assert manager.get_task_progress("test_task").overall_progress == 1.0
