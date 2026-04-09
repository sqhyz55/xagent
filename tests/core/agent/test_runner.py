"""
Comprehensive unit tests for the AgentRunner class.

This module tests the AgentRunner functionality including:
- Pattern execution with success and failure scenarios
- Precondition resolution
- Exception handling and error reporting
- User input handling during execution
- Multiple pattern execution strategies
"""

from typing import Any, Dict, List, Optional
from unittest.mock import patch

import pytest

from xagent.core.agent.agent import Agent
from xagent.core.agent.context import AgentContext
from xagent.core.agent.exceptions import AgentException
from xagent.core.agent.pattern.base import AgentPattern
from xagent.core.agent.precondition import PreconditionResolver
from xagent.core.agent.runner import AgentRunner
from xagent.core.memory import MemoryStore
from xagent.core.memory.in_memory import InMemoryMemoryStore
from xagent.core.tools.adapters.vibe import Tool


class MockPattern(AgentPattern):
    """Mock pattern for testing purposes."""

    def __init__(self, name: str, result: Dict[str, Any], should_fail: bool = False):
        self.name = name
        self.result = result
        self.should_fail = should_fail
        self.call_count = 0
        self._pattern_name = name  # Store the original name for error reporting

    async def run(
        self,
        task: str,
        memory: MemoryStore,
        tools: List[Tool],
        context: Optional[AgentContext] = None,
    ) -> Dict[str, Any]:
        """Mock run method."""
        self.call_count += 1

        if self.should_fail:
            if self.result.get("exception_type") == "AgentException":
                raise AgentException(
                    self.result.get("error", "Mock AgentException"),
                    context={"test": "context"},
                )
            else:
                raise Exception(self.result.get("error", "Mock exception"))

        # If this is a user input pattern and we've already been called once,
        # return a successful result instead of requesting more input
        if (
            self.result.get("need_user_input")
            and self.call_count > 1
            and context
            and self.result.get("field") in context.state
        ):
            return {"success": True, "message": "Task completed with user input"}

        return self.result


class MockTool(Tool):
    """Mock tool for testing."""

    def __init__(self, name: str):
        super().__init__(name=name)

    async def execute(self, **kwargs) -> Any:
        return {"mock_tool_result": True}


class MockMemoryStore(InMemoryMemoryStore):
    """Mock memory store for testing."""

    def __init__(self):
        super().__init__()

    async def store(self, key: str, value: Any) -> None:
        # Create a MemoryNote for simple key-value storage
        from xagent.core.memory.core import MemoryNote

        note = MemoryNote(content=str(value), metadata={"key": key})
        self.add(note)

    async def retrieve(self, key: str) -> Optional[Any]:
        # Search for notes with this key
        results = self.search(key)
        for note in results:
            if note.metadata.get("key") == key:
                return note.content
        return None


class TestAgentRunner:
    """Test cases for AgentRunner class."""

    @pytest.fixture
    def mock_memory(self):
        """Create a mock memory store."""
        return MockMemoryStore()

    @pytest.fixture
    def mock_tools(self):
        """Create mock tools."""
        return [MockTool("test_tool")]

    @pytest.fixture
    def successful_pattern(self):
        """Create a successful mock pattern."""
        return MockPattern(
            name="successful_pattern",
            result={"success": True, "message": "Task completed successfully"},
        )

    @pytest.fixture
    def failing_pattern(self):
        """Create a failing mock pattern."""
        return MockPattern(
            name="failing_pattern",
            result={"success": False, "error": "Pattern failed"},
            should_fail=True,
        )

    @pytest.fixture
    def user_input_pattern(self):
        """Create a pattern that requests user input."""
        return MockPattern(
            name="user_input_pattern",
            result={
                "need_user_input": True,
                "field": "test_field",
                "question": "Please provide test_field value:",
            },
        )

    @pytest.fixture
    def precondition_resolver(self):
        """Create a precondition resolver."""
        return PreconditionResolver(
            required_fields=["required_field"],
            questions={"required_field": "Please enter the required field:"},
        )

    @pytest.fixture
    def agent(self, successful_pattern, mock_memory, mock_tools):
        """Create a test agent."""
        return Agent(
            name="test_agent",
            patterns=[successful_pattern],
            memory=mock_memory,
            tools=mock_tools,
        )

    @pytest.fixture
    def agent_with_precondition(
        self, successful_pattern, mock_memory, mock_tools, precondition_resolver
    ):
        """Create an agent with precondition resolver."""
        return Agent(
            name="test_agent_with_precondition",
            patterns=[successful_pattern],
            memory=mock_memory,
            tools=mock_tools,
        )

    @pytest.fixture
    def runner(self, agent):
        """Create an AgentRunner instance."""
        return AgentRunner(agent=agent)

    @pytest.fixture
    def runner_with_precondition(self, agent_with_precondition, precondition_resolver):
        """Create an AgentRunner with precondition resolver."""
        return AgentRunner(
            agent=agent_with_precondition, precondition=precondition_resolver
        )

    def test_agent_runner_initialization(self, runner, agent):
        """Test AgentRunner initialization."""
        assert runner.agent == agent
        assert runner.context is not None
        assert isinstance(runner.context, AgentContext)
        assert runner.precondition is None

    def test_agent_runner_with_precondition(
        self, runner_with_precondition, precondition_resolver
    ):
        """Test AgentRunner initialization with precondition."""
        assert runner_with_precondition.precondition == precondition_resolver

    @pytest.mark.asyncio
    async def test_successful_pattern_execution(self, runner):
        """Test successful execution of a single pattern."""
        result = await runner.run("test task")

        assert result["success"] is True
        assert result["message"] == "Task completed successfully"
        assert runner.agent.patterns[0].call_count == 1

    @pytest.mark.asyncio
    async def test_precondition_resolution_success(self, runner_with_precondition):
        """Test successful precondition resolution."""
        # Set the required field in context
        runner_with_precondition.context.state["required_field"] = "test_value"

        result = await runner_with_precondition.run("test task")

        assert result["success"] is True
        assert "required_field" in runner_with_precondition.context.state
        assert runner_with_precondition.context.state["required_field"] == "test_value"

    @pytest.mark.asyncio
    async def test_precondition_resolution_with_user_input(
        self, runner_with_precondition, capsys
    ):
        """Test precondition resolution requiring user input."""
        # Mock user input
        with patch("builtins.input", return_value="user_provided_value"):
            result = await runner_with_precondition.run("test task")

        assert result["success"] is True
        assert (
            runner_with_precondition.context.state["required_field"]
            == "user_provided_value"
        )

        # Check that the user was prompted
        captured = capsys.readouterr()
        assert "[Agent asks] Please enter the required field:" in captured.out

    @pytest.mark.asyncio
    async def test_multiple_patterns_first_succeeds(self, mock_memory, mock_tools):
        """Test multiple patterns where the first one succeeds."""
        patterns = [
            MockPattern(
                "pattern1", {"success": True, "message": "First pattern succeeded"}
            ),
            MockPattern(
                "pattern2", {"success": True, "message": "Second pattern would succeed"}
            ),
        ]

        agent = Agent(
            name="test_agent", patterns=patterns, memory=mock_memory, tools=mock_tools
        )

        runner = AgentRunner(agent=agent)
        result = await runner.run("test task")

        assert result["success"] is True
        assert result["message"] == "First pattern succeeded"
        assert patterns[0].call_count == 1
        assert patterns[1].call_count == 0  # Second pattern should not be called

    @pytest.mark.asyncio
    async def test_multiple_patterns_first_fails_second_succeeds(
        self, mock_memory, mock_tools
    ):
        """Test multiple patterns where the first fails and the second succeeds."""
        patterns = [
            MockPattern(
                "failing_pattern",
                {"success": False, "error": "First failed"},
                should_fail=True,
            ),
            MockPattern(
                "successful_pattern", {"success": True, "message": "Second succeeded"}
            ),
        ]

        agent = Agent(
            name="test_agent", patterns=patterns, memory=mock_memory, tools=mock_tools
        )

        runner = AgentRunner(agent=agent)
        result = await runner.run("test task")

        assert result["success"] is True
        assert result["message"] == "Second succeeded"
        assert patterns[0].call_count == 1
        assert patterns[1].call_count == 1

    @pytest.mark.asyncio
    async def test_all_patterns_fail(self, mock_memory, mock_tools):
        """Test scenario where all patterns fail."""
        patterns = [
            MockPattern(
                "failing_pattern1",
                {"success": False, "error": "First failed"},
                should_fail=True,
            ),
            MockPattern(
                "failing_pattern2",
                {"success": False, "error": "Second failed"},
                should_fail=True,
            ),
        ]

        agent = Agent(
            name="test_agent", patterns=patterns, memory=mock_memory, tools=mock_tools
        )

        runner = AgentRunner(agent=agent)
        result = await runner.run("test task")

        assert result["success"] is False
        assert "All 2 patterns failed" in result["error"]
        assert "pattern_errors" in result
        assert len(result["pattern_errors"]) == 2
        assert result["patterns_attempted"] == 2

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_pattern_with_user_input_request(self, mock_memory, mock_tools):
        """Test pattern that requests user input during execution."""
        patterns = [
            MockPattern(
                "user_input_pattern",
                result={
                    "need_user_input": True,
                    "field": "dynamic_field",
                    "question": "Please provide dynamic_field value:",
                },
            )
        ]

        agent = Agent(
            name="test_agent", patterns=patterns, memory=mock_memory, tools=mock_tools
        )

        runner = AgentRunner(agent=agent)

        # Mock user input
        with patch("builtins.input", return_value="dynamic_value"):
            await runner.run("test task")

        # The pattern should be called twice - once to request input, once with input
        assert patterns[0].call_count == 2
        assert runner.context.state["dynamic_field"] == "dynamic_value"

    @pytest.mark.asyncio
    async def test_agent_exception_handling(self, mock_memory, mock_tools):
        """Test handling of AgentException."""
        patterns = [
            MockPattern(
                "agent_exception_pattern",
                result={
                    "exception_type": "AgentException",
                    "error": "Agent-specific error occurred",
                },
                should_fail=True,
            )
        ]

        agent = Agent(
            name="test_agent", patterns=patterns, memory=mock_memory, tools=mock_tools
        )

        runner = AgentRunner(agent=agent)
        result = await runner.run("test task")

        assert result["success"] is False
        assert "pattern_errors" in result
        assert len(result["pattern_errors"]) == 1
        error_info = result["pattern_errors"][0]
        assert error_info["pattern"] == "MockPattern"
        assert error_info["error"] == "Agent-specific error occurred"
        assert error_info["exception_type"] == "AgentException"
        assert "exception_context" in error_info
        assert "full_traceback" in error_info

    @pytest.mark.asyncio
    async def test_general_exception_handling(self, mock_memory, mock_tools):
        """Test handling of general exceptions."""
        patterns = [
            MockPattern(
                "general_exception_pattern",
                result={"error": "General error occurred"},
                should_fail=True,
            )
        ]

        agent = Agent(
            name="test_agent", patterns=patterns, memory=mock_memory, tools=mock_tools
        )

        runner = AgentRunner(agent=agent)
        result = await runner.run("test task")

        assert result["success"] is False
        assert "pattern_errors" in result
        assert len(result["pattern_errors"]) == 1
        error_info = result["pattern_errors"][0]
        assert error_info["pattern"] == "MockPattern"
        assert "General error occurred" in error_info["error"]
        assert "exception_type" in error_info
        assert "exception_category" in error_info
        assert error_info["exception_category"] == "unexpected_error"

    @pytest.mark.asyncio
    async def test_context_task_id_generation(self, runner):
        """Test that task_id is properly generated in context."""
        await runner.run("test task")

        assert runner.context.task_id is not None
        assert isinstance(runner.context.task_id, str)
        assert len(runner.context.task_id) > 0

    @pytest.mark.asyncio
    async def test_context_start_time_set(self, runner):
        """Test that start_time is properly set in context."""
        from datetime import datetime, timezone

        await runner.run("test task")

        assert runner.context.start_time is not None
        # The start_time should be set during the run, just check it's not in the future
        assert runner.context.start_time <= datetime.now(timezone.utc)

    @pytest.mark.asyncio
    async def test_context_history_tracking(self, runner):
        """Test that context history is tracked."""
        await runner.run("test task")

        assert isinstance(runner.context.history, list)
        # History tracking depends on implementation details

    @pytest.mark.asyncio
    async def test_pattern_execution_with_context(self, mock_memory, mock_tools):
        """Test that patterns receive the correct context."""
        context_received = []

        class ContextTrackingPattern(AgentPattern):
            async def run(
                self,
                task: str,
                memory: MemoryStore,
                tools: List[Tool],
                context: Optional[AgentContext] = None,
            ) -> Dict[str, Any]:
                context_received.append(context)
                return {"success": True}

        patterns = [ContextTrackingPattern()]

        agent = Agent(
            name="test_agent", patterns=patterns, memory=mock_memory, tools=mock_tools
        )

        runner = AgentRunner(agent=agent)
        await runner.run("test task")

        assert len(context_received) == 1
        assert context_received[0] == runner.context

    def test_get_full_traceback_with_chained_exceptions(self, runner):
        """Test _get_full_traceback method with chained exceptions."""
        # Create a chained exception
        try:
            try:
                raise ValueError("Original error")
            except ValueError as e:
                raise RuntimeError("Chained error") from e
        except Exception as e:
            traceback = runner._get_full_traceback(e)

            assert "ValueError: Original error" in traceback
            assert "RuntimeError: Chained error" in traceback
            assert "Caused by:" in traceback

    def test_get_full_traceback_with_single_exception(self, runner):
        """Test _get_full_traceback method with single exception."""
        try:
            raise ValueError("Single error")
        except Exception as e:
            traceback = runner._get_full_traceback(e)

            assert "ValueError: Single error" in traceback
            assert "Caused by:" not in traceback

    def test_get_full_traceback_with_non_exception(self, runner):
        """Test _get_full_traceback method with non-exception."""
        # Test with a real exception to verify the method works
        try:
            raise ValueError("Test exception")
        except Exception as e:
            traceback = runner._get_full_traceback(e)
            assert isinstance(traceback, str)
            assert len(traceback) > 0
            assert "ValueError: Test exception" in traceback

    @pytest.mark.asyncio
    async def test_empty_patterns_list(self, mock_memory, mock_tools):
        """Test AgentRunner with no patterns."""
        agent = Agent(
            name="test_agent", patterns=[], memory=mock_memory, tools=mock_tools
        )

        runner = AgentRunner(agent=agent)
        result = await runner.run("test task")

        assert result["success"] is False
        assert "All 0 patterns failed" in result["error"]
        assert result["patterns_attempted"] == 0


if __name__ == "__main__":
    pytest.main([__file__])
