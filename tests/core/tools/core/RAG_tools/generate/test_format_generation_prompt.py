import logging

import pytest

from xagent.core.tools.core.RAG_tools.core.exceptions import ConfigurationError
from xagent.core.tools.core.RAG_tools.generate.format_generation_prompt import (
    format_generation_prompt,
)

logger = logging.getLogger(__name__)


@pytest.fixture
def sample_prompt_template() -> str:
    """Provides a sample prompt template for testing."""
    return "Please summarize the following context:\n{context}"


@pytest.fixture
def sample_formatted_contexts() -> str:
    """Provides sample formatted contexts for testing."""
    return "This is the first chunk.\n---\nThis is the second chunk."


@pytest.fixture
def expected_full_prompt(
    sample_prompt_template: str, sample_formatted_contexts: str
) -> str:
    """Provides the expected full prompt string."""
    return (
        f"{sample_prompt_template}\n\nContext:\n{sample_formatted_contexts}\n\nAnswer:"
    )


class TestFormatGenerationPrompt:
    """Tests for the format_generation_prompt core function."""

    def test_format_generation_prompt_success(
        self,
        sample_prompt_template: str,
        sample_formatted_contexts: str,
        expected_full_prompt: str,
    ) -> None:
        """Test successful prompt formatting."""
        result = format_generation_prompt(
            prompt_template=sample_prompt_template,
            formatted_contexts=sample_formatted_contexts,
        )

        assert isinstance(result, str)
        assert result == expected_full_prompt

    def test_format_generation_prompt_empty_template_raises_error(
        self,
        sample_formatted_contexts: str,
    ) -> None:
        """Test that empty prompt template raises ConfigurationError."""
        with pytest.raises(
            ConfigurationError, match="Prompt template cannot be empty."
        ):
            format_generation_prompt(
                prompt_template="",
                formatted_contexts=sample_formatted_contexts,
            )

    def test_format_generation_prompt_empty_contexts_produces_warning_and_formats(
        self,
        sample_prompt_template: str,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test that empty formatted contexts produce a warning but still format."""
        with caplog.at_level(logging.WARNING):
            result = format_generation_prompt(
                prompt_template=sample_prompt_template,
                formatted_contexts="",
            )
            assert "Formatted contexts are empty" in caplog.text

        expected_prompt_for_empty_context = (
            f"{sample_prompt_template}\n\nContext:\n\n\nAnswer:"
        )
        assert result == expected_prompt_for_empty_context
