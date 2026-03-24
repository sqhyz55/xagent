"""Unit tests for web ingestion pipeline."""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from xagent.core.tools.core.RAG_tools.core.schemas import (
    IngestionConfig,
    IngestionResult,
    WebCrawlConfig,
)
from xagent.core.tools.core.RAG_tools.pipelines.web_ingestion import run_web_ingestion
from xagent.core.tools.core.RAG_tools.utils.string_utils import sanitize_for_doc_id


class TestWebIngestionPipeline:
    """Test web ingestion pipeline functionality."""

    @pytest.fixture
    def crawl_config(self):
        """Create a test crawl configuration."""
        return WebCrawlConfig(
            start_url="https://example.com",
            max_pages=3,
            max_depth=1,
            concurrent_requests=1,
            request_delay=0,
        )

    @pytest.fixture
    def ingestion_config(self):
        """Create a test ingestion configuration."""
        return IngestionConfig(
            chunk_size=500,
            chunk_overlap=100,
        )

    @pytest.mark.asyncio
    async def test_successful_web_ingestion(self, crawl_config, ingestion_config):
        """Test successful web ingestion."""
        # Mock crawler results
        mock_crawl_results = [
            MagicMock(
                url="https://example.com/page1",
                title="Page 1",
                content_markdown="# Page 1\n\nContent for page 1.",
                status="success",
                depth=0,
                timestamp=datetime(2025, 1, 1, 12, 0, 0),
                content_length=50,
            ),
            MagicMock(
                url="https://example.com/page2",
                title="Page 2",
                content_markdown="# Page 2\n\nContent for page 2.",
                status="success",
                depth=0,
                timestamp=datetime(2025, 1, 1, 12, 0, 0),
                content_length=50,
            ),
        ]

        # Mock ingestion results
        mock_ingestion_result = IngestionResult(
            status="success",
            doc_id="test_doc_id",
            parse_hash="test_hash",
            chunk_count=5,
            embedding_count=5,
            vector_count=5,
            completed_steps=[],
            failed_step=None,
            message="Success",
            warnings=[],
        )

        # Mock the crawler and document ingestion
        with patch(
            "xagent.core.tools.core.RAG_tools.pipelines.web_ingestion.WebCrawler"
        ) as mock_crawler_class:
            mock_crawler = MagicMock()
            mock_crawler.crawl = AsyncMock(return_value=mock_crawl_results)
            mock_crawler.total_urls_found = 2
            mock_crawler.failed_urls = {}
            mock_crawler_class.return_value = mock_crawler

            with patch(
                "xagent.core.tools.core.RAG_tools.pipelines.web_ingestion.run_document_ingestion",
                return_value=mock_ingestion_result,
            ):
                result = await run_web_ingestion(
                    collection="test_collection",
                    crawl_config=crawl_config,
                    ingestion_config=ingestion_config,
                )

        # Verify result
        assert result.status == "success"
        assert result.collection == "test_collection"
        assert result.pages_crawled == 2
        assert result.documents_created == 2
        assert result.chunks_created == 10  # 5 per page
        assert result.embeddings_created == 10

    @pytest.mark.asyncio
    async def test_crawl_failure(self, crawl_config, ingestion_config):
        """Test handling of crawl failure."""
        # Mock crawler exception
        with patch(
            "xagent.core.tools.core.RAG_tools.pipelines.web_ingestion.WebCrawler"
        ) as mock_crawler_class:
            mock_crawler = MagicMock()
            mock_crawler.crawl = AsyncMock(side_effect=Exception("Crawl failed"))
            mock_crawler_class.return_value = mock_crawler

            result = await run_web_ingestion(
                collection="test_collection",
                crawl_config=crawl_config,
                ingestion_config=ingestion_config,
            )

        # Should return error status
        assert result.status == "error"
        assert result.pages_crawled == 0
        assert result.documents_created == 0
        assert "Crawl failed" in result.message

    @pytest.mark.asyncio
    async def test_partial_ingestion_failure(self, crawl_config, ingestion_config):
        """Test handling of partial ingestion failures."""
        # Mock crawl results
        mock_crawl_results = [
            MagicMock(
                url="https://example.com/page1",
                title="Page 1",
                content_markdown="# Page 1\n\nContent",
                status="success",
                depth=0,
                timestamp=datetime(2025, 1, 1, 12, 0, 0),
                content_length=30,
            ),
            MagicMock(
                url="https://example.com/page2",
                title="Page 2",
                content_markdown="# Page 2\n\nContent",
                status="success",
                depth=0,
                timestamp=datetime(2025, 1, 1, 12, 0, 0),
                content_length=30,
            ),
        ]

        # Mock mixed ingestion results
        success_result = IngestionResult(
            status="success",
            doc_id="doc1",
            parse_hash="hash1",
            chunk_count=5,
            embedding_count=5,
            vector_count=5,
            completed_steps=[],
            failed_step=None,
            message="Success",
            warnings=[],
        )

        error_result = IngestionResult(
            status="error",
            doc_id="doc2",
            parse_hash="hash2",
            chunk_count=0,
            embedding_count=0,
            vector_count=0,
            completed_steps=[],
            failed_step="parse",
            message="Parse failed",
            warnings=[],
        )

        with patch(
            "xagent.core.tools.core.RAG_tools.pipelines.web_ingestion.WebCrawler"
        ) as mock_crawler_class:
            mock_crawler = MagicMock()
            mock_crawler.crawl = AsyncMock(return_value=mock_crawl_results)
            mock_crawler.total_urls_found = 2
            mock_crawler.failed_urls = {}
            mock_crawler_class.return_value = mock_crawler

            with patch(
                "xagent.core.tools.core.RAG_tools.pipelines.web_ingestion.run_document_ingestion",
                side_effect=[success_result, error_result],
            ):
                result = await run_web_ingestion(
                    collection="test_collection",
                    crawl_config=crawl_config,
                    ingestion_config=ingestion_config,
                )

        # Should return partial status
        assert result.status == "partial"
        assert result.pages_crawled == 2
        assert result.documents_created == 1
        assert result.pages_failed == 1
        assert len(result.failed_urls) == 1
        assert "https://example.com/page2" in result.failed_urls

    @pytest.mark.asyncio
    async def test_empty_crawl_results(self, crawl_config, ingestion_config):
        """Test handling of empty crawl results."""
        with patch(
            "xagent.core.tools.core.RAG_tools.pipelines.web_ingestion.WebCrawler"
        ) as mock_crawler_class:
            mock_crawler = MagicMock()
            mock_crawler.crawl = AsyncMock(return_value=[])
            mock_crawler.total_urls_found = 0
            mock_crawler.failed_urls = {}
            mock_crawler_class.return_value = mock_crawler

            result = await run_web_ingestion(
                collection="test_collection",
                crawl_config=crawl_config,
                ingestion_config=ingestion_config,
            )

        # Should handle gracefully
        assert result.status == "success"
        assert result.pages_crawled == 0
        assert result.documents_created == 0

    @pytest.mark.asyncio
    async def test_ingestion_config_defaults(self, crawl_config):
        """Test that ingestion config defaults are applied."""
        # Mock successful crawl and ingestion
        mock_crawl_results = [
            MagicMock(
                url="https://example.com/page1",
                title="Page 1",
                content_markdown="# Page 1\n\nContent",
                status="success",
                depth=0,
                timestamp=datetime(2025, 1, 1, 12, 0, 0),
                content_length=30,
            )
        ]

        mock_ingestion_result = IngestionResult(
            status="success",
            doc_id="doc1",
            parse_hash="hash1",
            chunk_count=1,
            embedding_count=1,
            vector_count=1,
            completed_steps=[],
            failed_step=None,
            message="Success",
            warnings=[],
        )

        with patch(
            "xagent.core.tools.core.RAG_tools.pipelines.web_ingestion.WebCrawler"
        ) as mock_crawler_class:
            mock_crawler = MagicMock()
            mock_crawler.crawl = AsyncMock(return_value=mock_crawl_results)
            mock_crawler.total_urls_found = 1
            mock_crawler.failed_urls = {}
            mock_crawler_class.return_value = mock_crawler

            with patch(
                "xagent.core.tools.core.RAG_tools.pipelines.web_ingestion.run_document_ingestion",
                return_value=mock_ingestion_result,
            ) as mock_ingest:
                # Call without ingestion config
                await run_web_ingestion(
                    collection="test_collection",
                    crawl_config=crawl_config,
                )

                # Verify default config was used
                mock_ingest.assert_called_once()
                call_args = mock_ingest.call_args
                assert call_args[1]["ingestion_config"] is not None

    @pytest.mark.asyncio
    async def test_progress_callback(self, crawl_config, ingestion_config):
        """Test progress callback during ingestion."""
        progress_updates = []

        def progress_callback(message, completed, total):
            progress_updates.append((message, completed, total))

        mock_crawl_results = [
            MagicMock(
                url=f"https://example.com/page{i}",
                title=f"Page {i}",
                content_markdown=f"# Page {i}\n\nContent",
                status="success",
                depth=0,
                timestamp=datetime(2025, 1, 1, 12, 0, 0),
                content_length=30,
            )
            for i in range(3)
        ]

        mock_ingestion_result = IngestionResult(
            status="success",
            doc_id="doc1",
            parse_hash="hash1",
            chunk_count=1,
            embedding_count=1,
            vector_count=1,
            completed_steps=[],
            failed_step=None,
            message="Success",
            warnings=[],
        )

        with patch(
            "xagent.core.tools.core.RAG_tools.pipelines.web_ingestion.WebCrawler"
        ) as mock_crawler_class:
            mock_crawler = MagicMock()
            mock_crawler.crawl = AsyncMock(return_value=mock_crawl_results)
            mock_crawler.total_urls_found = 3
            mock_crawler.failed_urls = {}
            mock_crawler_class.return_value = mock_crawler

            with patch(
                "xagent.core.tools.core.RAG_tools.pipelines.web_ingestion.run_document_ingestion",
                return_value=mock_ingestion_result,
            ):
                await run_web_ingestion(
                    collection="test_collection",
                    crawl_config=crawl_config,
                    ingestion_config=ingestion_config,
                    progress_callback=progress_callback,
                )

        # Progress callback should have been called
        assert len(progress_updates) == 3
        assert all(len(update) == 3 for update in progress_updates)

    @pytest.mark.asyncio
    async def test_elapsed_time_tracking(self, crawl_config, ingestion_config):
        """Test that elapsed time is tracked."""
        mock_crawl_results = [
            MagicMock(
                url="https://example.com/page1",
                title="Page 1",
                content_markdown="# Page 1\n\nContent",
                status="success",
                depth=0,
                timestamp=datetime(2025, 1, 1, 12, 0, 0),
                content_length=30,
            )
        ]

        mock_ingestion_result = IngestionResult(
            status="success",
            doc_id="doc1",
            parse_hash="hash1",
            chunk_count=1,
            embedding_count=1,
            vector_count=1,
            completed_steps=[],
            failed_step=None,
            message="Success",
            warnings=[],
        )

        with patch(
            "xagent.core.tools.core.RAG_tools.pipelines.web_ingestion.WebCrawler"
        ) as mock_crawler_class:
            mock_crawler = MagicMock()
            mock_crawler.crawl = AsyncMock(return_value=mock_crawl_results)
            mock_crawler.total_urls_found = 1
            mock_crawler.failed_urls = {}
            mock_crawler_class.return_value = mock_crawler

            with patch(
                "xagent.core.tools.core.RAG_tools.pipelines.web_ingestion.run_document_ingestion",
                return_value=mock_ingestion_result,
            ):
                result = await run_web_ingestion(
                    collection="test_collection",
                    crawl_config=crawl_config,
                    ingestion_config=ingestion_config,
                )

        # Elapsed time should be tracked
        assert result.elapsed_time_ms >= 0


def test_sanitize_for_doc_id_behavior() -> None:
    """Test sanitize_for_doc_id behavior used by web ingestion."""
    # Replaces spaces and dots with underscores.
    assert sanitize_for_doc_id("report 2024.pdf") == "report_2024_pdf"

    # Path traversal-like input is normalized to safe token.
    assert sanitize_for_doc_id("../../etc/passwd") == "etc_passwd"

    # Non-allowed symbols collapse into underscores and trim boundaries.
    assert sanitize_for_doc_id("  .test.  ") == "test"

    # Empty input falls back to generated short identifier.
    fallback = sanitize_for_doc_id("")
    assert len(fallback) == 8
    assert fallback.isalnum()
