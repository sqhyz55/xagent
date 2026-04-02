"""
E2E test for the real document ingestion pipeline.

This test is not meant for CI/CD but serves as a utility to run the full
document ingestion pipeline on a real PDF. Its primary purpose is to generate
a valid LanceDB database artifact that can be used for subsequent, separate
testing of the retrieval pipeline.
"""

from __future__ import annotations

import logging
import shutil
import uuid
from pathlib import Path
from typing import List, Optional, Union

import pytest

from xagent.core.model.embedding.base import BaseEmbedding
from xagent.core.model.model import EmbeddingModelConfig
from xagent.core.storage.manager import initialize_storage_manager
from xagent.core.tools.core.RAG_tools.core.schemas import (
    CollectionInfo,
    IngestionConfig,
    ParseMethod,
)
from xagent.core.tools.core.RAG_tools.management import collection_manager
from xagent.core.tools.core.RAG_tools.pipelines import document_ingestion
from xagent.core.tools.core.RAG_tools.utils import model_resolver

logger = logging.getLogger(__name__)


class _StubEmbeddingAdapter(BaseEmbedding):
    """Deterministic embedding adapter for tests to ensure reproducibility."""

    def encode(
        self,
        text: Union[str, List[str]],
        dimension: int | None = None,
        instruct: str | None = None,
    ) -> Union[List[float], List[List[float]]]:
        """Generates a fake vector based on text length."""
        if isinstance(text, str):
            return [float(len(text)), 0.0]
        return [[float(len(item)), float(index)] for index, item in enumerate(text)]

    def get_dimension(self) -> int:
        return 2

    @property
    def abilities(self) -> List[str]:
        return ["embedding"]


def _patch_embedding_adapter(monkeypatch: pytest.MonkeyPatch, model_id: str) -> None:
    """Stub embedding adapter resolution to avoid real model calls."""
    stub_config = EmbeddingModelConfig(
        id=model_id,
        model_name="test-embedding-model",
        model_provider="test",
        dimension=2,
    )
    stub_adapter = _StubEmbeddingAdapter()

    # Mock both document_ingestion and model_resolver's resolve_embedding_adapter
    monkeypatch.setattr(
        document_ingestion,
        "_resolve_embedding_adapter",
        lambda _cfg: (stub_config, stub_adapter),
    )

    # Also mock resolve_embedding_adapter used by collection_manager

    def mock_resolve_embedding_adapter(
        model_id: Optional[str] = None, **kwargs
    ) -> tuple[EmbeddingModelConfig, BaseEmbedding]:
        """Mock resolve_embedding_adapter for collection_manager."""
        return (stub_config, stub_adapter)

    monkeypatch.setattr(
        model_resolver,
        "resolve_embedding_adapter",
        mock_resolve_embedding_adapter,
    )

    # Also mock resolve_embedding_adapter in collection_manager module
    # since it has its own import and monkeypatch can't affect already imported references
    monkeypatch.setattr(
        collection_manager,
        "resolve_embedding_adapter",
        mock_resolve_embedding_adapter,
    )

    # Mock initialize_collection_embedding_sync to avoid hub/env resolution

    def mock_initialize_collection_embedding_sync(
        collection_name: str, embedding_model_id: str
    ) -> CollectionInfo:
        """Mock collection initialization."""
        return CollectionInfo(
            name=collection_name,
            embedding_model_id=embedding_model_id,
            embedding_dimension=2,  # Match stub config
        )

    monkeypatch.setattr(
        collection_manager,
        "initialize_collection_embedding_sync",
        mock_initialize_collection_embedding_sync,
    )


def test_run_real_ingestion_pipeline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Runs the real document ingestion pipeline to generate a DB artifact.
    This test's main goal is to produce a side-effect (the DB) for manual
    cross-pipeline testing.
    """
    # 1. --- Environment Setup ---
    # Use a predictable, persistent path for the database so we can access it later.
    # This path will be relative to the xagent project root.
    db_output_dir = Path("generated_db_for_test")
    # Clean up previous runs if they exist
    if db_output_dir.exists():
        shutil.rmtree(db_output_dir)
    db_output_dir.mkdir(parents=True, exist_ok=True)

    try:
        monkeypatch.setenv("LANCEDB_DIR", str(db_output_dir.resolve()))

        storage_root = tmp_path / "storage"
        uploads_dir = storage_root / "uploads"
        uploads_dir.mkdir(parents=True, exist_ok=True)
        initialize_storage_manager(str(storage_root), str(uploads_dir))

        embedding_model_id = "rag-test-embedding-deterministic"
        _patch_embedding_adapter(monkeypatch, embedding_model_id)

        logger.info("--- E2E Ingestion Pipeline Runner ---")
        logger.info(f"[*] Using output LanceDB directory: {db_output_dir.resolve()}")

        # 2. --- Pipeline Execution ---
        # Robustly locate the project root relative to this test file
        project_root = Path(__file__).resolve().parents[6]
        test_pdf = project_root / "tests" / "resources" / "test_files" / "test.pdf"
        collection = f"test_collection_{uuid.uuid4().hex[:8]}"

        logger.info(f"[*] Ingesting document: {test_pdf}")
        logger.info(f"[*] Target collection: {collection}")

        # Call the real pipeline function
        result = document_ingestion.process_document(
            collection=collection,
            source_path=str(test_pdf),
            config=IngestionConfig(
                embedding_model_id=embedding_model_id,
                parse_method=ParseMethod.PYPDF,
            ),
            user_id=1,
            is_admin=True,
        )

        # 3. --- Log Results ---
        logger.info("--- Ingestion Result ---")
        logger.info(f"[*] Status: {result.status}")
        logger.info(f"[*] Message: {result.message}")
        logger.info(f"[*] Doc ID: {result.doc_id}")
        completed_steps = [step.name for step in result.completed_steps]
        logger.info(f"[*] Steps completed: {completed_steps}")

        if result.failed_step:
            logger.error(f"[!] FAILED at step: {result.failed_step}")

        # Final check to ensure the test framework knows if it succeeded.
        assert result.status == "success", "Document ingestion pipeline failed."

        logger.info("\n\n[SUCCESS] Pipeline finished.")
        logger.info("The generated database is available for the next step at:")
        logger.info(f"==> {db_output_dir.resolve()}\n")
    finally:
        # Cleanup generated database directory
        if db_output_dir.exists():
            logger.info(
                f"[*] Cleaning up generated database: {db_output_dir.resolve()}"
            )
            shutil.rmtree(db_output_dir)
