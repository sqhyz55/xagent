"""Tests for KB uploaded-file/document bridge helpers."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from xagent.web.services import kb_file_service


def test_list_documents_for_user_delegates_to_vector_store_without_raw_lancedb() -> (
    None
):
    """Document listing should go through the storage contract, not LanceDB directly."""
    fake_store = SimpleNamespace()
    fake_store.list_document_records = lambda **kwargs: [
        SimpleNamespace(
            collection="demo",
            doc_id="doc-1",
            file_id="file-1",
            source_path="/tmp/demo.pdf",
        )
    ]

    with (
        patch.object(
            kb_file_service,
            "get_connection_from_env",
            side_effect=AssertionError("raw LanceDB connection should not be used"),
            create=True,
        ),
        patch.object(
            kb_file_service,
            "get_vector_index_store",
            return_value=fake_store,
            create=True,
        ) as mock_get_store,
    ):
        records = kb_file_service.list_documents_for_user(
            user_id=123,
            is_admin=False,
            collection_name="demo",
        )

    mock_get_store.assert_called_once_with()
    assert records == [
        {
            "collection": "demo",
            "doc_id": "doc-1",
            "file_id": "file-1",
            "source_path": "/tmp/demo.pdf",
        }
    ]
