from typing import Any

import pytest

from xagent.core.tools.core.RAG_tools.management.collection_manager import (
    collection_manager,
    validate_document_processing_sync,
)


class _DummyCollection:
    """简化版 CollectionInfo，用于隔离测试逻辑."""

    def __init__(
        self,
        *,
        allow_mixed_parse_methods: bool = False,
        skip_config_validation: bool = False,
        collection_locked: bool = False,
    ) -> None:
        self.allow_mixed_parse_methods = allow_mixed_parse_methods
        self.skip_config_validation = skip_config_validation
        self.collection_locked = collection_locked


def test_validate_document_processing_raises_for_incompatible_type_when_collection_missing(
    monkeypatch: Any,
) -> None:
    """当 collection 不存在时，也应对扩展名与 parse_method 做基础自洽校验."""

    async def _raise_value_error(collection_name: str) -> Any:  # type: ignore[unused-argument]
        raise ValueError("collection not found")

    monkeypatch.setattr(collection_manager, "get_collection", _raise_value_error)

    with pytest.raises(ValueError) as exc_info:
        validate_document_processing_sync(
            collection_name="kb-docx",
            file_path="/tmp/sample.docx",
            parsing_method="pypdf",
            chunking_method="recursive",
        )

    msg = str(exc_info.value)
    assert "not compatible" in msg
    assert ".docx" in msg
    # 支持列表应来自 docx 对应 parser，而不是 pypdf
    assert "Supported methods" in msg


def test_validate_document_processing_allows_default_method_without_collection(
    monkeypatch: Any,
) -> None:
    """parse_method=default 时，不应触发类型兼容性校验."""

    async def _raise_value_error(collection_name: str) -> Any:  # type: ignore[unused-argument]
        raise ValueError("collection not found")

    monkeypatch.setattr(collection_manager, "get_collection", _raise_value_error)

    # 不应抛出异常
    validate_document_processing_sync(
        collection_name="kb-docx",
        file_path="/tmp/sample.docx",
        parsing_method="default",
        chunking_method="recursive",
    )


def test_validate_document_processing_respects_allow_mixed(monkeypatch: Any) -> None:
    """当 allow_mixed_parse_methods=True 时，应跳过扩展名与 parse_method 的限制."""

    async def _get_collection(collection_name: str) -> _DummyCollection:  # type: ignore[unused-argument]
        return _DummyCollection(allow_mixed_parse_methods=True)

    monkeypatch.setattr(collection_manager, "get_collection", _get_collection)

    # 虽然 .docx + pypdf 正常情况下不兼容，但在 allow_mixed=True 时应允许通过
    validate_document_processing_sync(
        collection_name="kb-docx",
        file_path="/tmp/sample.docx",
        parsing_method="pypdf",
        chunking_method="recursive",
    )
