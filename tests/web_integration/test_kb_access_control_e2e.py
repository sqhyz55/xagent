"""E2E tests for KB collection-level access control (403/404 contracts).

These tests complement ``test_multitenancy_isolation_e2e.py`` by asserting
single, explicit HTTP outcomes on routes that use
:func:`xagent.web.api.kb._ensure_collection_access` (and related rename rules).

Ingest uses a **stub embedding pipeline** so CI does not depend on external
embedding keys; assertions are HTTP semantics (403/404), not provider quality.
Real embedding coverage lives in ``real_rag`` suites (e.g. multitenancy E2E).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from tests.web_integration.http_helpers import http_detail
from xagent.core.model.embedding.base import BaseEmbedding
from xagent.core.model.model import EmbeddingModelConfig
from xagent.core.tools.core.RAG_tools.core.schemas import CollectionInfo

pytestmark = [pytest.mark.e2e, pytest.mark.contract_stub]


class _StubEmbeddingAdapter(BaseEmbedding):
    """Deterministic embedding for access-control contract tests."""

    def encode(
        self,
        text: Any,
        dimension: int | None = None,
        instruct: str | None = None,
    ) -> Any:
        if isinstance(text, str):
            return [float(len(text)), 0.0]
        return [[float(len(item)), float(index)] for index, item in enumerate(text)]

    def get_dimension(self) -> int:
        return 2

    @property
    def abilities(self) -> list[str]:
        return ["embedding"]


@pytest.fixture
def stub_embedding_config() -> EmbeddingModelConfig:
    """Stub embedding configuration."""
    return EmbeddingModelConfig(
        id="e2e-ac-embedding",
        model_name="e2e-ac-embedding-model",
        model_provider="test",
        dimension=2,
    )


@pytest.fixture
def stub_embedding_adapter() -> _StubEmbeddingAdapter:
    return _StubEmbeddingAdapter()


@pytest.fixture(autouse=True)
def mock_access_control_rag_pipeline(
    monkeypatch: pytest.MonkeyPatch,
    stub_embedding_config: EmbeddingModelConfig,
    stub_embedding_adapter: _StubEmbeddingAdapter,
) -> None:
    """Stub RAG embedding resolution so ingest succeeds without external APIs."""
    from xagent.core.tools.core.RAG_tools import pipelines as pipelines_module
    from xagent.core.tools.core.RAG_tools.management import collection_manager as cm_mod
    from xagent.core.tools.core.RAG_tools.utils import model_resolver

    mgr = cm_mod.collection_manager

    async def mock_get_collection(collection_name: str) -> CollectionInfo:
        return CollectionInfo(
            name=collection_name,
            embedding_model_id="e2e-ac-embedding",
            embedding_dimension=2,
        )

    async def mock_initialize_collection(
        collection_name: str, embedding_model_id: str
    ) -> CollectionInfo:
        return CollectionInfo(
            name=collection_name,
            embedding_model_id=embedding_model_id,
            embedding_dimension=2,
        )

    def mock_resolve_embedding_adapter(
        model_id: str | None = None, **kwargs: Any
    ) -> tuple[EmbeddingModelConfig, BaseEmbedding]:
        return (stub_embedding_config, stub_embedding_adapter)

    monkeypatch.setattr(mgr, "get_collection", mock_get_collection)
    monkeypatch.setattr(
        mgr, "initialize_collection_embedding", mock_initialize_collection
    )
    monkeypatch.setattr(
        model_resolver,
        "resolve_embedding_adapter",
        mock_resolve_embedding_adapter,
    )
    monkeypatch.setattr(
        pipelines_module.document_ingestion,
        "_resolve_embedding_adapter",
        lambda cfg: (stub_embedding_config, stub_embedding_adapter),
    )


def _register_and_login(
    client: TestClient, username: str, password: str, email: str
) -> str:
    """Register (idempotent) and return JWT access token."""
    reg = client.post(
        "/api/auth/register",
        json={"username": username, "password": password, "email": email},
    )
    assert reg.status_code in (200, 400)
    login = client.post(
        "/api/auth/login",
        json={"username": username, "password": password},
    )
    assert login.status_code == 200
    return str(login.json()["access_token"])


def _write_sample_txt(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("access-control e2e sample content", encoding="utf-8")
    return path


def _ingest_txt(
    client: TestClient,
    token: str,
    collection: str,
    file_path: Path,
) -> None:
    headers = {"Authorization": f"Bearer {token}"}
    with open(file_path, "rb") as handle:
        resp = client.post(
            "/api/kb/ingest",
            files={"file": (file_path.name, handle, "text/plain")},
            data={"collection": collection},
            headers=headers,
        )
    assert resp.status_code == 200, http_detail(resp)


class TestKbAccessControlContract:
    """Strict status-code checks for KB access boundaries."""

    def test_parse_result_cross_tenant_returns_403(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        """Another tenant cannot read parse results for a foreign collection name."""
        t1 = _register_and_login(
            client, "ac_parse_t1", "pw-t1-", "ac_parse_t1@example.com"
        )
        t2 = _register_and_login(
            client, "ac_parse_t2", "pw-t2-", "ac_parse_t2@example.com"
        )

        coll = "ac_parse_coll_t1"
        sample = _write_sample_txt(tmp_path / "ac_parse_doc.txt")
        _ingest_txt(client, t1, coll, sample)

        doc_id = "legit-doc-id-01"
        url = f"/api/kb/collections/{coll}/parses/{doc_id}/parse_result"
        resp = client.get(
            url,
            headers={"Authorization": f"Bearer {t2}"},
        )
        assert resp.status_code == 403
        assert "Access denied" in resp.json()["detail"]

    def test_parse_result_unknown_collection_returns_404(
        self, client: TestClient
    ) -> None:
        """A collection name that does not exist anywhere yields 404 (not 403)."""
        token = _register_and_login(
            client, "ac_parse_404", "pw-404-", "ac_parse_404@example.com"
        )
        missing = "ac_no_such_collection_xyz"
        doc_id = "legit-doc-id-02"
        url = f"/api/kb/collections/{missing}/parses/{doc_id}/parse_result"
        resp = client.get(
            url,
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"].lower()

    def test_rename_target_name_taken_by_other_tenant_returns_403(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        """Renaming into a name that exists on another tenant is forbidden."""
        t1 = _register_and_login(client, "ac_rn_t1", "pw-r1-", "ac_rn_t1@example.com")
        t2 = _register_and_login(client, "ac_rn_t2", "pw-r2-", "ac_rn_t2@example.com")

        coll_a = "ac_rename_source_coll"
        coll_b = "ac_rename_target_coll"
        sample1 = _write_sample_txt(tmp_path / "ac_rn_a.txt")
        sample2 = _write_sample_txt(tmp_path / "ac_rn_b.txt")
        _ingest_txt(client, t1, coll_a, sample1)
        _ingest_txt(client, t2, coll_b, sample2)

        resp = client.put(
            f"/api/kb/collections/{coll_a}",
            data={"new_name": coll_b},
            headers={"Authorization": f"Bearer {t1}"},
        )
        assert resp.status_code == 403
        assert "Access denied" in resp.json()["detail"]

    def test_documents_check_cross_tenant_returns_403(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        """Duplicate-check on another tenant's collection name is forbidden."""
        t1 = _register_and_login(client, "ac_chk_t1", "pw-c1-", "ac_chk_t1@example.com")
        t2 = _register_and_login(client, "ac_chk_t2", "pw-c2-", "ac_chk_t2@example.com")

        coll = "ac_check_foreign_coll"
        sample = _write_sample_txt(tmp_path / "ac_chk.txt")
        _ingest_txt(client, t1, coll, sample)

        resp = client.post(
            f"/api/kb/collections/{coll}/documents/check",
            json={"filenames": ["any.txt"]},
            headers={"Authorization": f"Bearer {t2}"},
        )
        assert resp.status_code == 403
        assert "Access denied" in resp.json()["detail"]

    def test_save_collection_config_cross_tenant_returns_403(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        """Saving config against another tenant's collection name is forbidden."""
        t1 = _register_and_login(client, "ac_cfg_t1", "pw-g1-", "ac_cfg_t1@example.com")
        t2 = _register_and_login(client, "ac_cfg_t2", "pw-g2-", "ac_cfg_t2@example.com")

        coll = "ac_config_foreign_coll"
        sample = _write_sample_txt(tmp_path / "ac_cfg.txt")
        _ingest_txt(client, t1, coll, sample)

        resp = client.post(
            f"/api/kb/collections/{coll}/config",
            json={"embedding_model_id": "text-embedding-v4"},
            headers={"Authorization": f"Bearer {t2}"},
        )
        assert resp.status_code == 403
        assert "Access denied" in resp.json()["detail"]
