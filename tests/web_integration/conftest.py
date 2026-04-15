"""
Shared fixtures for web integration tests.

This module provides common fixtures used across all E2E web integration tests.
"""

import os
import tempfile

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from xagent.web.api.auth import hash_password
from xagent.web.api.kb import kb_router
from xagent.web.models.database import Base, get_db
from xagent.web.models.user import User


@pytest.fixture(scope="function")
def test_env():
    """Setup test database and app for E2E tests."""
    temp_db_fd, temp_db_path = tempfile.mkstemp(suffix=".db")
    os.close(temp_db_fd)

    test_engine = create_engine(f"sqlite:///{temp_db_path}")
    TestingSessionLocal = sessionmaker(bind=test_engine)

    def override_get_db():
        db = TestingSessionLocal()
        try:
            yield db
        finally:
            db.close()

    app = FastAPI()
    app.include_router(kb_router)
    # Include auth router for register/login endpoints
    from xagent.web.api.auth import auth_router

    app.include_router(auth_router)
    app.dependency_overrides[get_db] = override_get_db

    Base.metadata.create_all(bind=test_engine)

    session = TestingSessionLocal()
    user = User(
        username="testuser", password_hash=hash_password("test"), is_admin=False
    )
    session.add(user)
    session.commit()
    session.refresh(user)

    # Mock JWT token (must include type="access" for get_current_user)
    from datetime import datetime, timedelta

    import jwt

    from xagent.web.auth_config import JWT_ALGORITHM, JWT_SECRET_KEY

    payload = {
        "sub": user.username,
        "user_id": user.id,
        "type": "access",
        "exp": datetime.utcnow() + timedelta(hours=1),
        "iat": datetime.utcnow(),
    }
    token = jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
    headers = {"Authorization": f"Bearer {token}"}

    yield app, headers, user, TestingSessionLocal

    session.close()
    os.unlink(temp_db_path)


@pytest.fixture
def client(test_env):
    """Provide test client for E2E tests."""
    app, headers, user, TestingSessionLocal = test_env
    return TestClient(app)


@pytest.fixture
def auth_headers(test_env):
    """Provide authentication headers for E2E tests."""
    app, headers, user, TestingSessionLocal = test_env
    return headers


@pytest.fixture
def db_session_factory(test_env):
    """Expose SQLAlchemy session factory for test data adjustments."""
    app, headers, user, TestingSessionLocal = test_env
    return TestingSessionLocal


@pytest.fixture
def clean_storage() -> None:
    """Backward-compatible no-op fixture for legacy e2e tests.

    Global RAG storage isolation is already handled by root ``isolate_rag_storage``
    autouse fixture in ``tests/conftest.py``.
    """
    return None
