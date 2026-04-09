"""
Shared fixtures for web integration tests.

This module provides common fixtures used across all E2E web integration tests.
"""

import pytest


@pytest.fixture
def test_env():
    """Provide complete test environment for E2E tests.

    This fixture sets up:
    - FastAPI app with all routes
    - Authentication headers
    - Test database session
    - Temporary upload directory

    This fixture reuses the test environment from the web API tests.
    """
    from tests.web.api.test_kb_dir import test_env as kb_test_env

    # Reuse existing test environment from kb_dir tests
    yield from kb_test_env()


@pytest.fixture
def client(test_env):
    """Provide test client for E2E tests."""
    app, headers, user, TestingSessionLocal = test_env
    from fastapi.testclient import TestClient

    return TestClient(app)


@pytest.fixture
def auth_headers(test_env):
    """Provide authentication headers for E2E tests."""
    app, headers, user, TestingSessionLocal = test_env
    return headers
