"""Contract tests for KB/RAG user scope context propagation."""

from __future__ import annotations

import asyncio

from xagent.core.tools.core.RAG_tools.utils.user_scope import (
    get_user_scope,
    resolve_user_scope,
    user_scope_context,
)


def test_resolve_user_scope_prefers_explicit_values() -> None:
    """Explicit scope arguments should override context values."""
    with user_scope_context(user_id=10, is_admin=False):
        resolved = resolve_user_scope(user_id=99, is_admin=True)
    assert resolved.user_id == 99
    assert resolved.is_admin is True


def test_resolve_user_scope_none_falls_back_to_context() -> None:
    """When user_id and is_admin are both None, should fall back to context."""
    with user_scope_context(user_id=42, is_admin=True):
        resolved = resolve_user_scope(user_id=None, is_admin=None)
    assert resolved.user_id == 42
    assert resolved.is_admin is True


def test_resolve_user_scope_explicit_false_is_admin() -> None:
    """Explicit is_admin=False should not fall back to context even if context has True."""
    with user_scope_context(user_id=10, is_admin=True):
        resolved = resolve_user_scope(user_id=None, is_admin=False)
    assert resolved.user_id is None
    assert resolved.is_admin is False


def test_user_scope_context_resets_after_exit() -> None:
    """Context should be restored after exiting scope manager."""
    before = get_user_scope()
    with user_scope_context(user_id=7, is_admin=False):
        during = get_user_scope()
        assert during.user_id == 7
    after = get_user_scope()
    assert after == before


def test_context_isolation_between_async_tasks() -> None:
    """Async tasks should keep their own context values."""

    async def _worker(user_id: int) -> int:
        with user_scope_context(user_id=user_id, is_admin=False):
            await asyncio.sleep(0)
            scope = get_user_scope()
            assert scope.user_id == user_id
            return int(scope.user_id or -1)

    async def _main() -> list[int]:
        return list(await asyncio.gather(_worker(1), _worker(2)))

    result = asyncio.run(_main())
    assert sorted(result) == [1, 2]
