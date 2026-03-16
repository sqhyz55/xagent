"""Tests for storage factory and coordinator wiring."""

from xagent.core.tools.core.RAG_tools.storage import factory


def test_get_kb_write_coordinator_is_singleton(monkeypatch) -> None:
    """Factory should return the same coordinator instance per process."""
    monkeypatch.setattr(factory, "_default_coordinator", None)

    first = factory.get_kb_write_coordinator()
    second = factory.get_kb_write_coordinator()

    assert first is second


def test_accessors_return_coordinator_stores(monkeypatch) -> None:
    """Convenience accessors should delegate to the singleton coordinator."""
    monkeypatch.setattr(factory, "_default_coordinator", None)

    coordinator = factory.get_kb_write_coordinator()
    assert factory.get_metadata_store() is coordinator.metadata_store()
    assert factory.get_vector_index_store() is coordinator.vector_index_store()
