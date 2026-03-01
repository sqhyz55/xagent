import pytest
from cryptography.fernet import Fernet
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from xagent.core.model.model import VectorDBConfig, VectorDBType
from xagent.core.model.storage.db.adapter import SQLAlchemyModelHub
from xagent.core.model.storage.db.db_models import create_model_table

Base = declarative_base()
Model = create_model_table(Base)


@pytest.fixture(autouse=True)
def setup_encryption_key(monkeypatch):
    # Set a valid Base64-encoded 32-byte key for Fernet encryption
    key = "RQMpe38gK3m0szjpSmTNw_sP3Y54r6hDc6JewBoPKXc="
    monkeypatch.setenv("ENCRYPTION_KEY", key)
    return key


@pytest.fixture
def db_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


def test_vector_db_config_roundtrip(db_session):
    hub = SQLAlchemyModelHub(db_session, Model)

    # 1. Store a VectorDBConfig with extra config
    config_id = "test-weaviate"
    vdb_config = VectorDBConfig(
        id=config_id,
        model_name="My Weaviate",
        base_url="http://localhost:8080",
        api_key="test-key",
        db_type=VectorDBType.WEAVIATE_SAAS,
        config={"grpc_port": 50051, "secure": True},
    )

    hub.store(vdb_config)

    # 2. Load it back
    loaded = hub.load(config_id)

    assert isinstance(loaded, VectorDBConfig)
    assert loaded.id == config_id
    assert loaded.db_type == VectorDBType.WEAVIATE_SAAS
    assert loaded.config["grpc_port"] == 50051
    assert loaded.config["secure"] is True
    assert loaded.api_key == "test-key"


def test_vector_db_config_list(db_session):
    hub = SQLAlchemyModelHub(db_session, Model)

    # Store multiple configs (must provide api_key to avoid IntegrityError on _api_key_encrypted)
    hub.store(
        VectorDBConfig(
            id="v1", model_name="V1", db_type=VectorDBType.LANCEDB, api_key=""
        )
    )
    hub.store(
        VectorDBConfig(
            id="v2",
            model_name="V2",
            db_type=VectorDBType.WEAVIATE_SAAS,
            config={"k": "v"},
            api_key="",
        )
    )

    configs = hub.list()
    assert len(configs) >= 2
    assert isinstance(configs["v1"], VectorDBConfig)
    assert isinstance(configs["v2"], VectorDBConfig)
    assert configs["v2"].config["k"] == "v"


def test_vector_db_config_fallback(db_session, setup_encryption_key):
    hub = SQLAlchemyModelHub(db_session, Model)

    # Generate a valid encrypted token
    cipher = Fernet(setup_encryption_key.encode())
    valid_encrypted = cipher.encrypt(b"dummy-key").decode()

    # Manually insert a record with invalid db_type in model_provider
    db_record = Model(
        model_id="invalid-db",
        category="vector_db",
        model_provider="unknown_db",
        model_name="Unknown",
        _api_key_encrypted=valid_encrypted,
        is_active=True,
    )
    db_session.add(db_record)
    db_session.commit()

    # Should fallback to LANCEDB and not crash
    loaded = hub.load("invalid-db")
    assert isinstance(loaded, VectorDBConfig)
    assert loaded.db_type == VectorDBType.LANCEDB
    assert loaded.api_key == "dummy-key"


def test_vector_db_legacy_weaviate_local_alias(db_session, setup_encryption_key):
    hub = SQLAlchemyModelHub(db_session, Model)

    cipher = Fernet(setup_encryption_key.encode())
    valid_encrypted = cipher.encrypt(b"dummy-key").decode()

    # Legacy provider value should be mapped to the new canonical enum value.
    db_record = Model(
        model_id="legacy-weaviate-local",
        category="vector_db",
        model_provider="weaviate_local",
        model_name="LegacyWeaviateLocal",
        _api_key_encrypted=valid_encrypted,
        is_active=True,
    )
    db_session.add(db_record)
    db_session.commit()

    loaded = hub.load("legacy-weaviate-local")
    assert isinstance(loaded, VectorDBConfig)
    assert loaded.db_type == VectorDBType.WEAVIATE
