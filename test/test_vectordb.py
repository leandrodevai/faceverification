from unittest.mock import Mock

import numpy as np
import pytest

from faceverification.core import vectordb
from faceverification.core.vectordb import VectorDB


class FakeClient:
    def __init__(self, settings):
        self.settings = settings
        self.collection = Mock()

    def get_or_create_collection(self, **kwargs):
        self.collection_kwargs = kwargs
        return self.collection


def test_init_creates_in_memory_collection_with_configured_metric(monkeypatch):
    clients = []

    def fake_client(settings):
        client = FakeClient(settings)
        clients.append(client)
        return client

    monkeypatch.setattr(vectordb.chromadb, "Client", fake_client)

    db = VectorDB(distance_metric="cosine", name_collection="faces")

    assert db.client is clients[0]
    assert clients[0].settings.is_persistent is False
    assert clients[0].settings.persist_directory == ""
    assert clients[0].collection_kwargs == {
        "name": "faces",
        "metadata": {"hnsw:space": "cosine"},
    }
    assert db.collection is clients[0].collection


def test_init_passes_persist_directory_when_configured(monkeypatch):
    clients = []

    def fake_client(settings):
        client = FakeClient(settings)
        clients.append(client)
        return client

    monkeypatch.setattr(vectordb.chromadb, "Client", fake_client)

    VectorDB(
        distance_metric="l2",
        name_collection="faces",
        persist_directory="tmp/chroma",
    )

    assert clients[0].settings.is_persistent is True
    assert clients[0].settings.persist_directory == "tmp/chroma"


def test_add_embedding_stores_embedding_metadata_and_generated_id(monkeypatch):
    collection = Mock()
    db = VectorDB.__new__(VectorDB)
    db.collection = collection
    embedding = np.array([0.1, 0.2, 0.3])
    metadata = {"name": "Ada"}

    monkeypatch.setattr(vectordb.uuid, "uuid4", lambda: "fixed-id")

    db.add_embedding(embedding, metadata)

    collection.add.assert_called_once()
    kwargs = collection.add.call_args.kwargs
    np.testing.assert_array_equal(kwargs["embeddings"][0], embedding)
    assert kwargs["metadatas"] == [metadata]
    assert kwargs["ids"] == ["fixed-id"]


def test_query_embedding_returns_closest_metadata_within_threshold():
    collection = Mock()
    collection.query.return_value = {
        "embeddings": [
            [
                np.array([3.0, 4.0]),
                np.array([0.2, 0.1]),
                np.array([1.0, 1.0]),
            ]
        ],
        "metadatas": [[{"name": "Far"}, {"name": "Near"}, {"name": "Middle"}]],
        "distances": [[5.0, 0.22, 1.41]],
    }
    db = VectorDB.__new__(VectorDB)
    db.collection = collection
    query_embedding = np.array([0.0, 0.0])

    metadata, distance = db.query_embedding(
        query_embedding,
        threshold=0.5,
        n_results=3,
    )

    assert metadata == {"name": "Near"}
    assert distance == np.linalg.norm(query_embedding - np.array([0.2, 0.1]))
    collection.query.assert_called_once()
    assert collection.query.call_args.kwargs["include"] == [
        "metadatas",
        "distances",
        "embeddings",
    ]
    assert collection.query.call_args.kwargs["n_results"] == 3
    np.testing.assert_array_equal(
        collection.query.call_args.kwargs["query_embeddings"][0],
        query_embedding,
    )


def test_query_embedding_returns_none_when_closest_distance_exceeds_threshold():
    collection = Mock()
    collection.query.return_value = {
        "embeddings": [[np.array([2.0, 0.0]), np.array([0.0, 3.0])]],
        "metadatas": [[{"name": "Ada"}, {"name": "Grace"}]],
        "distances": [[2.0, 3.0]],
    }
    db = VectorDB.__new__(VectorDB)
    db.collection = collection

    metadata, distance = db.query_embedding(
        np.array([0.0, 0.0]),
        threshold=1.0,
        n_results=2,
    )

    assert metadata is None
    assert distance == 2.0


def test_query_embedding_raises_when_database_has_no_embeddings():
    collection = Mock()
    collection.query.return_value = {
        "embeddings": [np.array([], dtype=float)],
        "metadatas": [[]],
        "distances": [[]],
    }
    db = VectorDB.__new__(VectorDB)
    db.collection = collection

    with pytest.raises(ValueError, match="Add a person before verifying faces"):
        db.query_embedding(np.array([0.0, 0.0]), threshold=1.0, n_results=2)
