import uuid

import numpy as np
import pytest

from faceverification.core.vectordb import VectorDB

pytestmark = pytest.mark.integration


def test_vectordb_adds_and_queries_embedding_with_real_chromadb():
    db = VectorDB(
        distance_metric="l2",
        name_collection=f"test_faces_{uuid.uuid4().hex}",
    )
    stored_embedding = np.array([0.1, 0.2, 0.3])
    query_embedding = np.array([0.11, 0.19, 0.31])

    db.add_embedding(stored_embedding, {"name": "Ada"})

    metadata, distance = db.query_embedding(
        query_embedding,
        threshold=0.1,
        n_results=1,
    )

    assert metadata == {"name": "Ada"}
    assert distance == pytest.approx(np.linalg.norm(query_embedding - stored_embedding))


def test_vectordb_raises_when_real_chromadb_collection_is_empty():
    db = VectorDB(
        distance_metric="l2",
        name_collection=f"test_empty_faces_{uuid.uuid4().hex}",
    )

    with pytest.raises(ValueError, match="Add a person before verifying faces"):
        db.query_embedding(np.array([0.1, 0.2, 0.3]), threshold=0.1, n_results=1)


def test_vectordb_persists_embeddings_between_instances(tmp_path):
    persist_directory = tmp_path / "chroma"
    collection_name = f"test_persistent_faces_{uuid.uuid4().hex}"
    stored_embedding = np.array([0.1, 0.2, 0.3])
    query_embedding = np.array([0.11, 0.19, 0.31])

    first_db = VectorDB(
        distance_metric="l2",
        name_collection=collection_name,
        persist_directory=str(persist_directory),
    )
    first_db.add_embedding(stored_embedding, {"name": "Ada"})

    second_db = VectorDB(
        distance_metric="l2",
        name_collection=collection_name,
        persist_directory=str(persist_directory),
    )
    metadata, distance = second_db.query_embedding(
        query_embedding,
        threshold=0.1,
        n_results=1,
    )

    assert metadata == {"name": "Ada"}
    assert distance == pytest.approx(np.linalg.norm(query_embedding - stored_embedding))
