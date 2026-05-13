"""Vector database adapter for face embedding storage and lookup.

This module wraps ChromaDB behind a small project-specific interface. The rest
of the application only needs to add face embeddings and query the nearest
stored embedding, while this class owns the Chroma collection setup and result
filtering.
"""

import uuid
from typing import Any, Mapping

import chromadb
import numpy as np
from chromadb.config import Settings as ChromaSettings

from faceverification.config import settings


class VectorDB:
    """Store and query face embeddings in a ChromaDB collection.

    The collection is configured with the selected HNSW distance metric and can
    run either in memory or against a persistent directory when one is provided.
    Query results are post-processed with NumPy so the service layer receives a
    simple `(metadata, distance)` pair.
    """

    def __init__(
        self,
        distance_metric: str | None = None,
        name_collection: str | None = None,
        persist_directory: str | None = None,
    ):
        """Initialize the ChromaDB client and face embeddings collection.

        Args:
            distance_metric: HNSW distance metric used by ChromaDB. Common
                values are `"l2"`, `"cosine"`, and `"ip"`.
            name_collection: Name of the collection that stores face
                embeddings.
            persist_directory: Optional directory where ChromaDB should persist
                data. When omitted, the database runs in memory.
        """
        if distance_metric is None:
            distance_metric = settings.vector_db_distance_metric
        if name_collection is None:
            name_collection = settings.vector_db_collection
        if persist_directory is None:
            persist_directory = settings.vector_db_persist_directory

        chroma_settings = ChromaSettings(
            is_persistent=bool(persist_directory),
            persist_directory=persist_directory or "",
        )
        self.client = chromadb.Client(chroma_settings)

        self.collection = self.client.get_or_create_collection(
            name=name_collection, metadata={"hnsw:space": distance_metric}
        )

    def add_embedding(self, embedding: np.ndarray, metadata: Mapping[str, Any]) -> None:
        """Add one face embedding and its metadata to the collection.

        Args:
            embedding: Face embedding vector produced by the face recognition
                model.
            metadata: Metadata associated with the embedding, such as the
                person's name.
        """
        self.collection.add(
            embeddings=[embedding],
            metadatas=[metadata],
            ids=[str(uuid.uuid4())],
        )

    def query_embedding(
        self,
        embedding: np.ndarray,
        threshold: float | None = None,
        n_results: int | None = None,
    ) -> tuple[Mapping[str, Any] | None, float]:
        """Find the closest stored embedding within the configured threshold.

        Args:
            embedding: Query embedding vector to compare against stored
                embeddings.
            threshold: Maximum Euclidean distance accepted as a match.
            n_results: Number of nearest ChromaDB candidates to inspect.

        Returns:
            A tuple containing the matched metadata and its distance. If no
            candidate is within the threshold, metadata is `None` and the best
            distance is still returned.
        """
        if threshold is None:
            threshold = settings.face_match_threshold
        if n_results is None:
            n_results = settings.vector_db_n_results

        result = self.collection.query(
            query_embeddings=[embedding],
            include=["metadatas", "distances", "embeddings"],
            n_results=n_results,
        )
        embeddings = result.get("embeddings")
        if not embeddings or embeddings[0] is None or len(embeddings[0]) == 0:
            raise ValueError(
                "No record found in the vector database. "
                "Add a person before verifying faces."
            )

        best_dist = 100000
        best_idx = 0
        for i, emb_res in enumerate(result["embeddings"][0]):
            dist = np.linalg.norm(embedding - emb_res)
            if dist < best_dist:
                best_dist = dist
                best_idx = i

        if best_dist <= threshold:
            return result["metadatas"][0][best_idx], best_dist
        else:
            return None, best_dist
