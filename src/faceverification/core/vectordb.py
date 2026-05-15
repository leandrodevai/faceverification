"""Small ChromaDB adapter for face embedding storage and lookup."""

import uuid
from collections.abc import Mapping
from typing import Any

import chromadb
import numpy as np
from chromadb.config import Settings as ChromaSettings

from faceverification.config import settings


class VectorDB:
    """Store face embeddings and query the nearest known identity."""

    def __init__(
        self,
        distance_metric: str | None = None,
        name_collection: str | None = None,
        persist_directory: str | None = None,
    ):
        """Create an in-memory or persistent Chroma collection.

        Args:
            distance_metric: Chroma HNSW metric, such as `"l2"` or `"cosine"`.
            name_collection: Collection name for stored face embeddings.
            persist_directory: Directory for persistent storage, or `None` for memory.
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
        """Store one embedding with its metadata.

        Args:
            embedding: Face embedding vector.
            metadata: Data associated with the embedding, such as a person's name.
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
        """Return matched metadata and distance, or `None` when outside threshold.

        Args:
            embedding: Query embedding vector.
            threshold: Maximum accepted distance for a match.
            n_results: Number of nearest Chroma candidates to inspect.

        Returns:
            Matched metadata and best distance. Metadata is `None` when no
            candidate is close enough.
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
                "No record found in the vector database. Add a person before verifying faces."
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
