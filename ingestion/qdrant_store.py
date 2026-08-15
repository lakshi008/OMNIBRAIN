"""
Qdrant vector store integration for the OmniBrain ingestion pipeline.

Provides vector collection management, batch vector upserting, and similarity search
with full citation lineage metadata preservation.
"""

from __future__ import annotations

import math
from typing import Any

from qdrant_client import QdrantClient, models

from ingestion.models import (
    EmbeddingGenerationResult,
    EmbeddingVectorRecord,
    VectorSearchResult,
)
from ingestion.qdrant_config import QdrantConfig


class QdrantVectorStore:
    """Vector store abstraction wrapping the official Qdrant Python client."""

    def __init__(
        self,
        config: QdrantConfig | None = None,
        client: QdrantClient | None = None,
    ) -> None:
        """Initialize QdrantVectorStore.

        Args:
            config: Optional QdrantConfig. If None, loaded from environment.
            client: Optional pre-configured QdrantClient instance (useful for in-memory testing).
        """
        self.config = config or QdrantConfig.from_env()

        if client is not None:
            self.client = client
        elif self.config.url == ":memory:":
            self.client = QdrantClient(location=":memory:")
        else:
            self.client = QdrantClient(
                url=self.config.url,
                api_key=self.config.api_key,
                timeout=self.config.timeout,
            )

    def create_collection(self, collection_name: str, vector_dimension: int) -> None:
        """Create a new collection in Qdrant with Cosine distance metric.

        Args:
            collection_name: Unique name for the collection.
            vector_dimension: Dimensionality of the vectors (must be > 0).

        Raises:
            ValueError: If collection_name is empty or vector_dimension <= 0.
        """
        if not collection_name or not isinstance(collection_name, str) or not collection_name.strip():
            raise ValueError("collection_name must be a non-empty string.")
        if not isinstance(vector_dimension, int) or isinstance(vector_dimension, bool) or vector_dimension <= 0:
            raise ValueError(f"vector_dimension must be a positive integer > 0, got {vector_dimension!r}.")

        self.client.create_collection(
            collection_name=collection_name.strip(),
            vectors_config=models.VectorParams(
                size=vector_dimension,
                distance=models.Distance.COSINE,
            ),
        )

    def collection_exists(self, collection_name: str) -> bool:
        """Check whether a collection exists in Qdrant.

        Args:
            collection_name: Name of the collection to check.

        Returns:
            True if the collection exists, False otherwise.
        """
        if not collection_name or not isinstance(collection_name, str):
            return False
        return self.client.collection_exists(collection_name=collection_name.strip())

    def delete_collection(self, collection_name: str) -> None:
        """Delete an existing collection from Qdrant.

        Args:
            collection_name: Name of the collection to delete.
        """
        if self.collection_exists(collection_name):
            self.client.delete_collection(collection_name=collection_name.strip())

    def get_collection_info(self, collection_name: str) -> dict[str, Any]:
        """Retrieve metadata, status, and vector counts for a collection.

        Args:
            collection_name: Name of the collection.

        Returns:
            Dictionary with collection status, points_count, vector_dimension, distance.

        Raises:
            ValueError: If the collection does not exist.
        """
        if not self.collection_exists(collection_name):
            raise ValueError(f"Collection '{collection_name}' does not exist.")

        col_info = self.client.get_collection(collection_name=collection_name.strip())

        vectors_config = col_info.config.params.vectors
        dimension = getattr(vectors_config, "size", None)
        distance = getattr(vectors_config, "distance", None)
        if distance is not None and hasattr(distance, "value"):
            distance = distance.value
        elif distance is not None:
            distance = str(distance)

        return {
            "collection_name": collection_name.strip(),
            "status": str(col_info.status.value) if hasattr(col_info.status, "value") else str(col_info.status),
            "points_count": getattr(col_info, "points_count", 0) or 0,
            "indexed_vectors_count": getattr(col_info, "indexed_vectors_count", 0) or 0,
            "vector_dimension": dimension,
            "distance": distance,
        }

    def upsert_embeddings(
        self,
        collection_name: str,
        result: EmbeddingGenerationResult,
    ) -> int:
        """Upsert generated embedding vector records into the specified Qdrant collection.

        Preserves full citation lineage in the payload:
        - chunk_id, document_id, filename, page_number, chunk_index, content_type, metadata

        Args:
            collection_name: Target collection name.
            result: EmbeddingGenerationResult containing EmbeddingVectorRecord items.

        Returns:
            Number of points successfully upserted.

        Raises:
            ValueError: If collection does not exist, or if vector dimensions do not match.
            TypeError: If result is invalid.
        """
        if not isinstance(result, EmbeddingGenerationResult):
            raise TypeError(
                f"Expected EmbeddingGenerationResult, got {type(result).__name__}."
            )

        col_name = collection_name.strip()
        if not self.collection_exists(col_name):
            raise ValueError(f"Collection '{col_name}' does not exist.")

        if not result.items:
            return 0

        # Verify collection dimension matches
        col_info = self.get_collection_info(col_name)
        col_dim = col_info["vector_dimension"]

        if result.dimension != col_dim:
            raise ValueError(
                f"Dimension mismatch: collection '{col_name}' expects {col_dim}, "
                f"but EmbeddingGenerationResult has {result.dimension}."
            )

        points: list[models.PointStruct] = []
        for idx, rec in enumerate(result.items):
            if not isinstance(rec, EmbeddingVectorRecord):
                raise TypeError(f"Item at index {idx} is not an EmbeddingVectorRecord.")

            if len(rec.vector) != col_dim:
                raise ValueError(
                    f"Vector for chunk '{rec.chunk_id}' has dimension {len(rec.vector)}, expected {col_dim}."
                )

            for v in rec.vector:
                if not isinstance(v, (int, float)) or isinstance(v, bool) or math.isnan(v) or math.isinf(v):
                    raise ValueError(f"Vector for chunk '{rec.chunk_id}' contains invalid float value: {v!r}")

            # Prepare payload preserving all citation lineage
            payload: dict[str, Any] = {
                "chunk_id": rec.chunk_id,
                "document_id": rec.document_id,
                "filename": rec.filename,
                "page_number": rec.page_number,
                "chunk_index": rec.chunk_index,
                "content_type": rec.content_type,
                "metadata": dict(rec.metadata),
            }

            points.append(
                models.PointStruct(
                    id=rec.chunk_id,
                    vector=rec.vector,
                    payload=payload,
                )
            )

        self.client.upsert(
            collection_name=col_name,
            points=points,
        )

        return len(points)

    def search(
        self,
        collection_name: str,
        query_vector: list[float],
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """Perform cosine similarity search on the specified Qdrant collection.

        Args:
            collection_name: Target collection name.
            query_vector: Query dense vector (must match collection dimension).
            limit: Maximum number of nearest results to return (ordered by score descending).

        Returns:
            List of dictionaries containing score and complete citation metadata.

        Raises:
            ValueError: If collection does not exist, query vector is empty, or dimension mismatches.
        """
        col_name = collection_name.strip()
        if not self.collection_exists(col_name):
            raise ValueError(f"Collection '{col_name}' does not exist.")

        if not query_vector or not isinstance(query_vector, list):
            raise ValueError("query_vector must be a non-empty list of floats.")

        col_info = self.get_collection_info(col_name)
        col_dim = col_info["vector_dimension"]

        if len(query_vector) != col_dim:
            raise ValueError(
                f"Query vector dimension {len(query_vector)} does not match collection dimension {col_dim}."
            )

        if limit <= 0:
            return []

        # Perform query using query_points API
        query_res = self.client.query_points(
            collection_name=col_name,
            query=query_vector,
            limit=limit,
            with_payload=True,
        )

        results: list[dict[str, Any]] = []
        for point in query_res.points:
            payload = point.payload or {}
            results.append(
                {
                    "chunk_id": str(point.id),
                    "score": float(point.score) if point.score is not None else 0.0,
                    "document_id": payload.get("document_id", ""),
                    "filename": payload.get("filename", ""),
                    "page_number": payload.get("page_number"),
                    "chunk_index": payload.get("chunk_index", 0),
                    "content_type": payload.get("content_type", "text"),
                    "metadata": payload.get("metadata", {}),
                }
            )

        return results

    def search_records(
        self,
        collection_name: str,
        query_vector: list[float],
        limit: int = 5,
    ) -> list[VectorSearchResult]:
        """Perform similarity search and return structured VectorSearchResult dataclasses.

        Args:
            collection_name: Target collection name.
            query_vector: Query dense vector.
            limit: Maximum number of nearest results.

        Returns:
            List of VectorSearchResult objects ordered by score descending.
        """
        raw_results = self.search(collection_name=collection_name, query_vector=query_vector, limit=limit)
        return [
            VectorSearchResult(
                chunk_id=r["chunk_id"],
                score=r["score"],
                document_id=r["document_id"],
                filename=r["filename"],
                page_number=r["page_number"],
                chunk_index=r["chunk_index"],
                content_type=r["content_type"],
                content=r.get("metadata", {}).get("content", ""),
                metadata=r["metadata"],
            )
            for r in raw_results
        ]
