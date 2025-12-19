"""
Qdrant service for vector database operations.

This module provides a service layer for interacting with Qdrant vector database,
including search, upsert, and management operations for book content chunks.
"""

import logging
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import PointStruct, VectorParams, Distance, Filter, FieldCondition, MatchValue
import os
from dotenv import load_dotenv

from utils.logging_config import get_logger, log_error_with_context

# Load environment variables
load_dotenv()

# Set up logging
logger = get_logger(__name__)


class QdrantService:
    """Service class for Qdrant vector database operations."""

    def __init__(self, url: Optional[str] = None, api_key: Optional[str] = None, collection_name: str = "book_chunks"):
        """
        Initialize the Qdrant service.

        Args:
            url: Qdrant server URL (defaults to QDRANT_URL env var)
            api_key: Qdrant API key (defaults to QDRANT_API_KEY env var)
            collection_name: Name of the collection to use
        """
        self.url = url or os.getenv("QDRANT_URL")
        self.api_key = api_key or os.getenv("QDRANT_API_KEY")
        self.collection_name = collection_name

        if not self.url or not self.api_key:
            raise ValueError("Qdrant URL and API key are required. Set QDRANT_URL and QDRANT_API_KEY environment variables.")

        # Initialize Qdrant client
        self.client = QdrantClient(
            url=self.url,
            api_key=self.api_key,
            prefer_grpc=False  # Using HTTP for better compatibility
        )

        # Verify connection and collection
        self._verify_collection()

    def _verify_collection(self):
        """Verify that the collection exists, create if it doesn't."""
        try:
            self.client.get_collection(collection_name=self.collection_name)
            logger.info(f"Collection '{self.collection_name}' already exists")
        except:
            logger.info(f"Creating new collection: '{self.collection_name}'")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=1024, distance=Distance.COSINE),  # Cohere embed-english-v3.0 produces 1024-dim vectors
            )
            logger.info(f"Created collection '{self.collection_name}' with 1024-dim vectors and cosine distance")

    def search(self, query_vector: List[float], top_k: int = 6, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Search for similar vectors in the collection.

        Args:
            query_vector: The query vector to search for similar items
            top_k: Number of top results to return
            filters: Optional filters to apply to the search

        Returns:
            List of matching points with payload
        """
        # Validate inputs
        if not query_vector or len(query_vector) == 0:
            raise ValueError("Query vector cannot be empty")
        if len(query_vector) != 1024:  # Cohere embed-english-v3.0 produces 1024-dim vectors
            raise ValueError(f"Query vector must be 1024-dimensional, got {len(query_vector)} dimensions")
        if not (1 <= top_k <= 100):
            raise ValueError("top_k must be between 1 and 100")

        logger.info(f"Searching for similar vectors (top_k={top_k}) in collection '{self.collection_name}'")

        # Convert filters to Qdrant filter format if provided
        qdrant_filter = None
        if filters:
            conditions = []
            for key, value in filters.items():
                conditions.append(
                    FieldCondition(
                        key=key,
                        match=MatchValue(value=value)
                    )
                )
            if conditions:
                qdrant_filter = Filter(must=conditions)

        # Perform the search
        search_results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=top_k,
            query_filter=qdrant_filter,
            with_payload=True
        )

        # Format results
        formatted_results = []
        for result in search_results:
            formatted_results.append({
                "id": result.id,
                "score": result.score,
                "payload": result.payload
            })

        logger.info(f"Found {len(formatted_results)} results")
        return formatted_results

    def batch_search(self, query_vectors: List[List[float]], top_k: int = 6) -> List[List[Dict[str, Any]]]:
        """
        Perform batch search for multiple query vectors.

        Args:
            query_vectors: List of query vectors
            top_k: Number of top results to return for each query

        Returns:
            List of search results for each query vector
        """
        # Validate inputs
        if not query_vectors or len(query_vectors) == 0:
            raise ValueError("Query vectors list cannot be empty")
        if len(query_vectors) > 100:  # Reasonable batch size limit
            raise ValueError("Too many query vectors. Maximum 100 vectors allowed in batch")
        for i, query_vector in enumerate(query_vectors):
            if not query_vector or len(query_vector) == 0:
                raise ValueError(f"Query vector at index {i} cannot be empty")
            if len(query_vector) != 1024:  # Cohere embed-english-v3.0 produces 1024-dim vectors
                raise ValueError(f"Query vector at index {i} must be 1024-dimensional, got {len(query_vector)} dimensions")
        if not (1 <= top_k <= 100):
            raise ValueError("top_k must be between 1 and 100")

        logger.info(f"Performing batch search for {len(query_vectors)} query vectors")

        # Perform batch search
        search_results = self.client.search_batch(
            collection_name=self.collection_name,
            requests=[models.SearchRequest(
                vector=query_vector,
                limit=top_k,
                with_payload=True
            ) for query_vector in query_vectors]
        )

        # Format results
        formatted_results = []
        for result_batch in search_results:
            formatted_batch = []
            for result in result_batch:
                formatted_batch.append({
                    "id": result.id,
                    "score": result.score,
                    "payload": result.payload
                })
            formatted_results.append(formatted_batch)

        logger.info(f"Batch search completed for {len(formatted_results)} query vectors")
        return formatted_results

    def upsert_points(self, points: List[PointStruct]) -> bool:
        """
        Upsert points to the collection.

        Args:
            points: List of PointStruct objects to upsert

        Returns:
            True if successful
        """
        # Validate inputs
        if not points or len(points) == 0:
            raise ValueError("Points list cannot be empty")
        if len(points) > 1000:  # Reasonable batch size limit
            raise ValueError("Too many points. Maximum 1000 points allowed in batch")

        logger.info(f"Upserting {len(points)} points to collection '{self.collection_name}'")

        try:
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            logger.info(f"Successfully upserted {len(points)} points")
            return True
        except Exception as e:
            logger.error(f"Error upserting points: {e}")
            raise

    def get_point(self, point_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific point by ID.

        Args:
            point_id: ID of the point to retrieve

        Returns:
            Point data if found, None otherwise
        """
        # Validate inputs
        if not point_id or not point_id.strip():
            raise ValueError("Point ID cannot be empty or just whitespace")
        if len(point_id) > 100:
            raise ValueError("Point ID too long. Maximum 100 characters allowed.")

        try:
            points = self.client.retrieve(
                collection_name=self.collection_name,
                ids=[point_id]
            )
            if points:
                point = points[0]
                return {
                    "id": point.id,
                    "payload": point.payload
                }
        except Exception as e:
            logger.error(f"Error retrieving point {point_id}: {e}")
            return None

        return None

    def delete_points(self, point_ids: List[str]) -> bool:
        """
        Delete points by their IDs.

        Args:
            point_ids: List of point IDs to delete

        Returns:
            True if successful
        """
        # Validate inputs
        if not point_ids or len(point_ids) == 0:
            raise ValueError("Point IDs list cannot be empty")
        if len(point_ids) > 1000:  # Reasonable batch size limit
            raise ValueError("Too many point IDs. Maximum 1000 IDs allowed in batch")
        for i, point_id in enumerate(point_ids):
            if not point_id or not point_id.strip():
                raise ValueError(f"Point ID at index {i} cannot be empty or just whitespace")
            if len(point_id) > 100:
                raise ValueError(f"Point ID at index {i} too long. Maximum 100 characters allowed.")

        logger.info(f"Deleting {len(point_ids)} points from collection '{self.collection_name}'")

        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=point_ids
            )
            logger.info(f"Successfully deleted {len(point_ids)} points")
            return True
        except Exception as e:
            logger.error(f"Error deleting points: {e}")
            raise

    def get_collection_info(self) -> Dict[str, Any]:
        """
        Get information about the collection.

        Returns:
            Collection information
        """
        try:
            collection_info = self.client.get_collection(collection_name=self.collection_name)
            return {
                "name": collection_info.config.params.vectors.size,
                "vector_size": collection_info.config.params.vectors.size,
                "distance": collection_info.config.params.vectors.distance,
                "point_count": collection_info.points_count
            }
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            raise

    def scroll_collection(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Scroll through collection points (useful for debugging/pagination).

        Args:
            limit: Number of points to return

        Returns:
            List of points from the collection
        """
        # Validate inputs
        if not (1 <= limit <= 1000):
            raise ValueError("Limit must be between 1 and 1000")

        try:
            points, next_page = self.client.scroll(
                collection_name=self.collection_name,
                limit=limit,
                with_payload=True
            )

            formatted_points = []
            for point in points:
                formatted_points.append({
                    "id": point.id,
                    "payload": point.payload
                })

            return formatted_points
        except Exception as e:
            logger.error(f"Error scrolling collection: {e}")
            raise


# Global instance (singleton pattern)
_qdrant_service = None


def get_qdrant_service() -> QdrantService:
    """
    Get the global Qdrant service instance.

    Returns:
        QdrantService instance
    """
    global _qdrant_service
    if _qdrant_service is None:
        _qdrant_service = QdrantService()
    return _qdrant_service