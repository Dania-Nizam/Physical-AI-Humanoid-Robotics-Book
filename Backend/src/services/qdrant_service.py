import logging
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import PointStruct, VectorParams, Distance, Filter, FieldCondition, MatchValue
import os
from dotenv import load_dotenv

from utils.logging_config import get_logger

load_dotenv()
logger = get_logger(__name__)

class QdrantService:
    def __init__(self, url: Optional[str] = None, api_key: Optional[str] = None, collection_name: str = "book_chunks"):
        self.url = url or os.getenv("QDRANT_URL")
        self.api_key = api_key or os.getenv("QDRANT_API_KEY")
        self.collection_name = collection_name

        if not self.url or not self.api_key:
            raise ValueError("Qdrant URL and API key are missing in .env file.")

        # Initialize Client
        self.client = QdrantClient(
            url=self.url,
            api_key=self.api_key,
            timeout=30 
        )

        self._verify_collection()

    def _verify_collection(self):
        try:
            collections = self.client.get_collections().collections
            exists = any(c.name == self.collection_name for c in collections)
            
            if not exists:
                logger.info(f"Creating collection: {self.collection_name}")
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
                )
            else:
                logger.info(f"Collection '{self.collection_name}' is ready.")
        except Exception as e:
            logger.error(f"Qdrant Connection Error: {e}")

    # Is function ka naam humne 'perform_search' rakha hai takay confusion na ho
    def perform_search(self, query_vector: List[float], top_k: int = 6, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        try:
            # Hum 'search' ki jagah 'query_points' use karenge jo nayi libraries ka standard hai
            results = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=top_k,
                with_payload=True
            ).points

            return [{
                "id": r.id,
                "score": r.score,
                "payload": r.payload
            } for r in results]
        except AttributeError:
            # Agar 'query_points' bhi nahi milta (purani library), toh hum 'client.search' par wapas jayenge
            # Lekin is baar hum isay directly call karenge
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=top_k,
                with_payload=True
            )
            return [{
                "id": r.id,
                "score": r.score,
                "payload": r.payload
            } for r in results]
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

    def upsert_points(self, points: List[PointStruct]) -> bool:
        try:
            self.client.upsert(collection_name=self.collection_name, points=points)
            return True
        except Exception as e:
            logger.error(f"Upsert failed: {e}")
            return False

# Singleton
_qdrant_service = None

def get_qdrant_service() -> QdrantService:
    global _qdrant_service
    if _qdrant_service is None:
        _qdrant_service = QdrantService()
    return _qdrant_service