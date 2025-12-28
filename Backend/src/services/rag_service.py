import logging
import os
import uuid
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
import cohere
from cohere import ChatMessage

from services.qdrant_service import QdrantService, get_qdrant_service
from api.models import Citation
from utils.logging_config import get_logger, log_error_with_context
from utils.performance_monitor import timeout_handler

load_dotenv()
logger = get_logger(__name__)

class RAGService:
    def __init__(self, cohere_api_key: Optional[str] = None, qdrant_service: Optional[QdrantService] = None):
        self.cohere_api_key = cohere_api_key or os.getenv("COHERE_API_KEY")
        # Ensure Qdrant service is connected
        self.qdrant_service = qdrant_service or get_qdrant_service()

        if not self.cohere_api_key:
            raise ValueError("Cohere API key is required.")

        self.cohere_client = cohere.Client(api_key=self.cohere_api_key)

    def _format_documents_for_cohere(self, search_results: List[Any]) -> List[Dict[str, str]]:
        """
        Modified to match the Cohere RAG document format and our Qdrant payload.
        """
        formatted_docs = []
        for result in search_results:
            # Qdrant results usually have a 'payload' attribute or dict
            payload = getattr(result, 'payload', result.get('payload', {})) if not isinstance(result, dict) else result.get('payload', {})
            
            text = payload.get("text", "")
            source = payload.get("source_file", "Book Content")
            
            if text:
                # Cohere expects a list of dicts for the 'documents' parameter
                formatted_docs.append({
                    "title": os.path.basename(source),
                    "text": text
                })
        return formatted_docs

    def _create_citations_from_results(self, search_results: List[Any]) -> List[Citation]:
        citations = []
        for result in search_results:
            payload = getattr(result, 'payload', result.get('payload', {})) if not isinstance(result, dict) else result.get('payload', {})
            score = getattr(result, 'score', 0.0)
            
            text = payload.get("text", "")
            source = payload.get("source_file", "book.txt")
            
            citation = Citation(
                text=text[:300] + "..." if len(text) > 300 else text,
                source=f"Source: {os.path.basename(source)}",
                relevance_score=float(score)
            )
            citations.append(citation)
        return citations

    @timeout_handler(timeout_seconds=180)
    def query_full_book(self, query: str, top_k: int = 6, temperature: float = 0.3, max_tokens: int = 1000) -> Dict[str, Any]:
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        logger.info(f"RAG Query: {query[:100]}")

        # 1. Generate Query Embedding
        query_emb = self.cohere_client.embed(
            texts=[query],
            model="embed-english-v3.0",
            input_type="search_query"
        ).embeddings[0]

        # 2. Retrieve from Qdrant
        # Ensure this calls your qdrant_service.search which uses collection="book_chunks"
        search_results = self.qdrant_service.perform_search(
            query_vector=query_emb,
            top_k=top_k
        )

        if not search_results:
            return {
                "message": "I'm sorry, I couldn't find any relevant information in the book for your question.",
                "citations": [],
                "response_id": str(uuid.uuid4())
            }

        # 3. Prepare Documents and Citations
        documents = self._format_documents_for_cohere(search_results)
        citations = self._create_citations_from_results(search_results)

        # 4. Generate Answer with Grounding
        preamble = (
            "You are an expert AI assistant for a Physical AI Textbook. "
            "Use ONLY the provided context to answer. If the answer is not there, say you don't know. "
            "Structure your answer with bullet points if it's long."
        )

        response = self.cohere_client.chat(
            message=query,
            model="command-r-08-2024",
            documents=documents, # Passing docs here enables automatic RAG
            preamble=preamble,
            temperature=temperature,
            max_tokens=max_tokens
        )

        return {
            "message": response.text,
            "citations": citations,
            "response_id": str(uuid.uuid4())
        }

    # Helper for embedding (used during ingestion or on-the-fly)
    @timeout_handler(timeout_seconds=180)
    def embed_text(self, text: str) -> List[float]:
        response = self.cohere_client.embed(
            texts=[text],
            model="embed-english-v3.0",
            input_type="search_document"
        )
        return response.embeddings[0]

# Singleton instance
_rag_service = None

def get_rag_service() -> RAGService:
    global _rag_service
    if _rag_service is None:
        _rag_service = RAGService()
    return _rag_service