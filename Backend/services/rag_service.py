"""
RAG (Retrieval-Augmented Generation) service for book content querying.

This module provides the core RAG functionality, combining vector search with
language model generation to answer questions about book content with citations.
"""

import logging
from typing import List, Dict, Any, Optional
from services.qdrant_service import QdrantService, get_qdrant_service
import cohere
from cohere import ChatMessage
import os
from dotenv import load_dotenv
from api.models import Citation
import uuid
import asyncio
import time
from contextlib import contextmanager

from utils.logging_config import get_logger, log_error_with_context
from utils.performance_monitor import timeout_handler

# Load environment variables
load_dotenv()

# Set up logging
logger = get_logger(__name__)




class RAGService:
    """Service class for RAG (Retrieval-Augmented Generation) operations."""

    def __init__(self, cohere_api_key: Optional[str] = None, qdrant_service: Optional[QdrantService] = None):
        """
        Initialize the RAG service.

        Args:
            cohere_api_key: Cohere API key (defaults to COHERE_API_KEY env var)
            qdrant_service: Qdrant service instance (defaults to global instance)
        """
        self.cohere_api_key = cohere_api_key or os.getenv("COHERE_API_KEY")
        self.qdrant_service = qdrant_service or get_qdrant_service()

        if not self.cohere_api_key:
            raise ValueError("Cohere API key is required. Set COHERE_API_KEY environment variable.")

        # Initialize Cohere client
        self.cohere_client = cohere.Client(api_key=self.cohere_api_key)

    def _format_documents_for_cohere(self, search_results: List[Dict[str, Any]]) -> List[str]:
        """
        Format retrieved documents for input to Cohere.

        Args:
            search_results: Results from Qdrant search

        Returns:
            List of formatted document strings
        """
        formatted_docs = []
        for result in search_results:
            payload = result.get("payload", {})
            text = payload.get("text", "")
            page_number = payload.get("page_number", "unknown")
            source_file = payload.get("source_file", "unknown")

            # Format the document with source information
            formatted_doc = f"[Source: {source_file}, Page: {page_number}] {text}"
            formatted_docs.append(formatted_doc)

        return formatted_docs

    def _create_citations_from_results(self, search_results: List[Dict[str, Any]]) -> List[Citation]:
        """
        Create citation objects from search results.

        Args:
            search_results: Results from Qdrant search

        Returns:
            List of Citation objects
        """
        citations = []
        for result in search_results:
            payload = result.get("payload", {})
            text = payload.get("text", "")[:200] + "..." if len(payload.get("text", "")) > 200 else payload.get("text", "")  # Truncate for citation
            page_number = payload.get("page_number", "unknown")
            source_file = payload.get("source_file", "unknown")
            score = result.get("score", 0.0)

            citation = Citation(
                text=text,
                source=f"{source_file}, page {page_number}",
                relevance_score=score
            )
            citations.append(citation)

        return citations

    @timeout_handler(timeout_seconds=30)
    def query_full_book(self, query: str, top_k: int = 6, temperature: float = 0.1, max_tokens: int = 1000) -> Dict[str, Any]:
        """
        Query the full book content using RAG approach.

        Args:
            query: The user's question
            top_k: Number of documents to retrieve from vector database
            temperature: Generation temperature (lower = more deterministic)
            max_tokens: Maximum number of tokens in the response

        Returns:
            Dictionary containing the response and citations
        """
        # Validate inputs at service level
        if not query or not query.strip():
            raise ValueError("Query cannot be empty or just whitespace")
        if len(query) > 5000:
            raise ValueError("Query too long. Maximum 5000 characters allowed.")
        if not (0.0 <= temperature <= 2.0):
            raise ValueError("Temperature must be between 0.0 and 2.0")
        if not (1 <= max_tokens <= 4000):
            raise ValueError("Max tokens must be between 1 and 4000")
        if not (1 <= top_k <= 20):
            raise ValueError("top_k must be between 1 and 20")

        logger.info(f"Processing full-book query: '{query[:50]}...' (top_k={top_k})")

        # Generate embedding for the query
        try:
            query_embedding_response = self.cohere_client.embed(
                texts=[query],
                model="embed-english-v3.0",
                input_type="search_query"
            )
            query_embedding = query_embedding_response.embeddings[0]
        except Exception as e:
            logger.error(f"Error generating query embedding: {e}")
            log_error_with_context(
                logger_instance=logger,
                error_msg=str(e),
                context={
                    "query_length": len(query),
                    "top_k": top_k,
                    "operation": "embedding_generation"
                },
                error_type="rag_service_error"
            )
            raise

        # Search in Qdrant for relevant documents
        try:
            search_results = self.qdrant_service.search(
                query_vector=query_embedding,
                top_k=top_k
            )
        except Exception as e:
            logger.error(f"Error searching in Qdrant: {e}")
            log_error_with_context(
                logger_instance=logger,
                error_msg=str(e),
                context={
                    "query_length": len(query),
                    "top_k": top_k,
                    "operation": "qdrant_search"
                },
                error_type="rag_service_error"
            )
            raise

        # Format documents for Cohere
        if search_results:
            formatted_docs = self._format_documents_for_cohere(search_results)
            logger.info(f"Retrieved {len(formatted_docs)} relevant documents")
        else:
            logger.warning("No relevant documents found for the query")
            formatted_docs = []

        # Create citations
        citations = self._create_citations_from_results(search_results)

        # Prepare the chat request to Cohere with strong preamble for zero hallucinations
        preamble = (
            "You are a helpful assistant that answers questions based ONLY on the provided source documents. "
            "Do not make up information. If the answer is not in the provided documents, say 'I cannot find this information in the provided documents.' "
            "Always cite the source documents when providing answers. "
            "Be accurate, concise, and helpful."
        )

        # Create chat history with documents as context
        message = f"Question: {query}\n\nContext documents:\n"
        for i, doc in enumerate(formatted_docs, 1):
            message += f"Document {i}: {doc}\n\n"

        message += "Based on the above documents, please answer the question. Provide specific citations to the source documents."

        try:
            # Make the chat request to Cohere
            response = self.cohere_client.chat(
                message=message,
                model="command-r-plus",  # Using a strong model for accuracy
                temperature=temperature,
                max_tokens=max_tokens,
                preamble=preamble,
                documents=formatted_docs if formatted_docs else None  # Pass documents as context if available
            )

            # Extract the response text
            response_text = response.text

            # Generate a unique response ID
            response_id = str(uuid.uuid4())

            logger.info(f"Successfully generated response with {len(citations)} citations")

            return {
                "message": response_text,
                "citations": citations,
                "response_id": response_id
            }

        except Exception as e:
            logger.error(f"Error generating response from Cohere: {e}")
            log_error_with_context(
                logger_instance=logger,
                error_msg=str(e),
                context={
                    "query_length": len(query),
                    "top_k": top_k,
                    "operation": "cohere_generation"
                },
                error_type="rag_service_error"
            )
            raise

    def _split_long_text(self, text: str, max_chunk_size: int = 1000) -> List[str]:
        """
        Split long text into smaller chunks if needed.

        Args:
            text: The text to split
            max_chunk_size: Maximum size of each chunk

        Returns:
            List of text chunks
        """
        if len(text) <= max_chunk_size:
            return [text]

        # Split the text into chunks of approximately max_chunk_size
        chunks = []
        for i in range(0, len(text), max_chunk_size):
            chunk = text[i:i + max_chunk_size]
            chunks.append(chunk)

        return chunks

    @timeout_handler(timeout_seconds=30)
    def query_selected_text(self, query: str, selected_text: str, temperature: float = 0.1, max_tokens: int = 1000) -> Dict[str, Any]:
        """
        Query based on selected text only (no vector search).
        This ensures complete isolation from the full knowledge base (T046).

        Args:
            query: The user's question
            selected_text: The text the user has selected/highlighted
            temperature: Generation temperature (lower = more deterministic)
            max_tokens: Maximum number of tokens in the response

        Returns:
            Dictionary containing the response and citations
        """
        # Validate inputs at service level
        if not query or not query.strip():
            raise ValueError("Query cannot be empty or just whitespace")
        if len(query) > 5000:
            raise ValueError("Query too long. Maximum 5000 characters allowed.")
        if not selected_text or not selected_text.strip():
            raise ValueError("Selected text cannot be empty or just whitespace")
        if len(selected_text) > 10000:
            raise ValueError("Selected text too long. Maximum 10000 characters allowed.")
        if not (0.0 <= temperature <= 2.0):
            raise ValueError("Temperature must be between 0.0 and 2.0")
        if not (1 <= max_tokens <= 4000):
            raise ValueError("Max tokens must be between 1 and 4000")

        logger.info(f"Processing selected-text query: '{query[:50]}...' on selected text of {len(selected_text)} characters")

        # Split long selected text into chunks if needed (T045)
        text_chunks = self._split_long_text(selected_text, max_chunk_size=1000)

        # Create documents from the text chunks (T044 - direct pass of selected_text as documents)
        formatted_docs = text_chunks

        # Create citations for the selected text chunks
        citations = []
        for i, chunk in enumerate(text_chunks):
            citations.append(Citation(
                text=chunk[:200] + "..." if len(chunk) > 200 else chunk,
                source=f"user-selected text chunk {i+1}",
                relevance_score=1.0
            ))

        # Prepare the chat request to Cohere with strong preamble for zero hallucinations and grounding validation (T049)
        preamble = (
            "You are a helpful assistant that answers questions based ONLY on the provided selected text. "
            "Do not make up information. If the answer is not in the provided text, say 'I cannot find this information in the provided text.' "
            "Do not use any external knowledge or general world knowledge. "
            "Be accurate, concise, and helpful. Ensure all responses are strictly grounded in the provided text."
        )

        # Create the message with the query and selected text
        message = f"Question: {query}\n\nSelected text: {selected_text}\n\nBased on the above selected text, please answer the question."

        try:
            # Make the chat request to Cohere with selected text as documents (T044)
            # This ensures the model only uses the provided selected text, not the full knowledge base (T046)
            response = self.cohere_client.chat(
                message=message,
                model="command-r-plus",  # Using a strong model for accuracy
                temperature=temperature,
                max_tokens=max_tokens,
                preamble=preamble,
                documents=formatted_docs if formatted_docs else None  # Pass the selected text chunks as context (T044)
            )

            # Extract the response text
            response_text = response.text

            # Generate a unique response ID
            response_id = str(uuid.uuid4())

            logger.info("Successfully generated response for selected-text query with proper grounding")

            return {
                "message": response_text,
                "citations": citations,
                "response_id": response_id
            }

        except Exception as e:
            logger.error(f"Error generating response from Cohere: {e}")
            log_error_with_context(
                logger_instance=logger,
                error_msg=str(e),
                context={
                    "query_length": len(query),
                    "selected_text_length": len(selected_text),
                    "operation": "cohere_generation_selected_text"
                },
                error_type="rag_service_error"
            )
            raise

    @timeout_handler(timeout_seconds=10)
    def embed_text(self, text: str) -> List[float]:
        """
        Create an embedding for a given text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        # Validate input at service level
        if not text or not text.strip():
            raise ValueError("Text cannot be empty or just whitespace")
        if len(text) > 100000:  # Set a reasonable limit for embedding
            raise ValueError("Text too long for embedding. Maximum 100000 characters allowed.")

        try:
            response = self.cohere_client.embed(
                texts=[text],
                model="embed-english-v3.0",
                input_type="search_document"
            )
            return response.embeddings[0]
        except Exception as e:
            logger.error(f"Error creating embedding: {e}")
            raise


# Global instance (singleton pattern)
_rag_service = None


def get_rag_service() -> RAGService:
    """
    Get the global RAG service instance.

    Returns:
        RAGService instance
    """
    global _rag_service
    if _rag_service is None:
        _rag_service = RAGService()
    return _rag_service