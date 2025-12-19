import pytest
from fastapi.testclient import TestClient
from api.main import app
from api.models import ChatFullRequest, ChatResponse
import json
from unittest.mock import Mock, patch


@pytest.fixture
def client():
    """Create a test client for the API"""
    return TestClient(app)


class TestChatFullContract:
    """Contract tests for the chat/full endpoint"""

    def test_chat_full_endpoint_exists(self, client):
        """Test that the /chat/full endpoint exists and accepts POST requests"""
        # This would test that the endpoint exists and returns expected status codes
        # For now, we'll test the contract by making sure it's defined properly
        assert hasattr(app.router, 'routes')
        routes = [route for route in app.router.routes if route.path == '/chat/full']
        assert len(routes) > 0  # The route should exist
        # Check if it accepts POST method
        post_methods = [method for route in routes for method in route.methods if method == 'POST']
        assert len(post_methods) > 0

    def test_chat_full_request_model(self):
        """Test that the request model has the expected fields"""
        # Test the ChatFullRequest model structure
        request_data = {
            "query": "What is the main topic of this book?",
            "session_id": "test-session-123",
            "temperature": 0.1,
            "max_tokens": 1000
        }

        # Create a request object
        chat_request = ChatFullRequest(**request_data)

        # Validate the fields
        assert chat_request.query == "What is the main topic of this book?"
        assert chat_request.session_id == "test-session-123"
        assert chat_request.temperature == 0.1
        assert chat_request.max_tokens == 1000

    def test_chat_full_response_model(self):
        """Test that the response model has the expected fields"""
        # Test the ChatResponse model structure
        response_data = {
            "message": "The main topic of this book is about RAG systems.",
            "citations": [
                {
                    "text": "RAG systems combine retrieval and generation",
                    "source": "page_12",
                    "relevance_score": 0.95
                }
            ],
            "session_id": "test-session-123",
            "response_id": "response-123"
        }

        # Create a response object
        chat_response = ChatResponse(**response_data)

        # Validate the fields
        assert chat_response.message == "The main topic of this book is about RAG systems."
        assert len(chat_response.citations) == 1
        assert chat_response.session_id == "test-session-123"
        assert chat_response.response_id == "response-123"

    def test_chat_full_response_structure(self, client):
        """Test that the response structure matches the contract"""
        # This would make an actual request to the endpoint if it existed
        # For contract testing, we're validating the expected response structure
        pass


class TestFullBookRetrievalIntegration:
    """Integration tests for full-book retrieval"""

    def test_full_book_retrieval_with_cohere(self, mocker):
        """Test integration between Cohere and Qdrant for full-book retrieval"""
        # Mock Cohere client
        mock_cohere_client = mocker.Mock()
        mock_cohere_response = mocker.Mock()
        mock_cohere_response.text = "The main topic of this book is RAG systems."
        mock_cohere_client.chat.return_value = mock_cohere_response

        # Mock Qdrant client
        mock_qdrant_client = mocker.Mock()
        mock_search_result = [
            mocker.Mock(payload={
                "text": "RAG systems combine retrieval and generation",
                "page_number": 12,
                "chunk_index": 0,
                "source_file": "test_book.pdf"
            })
        ]
        mock_qdrant_client.search.return_value = mock_search_result

        # Test the integration between services
        # This would test that the RAG service properly connects Cohere and Qdrant
        pass

    def test_full_book_qdrant_search(self, mocker):
        """Test Qdrant search functionality for full-book queries"""
        # Mock Qdrant client
        mock_qdrant_client = mocker.Mock()
        mock_search_result = [
            mocker.Mock(payload={
                "text": "Sample relevant text",
                "page_number": 1,
                "chunk_index": 0,
                "source_file": "test_book.pdf",
                "text_length": 100
            })
        ]
        mock_qdrant_client.search.return_value = mock_search_result

        # Test that search returns results with proper metadata
        # This would test the Qdrant service functionality
        pass

    def test_full_book_retrieval_end_to_end(self, client, mocker):
        """Test end-to-end full-book retrieval flow"""
        # This would mock the entire flow from API request to response
        # with mocked Cohere and Qdrant services
        pass


class TestCitationAccuracy:
    """Unit tests for citation accuracy"""

    def test_citation_format(self):
        """Test that citations are properly formatted"""
        # Test that citations follow the expected format
        citation_data = {
            "text": "Sample citation text",
            "source": "page_15",
            "relevance_score": 0.85
        }

        # Would validate against the Citation model
        from api.models import Citation
        citation = Citation(**citation_data)

        assert citation.text == "Sample citation text"
        assert citation.source == "page_15"
        assert citation.relevance_score == 0.85

    def test_citation_source_accuracy(self):
        """Test that citations point to correct source locations"""
        # This would test that the citation source accurately reflects
        # where the information was retrieved from
        pass

    def test_citation_relevance_scoring(self):
        """Test that citation relevance scores are properly calculated"""
        # This would test the logic for calculating relevance scores
        # for retrieved documents
        pass

    def test_citation_text_matching(self):
        """Test that citation text matches the source content"""
        # This would test that the citation text actually comes from
        # the cited source
        pass


class TestChatSelectedContract:
    """Contract tests for the chat/selected endpoint"""

    def test_chat_selected_endpoint_exists(self, client):
        """Test that the /chat/selected endpoint exists and accepts POST requests"""
        # This would test that the endpoint exists and returns expected status codes
        assert hasattr(app.router, 'routes')
        routes = [route for route in app.router.routes if route.path == '/chat/selected']
        assert len(routes) > 0  # The route should exist
        # Check if it accepts POST method
        post_methods = [method for route in routes for method in route.methods if method == 'POST']
        assert len(post_methods) > 0

    def test_chat_selected_request_model(self):
        """Test that the selected text request model has the expected fields"""
        from api.models import ChatSelectedRequest

        # Test the ChatSelectedRequest model structure
        request_data = {
            "query": "What does this selected text mean?",
            "selected_text": "This is the text the user has selected.",
            "session_id": "test-session-123",
            "temperature": 0.1,
            "max_tokens": 1000
        }

        # Create a request object
        chat_request = ChatSelectedRequest(**request_data)

        # Validate the fields
        assert chat_request.query == "What does this selected text mean?"
        assert chat_request.selected_text == "This is the text the user has selected."
        assert chat_request.session_id == "test-session-123"
        assert chat_request.temperature == 0.1
        assert chat_request.max_tokens == 1000

    def test_chat_selected_response_model(self):
        """Test that the response model for selected text has the expected fields"""
        # Already tested in the base ChatResponse model test
        pass


class TestSelectedTextGroundingIntegration:
    """Integration tests for selected-text grounding"""

    def test_selected_text_grounding_with_cohere(self, mocker):
        """Test that the selected text is properly passed to Cohere without Qdrant search"""
        # Mock the RAG service to ensure it uses only the selected text
        from services.rag_service import RAGService

        # Create a mock rag service instance
        mock_cohere_client = mocker.Mock()
        mock_cohere_response = mocker.Mock()
        mock_cohere_response.text = "Based on the selected text, this means X."
        mock_cohere_client.chat.return_value = mock_cohere_response

        # Mock Qdrant service to ensure it's not called for selected text queries
        mock_qdrant_service = mocker.Mock()

        rag_service = RAGService(cohere_api_key="test-key")
        rag_service.cohere_client = mock_cohere_client

        # Test the query_selected_text method directly
        result = rag_service.query_selected_text(
            query="What does this mean?",
            selected_text="This is the selected text that should be the only source.",
            temperature=0.1,
            max_tokens=1000
        )

        # Verify that Cohere was called with the selected text as context
        assert "selected text that should be the only source" in result["message"]
        # Verify that no Qdrant search was performed
        # This will be validated by ensuring the method properly isolates the context

    def test_selected_text_vs_full_book_isolation(self, mocker):
        """Test that selected text responses are isolated from full book content"""
        # This would test that when using /chat/selected, the response only uses
        # the provided selected text and not the broader book content in Qdrant
        pass

    def test_selected_text_endpoint_integration(self, client, mocker):
        """Test the full integration of the selected text endpoint"""
        # Mock external services to test the endpoint flow
        pass


class TestKnowledgeBaseIsolation:
    """Unit tests for knowledge base isolation in selected text mode"""

    def test_rag_service_uses_only_selected_text(self, mocker):
        """Test that RAG service uses only selected text, not Qdrant search, for selected text queries"""
        from services.rag_service import RAGService

        # Mock Cohere client
        mock_cohere_client = mocker.Mock()
        mock_response = mocker.Mock()
        mock_response.text = "Response based on selected text only"
        mock_cohere_client.chat.return_value = mock_response

        # Mock Qdrant service to verify it's not called
        mock_qdrant_service = mocker.Mock()
        mock_qdrant_service.search = mocker.Mock()

        # Create RAG service instance with mocked clients
        rag_service = RAGService.__new__(RAGService)  # Create without calling __init__
        rag_service.cohere_client = mock_cohere_client
        rag_service.qdrant_service = mock_qdrant_service

        # Call the selected text query method
        rag_service.query_selected_text(
            query="What does this mean?",
            selected_text="Selected text content here",
            temperature=0.1,
            max_tokens=1000
        )

        # Verify that Qdrant search was NOT called (knowledge base isolation)
        assert not mock_qdrant_service.search.called

    def test_selected_text_does_not_access_vector_store(self, mocker):
        """Test that selected text mode does not access the vector store at all"""
        # This would ensure complete isolation from the ingested book content
        pass

    def test_full_query_still_uses_vector_store(self, mocker):
        """Test that full-book queries still use the vector store (regression test)"""
        from services.rag_service import RAGService

        # Mock Cohere client
        mock_cohere_client = mocker.Mock()
        mock_response = mocker.Mock()
        mock_cohere_client.embed.return_value = mocker.Mock(embeddings=[[0.1, 0.2, 0.3]])
        mock_cohere_client.chat.return_value = mock_response

        # Mock Qdrant service to verify it IS called for full queries
        mock_qdrant_service = mocker.Mock()
        mock_qdrant_service.search = mocker.Mock(return_value=[])

        # Create RAG service instance with mocked clients
        rag_service = RAGService.__new__(RAGService)  # Create without calling __init__
        rag_service.cohere_client = mock_cohere_client
        rag_service.qdrant_service = mock_qdrant_service

        # Call the full book query method
        rag_service.query_full_book(
            query="What is the book about?",
            top_k=6,
            temperature=0.1,
            max_tokens=1000
        )

        # Verify that Qdrant search WAS called (full queries should access vector store)
        assert mock_qdrant_service.search.called


class TestEdgeCases:
    """Unit tests for edge cases and error conditions"""

    def test_empty_query_validation(self, client):
        """Test validation for empty query"""
        # Test chat/full endpoint
        response = client.post("/chat/full", json={
            "query": "",
            "session_id": "test-session-123",
            "temperature": 0.1,
            "max_tokens": 1000
        })
        assert response.status_code == 422  # Validation error

        # Test chat/selected endpoint
        response = client.post("/chat/selected", json={
            "query": "",
            "selected_text": "Some selected text",
            "session_id": "test-session-123",
            "temperature": 0.1,
            "max_tokens": 1000
        })
        assert response.status_code == 422  # Validation error

    def test_whitespace_only_query_validation(self, client):
        """Test validation for whitespace-only query"""
        # Test chat/full endpoint
        response = client.post("/chat/full", json={
            "query": "   ",
            "session_id": "test-session-123",
            "temperature": 0.1,
            "max_tokens": 1000
        })
        assert response.status_code == 422  # Validation error

        # Test chat/selected endpoint
        response = client.post("/chat/selected", json={
            "query": "\t\n",
            "selected_text": "Some selected text",
            "session_id": "test-session-123",
            "temperature": 0.1,
            "max_tokens": 1000
        })
        assert response.status_code == 422  # Validation error

    def test_empty_selected_text_validation(self, client):
        """Test validation for empty selected text"""
        response = client.post("/chat/selected", json={
            "query": "What does this mean?",
            "selected_text": "",
            "session_id": "test-session-123",
            "temperature": 0.1,
            "max_tokens": 1000
        })
        assert response.status_code == 422  # Validation error

    def test_whitespace_only_selected_text_validation(self, client):
        """Test validation for whitespace-only selected text"""
        response = client.post("/chat/selected", json={
            "query": "What does this mean?",
            "selected_text": "   ",
            "session_id": "test-session-123",
            "temperature": 0.1,
            "max_tokens": 1000
        })
        assert response.status_code == 422  # Validation error

    def test_query_too_long_validation(self, client):
        """Test validation for query that exceeds maximum length"""
        long_query = "A" * 6000  # Exceeds 5000 max length

        # Test chat/full endpoint
        response = client.post("/chat/full", json={
            "query": long_query,
            "session_id": "test-session-123",
            "temperature": 0.1,
            "max_tokens": 1000
        })
        assert response.status_code == 422  # Validation error

        # Test chat/selected endpoint
        response = client.post("/chat/selected", json={
            "query": long_query,
            "selected_text": "Some selected text",
            "session_id": "test-session-123",
            "temperature": 0.1,
            "max_tokens": 1000
        })
        assert response.status_code == 422  # Validation error

    def test_selected_text_too_long_validation(self, client):
        """Test validation for selected text that exceeds maximum length"""
        long_selected_text = "A" * 15000  # Exceeds 10000 max length

        response = client.post("/chat/selected", json={
            "query": "What does this mean?",
            "selected_text": long_selected_text,
            "session_id": "test-session-123",
            "temperature": 0.1,
            "max_tokens": 1000
        })
        assert response.status_code == 422  # Validation error

    def test_invalid_temperature_validation(self, client):
        """Test validation for temperature outside valid range"""
        # Test temperature too low
        response = client.post("/chat/full", json={
            "query": "What is the book about?",
            "session_id": "test-session-123",
            "temperature": -0.5,  # Below 0.0
            "max_tokens": 1000
        })
        assert response.status_code == 422  # Validation error

        # Test temperature too high
        response = client.post("/chat/full", json={
            "query": "What is the book about?",
            "session_id": "test-session-123",
            "temperature": 3.0,  # Above 2.0
            "max_tokens": 1000
        })
        assert response.status_code == 422  # Validation error

    def test_invalid_max_tokens_validation(self, client):
        """Test validation for max_tokens outside valid range"""
        # Test max_tokens too low
        response = client.post("/chat/full", json={
            "query": "What is the book about?",
            "session_id": "test-session-123",
            "temperature": 0.1,
            "max_tokens": 0  # Below 1
        })
        assert response.status_code == 422  # Validation error

        # Test max_tokens too high
        response = client.post("/chat/full", json={
            "query": "What is the book about?",
            "session_id": "test-session-123",
            "temperature": 0.1,
            "max_tokens": 5000  # Above 4000
        })
        assert response.status_code == 422  # Validation error

    def test_invalid_session_id_format(self, client):
        """Test validation for session_id with invalid characters"""
        # Test session_id with special characters
        response = client.post("/chat/full", json={
            "query": "What is the book about?",
            "session_id": "test@session",  # Contains @ which is invalid
            "temperature": 0.1,
            "max_tokens": 1000
        })
        assert response.status_code == 422  # Validation error

        # Test session_id that's too long
        long_session_id = "A" * 150  # Exceeds 100 max length
        response = client.post("/chat/full", json={
            "query": "What is the book about?",
            "session_id": long_session_id,
            "temperature": 0.1,
            "max_tokens": 1000
        })
        assert response.status_code == 422  # Validation error

    def test_invalid_session_id_in_path(self, client):
        """Test validation for invalid session_id in path parameter"""
        # Test session history endpoint with invalid session_id
        response = client.get("/sessions/invalid@session/history")
        assert response.status_code == 400  # Bad request due to validation

        # Test with too long session_id
        long_session_id = "A" * 150
        response = client.get(f"/sessions/{long_session_id}/history")
        assert response.status_code == 400  # Bad request due to validation

    def test_missing_required_fields(self, client):
        """Test validation for missing required fields"""
        # Test chat/full without query
        response = client.post("/chat/full", json={
            "session_id": "test-session-123",
            "temperature": 0.1,
            "max_tokens": 1000
        })
        assert response.status_code == 422  # Validation error

        # Test chat/selected without query
        response = client.post("/chat/selected", json={
            "selected_text": "Some selected text",
            "session_id": "test-session-123",
            "temperature": 0.1,
            "max_tokens": 1000
        })
        assert response.status_code == 422  # Validation error

        # Test chat/selected without selected_text
        response = client.post("/chat/selected", json={
            "query": "What does this mean?",
            "session_id": "test-session-123",
            "temperature": 0.1,
            "max_tokens": 1000
        })
        assert response.status_code == 422  # Validation error

    def test_rag_service_input_validation_direct_calls(self):
        """Test RAG service input validation for direct method calls"""
        from services.rag_service import RAGService
        import pytest

        # Create a mock RAG service instance
        rag_service = RAGService.__new__(RAGService)

        # Test query_full_book with empty query
        with pytest.raises(ValueError, match="Query cannot be empty"):
            rag_service.query_full_book("", top_k=6, temperature=0.1, max_tokens=1000)

        # Test query_full_book with query too long
        with pytest.raises(ValueError, match="Query too long"):
            rag_service.query_full_book("A" * 6000, top_k=6, temperature=0.1, max_tokens=1000)

        # Test query_full_book with invalid temperature
        with pytest.raises(ValueError, match="Temperature must be between"):
            rag_service.query_full_book("Test query", top_k=6, temperature=-1.0, max_tokens=1000)

        # Test query_full_book with invalid max_tokens
        with pytest.raises(ValueError, match="Max tokens must be between"):
            rag_service.query_full_book("Test query", top_k=6, temperature=0.1, max_tokens=0)

        # Test query_full_book with invalid top_k
        with pytest.raises(ValueError, match="top_k must be between"):
            rag_service.query_full_book("Test query", top_k=0, temperature=0.1, max_tokens=1000)

        # Test query_selected_text with empty query
        with pytest.raises(ValueError, match="Query cannot be empty"):
            rag_service.query_selected_text("", "selected text", temperature=0.1, max_tokens=1000)

        # Test query_selected_text with empty selected_text
        with pytest.raises(ValueError, match="Selected text cannot be empty"):
            rag_service.query_selected_text("Test query", "", temperature=0.1, max_tokens=1000)

        # Test query_selected_text with query too long
        with pytest.raises(ValueError, match="Query too long"):
            rag_service.query_selected_text("A" * 6000, "selected text", temperature=0.1, max_tokens=1000)

        # Test query_selected_text with selected_text too long
        with pytest.raises(ValueError, match="Selected text too long"):
            rag_service.query_selected_text("Test query", "A" * 15000, temperature=0.1, max_tokens=1000)

    def test_qdrant_service_input_validation(self):
        """Test Qdrant service input validation"""
        from services.qdrant_service import QdrantService
        import pytest

        # Create a mock Qdrant service instance
        qdrant_service = QdrantService.__new__(QdrantService)

        # Test search with empty query vector
        with pytest.raises(ValueError, match="Query vector cannot be empty"):
            qdrant_service.search([], top_k=6)

        # Test search with wrong dimension vector
        with pytest.raises(ValueError, match="must be 1024-dimensional"):
            qdrant_service.search([0.1, 0.2], top_k=6)  # Too short

        # Test search with invalid top_k
        with pytest.raises(ValueError, match="top_k must be between"):
            qdrant_service.search([0.1] * 1024, top_k=0)

        # Test batch_search with empty list
        with pytest.raises(ValueError, match="Query vectors list cannot be empty"):
            qdrant_service.batch_search([])

        # Test batch_search with too many vectors
        with pytest.raises(ValueError, match="Too many query vectors"):
            qdrant_service.batch_search([[0.1] * 1024 for _ in range(150)])

        # Test upsert_points with empty list
        with pytest.raises(ValueError, match="Points list cannot be empty"):
            qdrant_service.upsert_points([])

        # Test get_point with empty ID
        with pytest.raises(ValueError, match="Point ID cannot be empty"):
            qdrant_service.get_point("")

        # Test delete_points with empty list
        with pytest.raises(ValueError, match="Point IDs list cannot be empty"):
            qdrant_service.delete_points([])

        # Test scroll_collection with invalid limit
        with pytest.raises(ValueError, match="Limit must be between"):
            qdrant_service.scroll_collection(limit=0)

    def test_crud_service_input_validation(self):
        """Test CRUD service input validation"""
        from db.crud import create_session, get_session, create_message, get_session_history, get_recent_sessions, update_session_timestamp
        import pytest
        from sqlalchemy.ext.asyncio import AsyncSession
        from unittest.mock import AsyncMock

        # Create a mock database session
        mock_db = AsyncMock(spec=AsyncSession)

        # Test create_session with invalid user_id
        with pytest.raises(ValueError, match="Invalid user_id format"):
            create_session(mock_db, "invalid@user")

        # Test get_session with invalid session_id
        with pytest.raises(ValueError, match="Invalid session_id format"):
            get_session(mock_db, "invalid@session")

        # Test create_message with invalid session_id
        with pytest.raises(ValueError, match="Invalid session_id format"):
            create_message(mock_db, "invalid@session", "user", "test content")

        # Test create_message with invalid role
        with pytest.raises(ValueError, match="Role must be either"):
            create_message(mock_db, "test-session", "invalid_role", "test content")

        # Test create_message with empty content
        with pytest.raises(ValueError, match="Content cannot be empty"):
            create_message(mock_db, "test-session", "user", "")

        # Test create_message with content too long
        with pytest.raises(ValueError, match="Content too long"):
            create_message(mock_db, "test-session", "user", "A" * 60000)

        # Test create_message with invalid message_type
        with pytest.raises(ValueError, match="Message type must be"):
            create_message(mock_db, "test-session", "user", "test content", message_type="invalid_type")

        # Test get_session_history with invalid session_id
        with pytest.raises(ValueError, match="Invalid session_id format"):
            get_session_history(mock_db, "invalid@session")

        # Test get_recent_sessions with invalid limit
        with pytest.raises(ValueError, match="Limit must be between"):
            get_recent_sessions(mock_db, limit=0)

        # Test get_recent_sessions with invalid user_id
        with pytest.raises(ValueError, match="Invalid user_id format"):
            get_recent_sessions(mock_db, "invalid@user", limit=10)

        # Test update_session_timestamp with invalid session_id
        with pytest.raises(ValueError, match="Invalid session_id format"):
            update_session_timestamp(mock_db, "invalid@session")