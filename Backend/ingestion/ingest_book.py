#!/usr/bin/env python3
"""
Book ingestion script for RAG chatbot.

This script processes book content (PDF or plain text) and ingests it into the vector database
for RAG-based question answering.
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging
from dotenv import load_dotenv

from utils.logging_config import get_logger, log_error_with_context

# Load environment variables
load_dotenv()

# Set up logging
logger = get_logger(__name__)

# Import required libraries
try:
    import fitz  # PyMuPDF
    import cohere
    from qdrant_client import QdrantClient
    from qdrant_client.http import models
    import uuid
    from tqdm import tqdm
except ImportError as e:
    logger.error(f"Missing required dependency: {e}")
    logger.error("Please install requirements.txt first: pip install -r requirements.txt")
    sys.exit(1)


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract text from PDF file with page numbers.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        Extracted text with page information
    """
    logger.info(f"Extracting text from PDF: {pdf_path}")
    text = ""
    try:
        doc = fitz.open(pdf_path)
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            page_text = page.get_text()
            text += f"<PAGE {page_num + 1}>\n{page_text}\n</PAGE {page_num + 1}>\n\n"
        doc.close()
        logger.info(f"Successfully extracted text from {len(doc)} pages")
        return text
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {e}")
        raise


def extract_text_from_txt(txt_path: str) -> str:
    """
    Extract text from plain text file.

    Args:
        txt_path: Path to the text file

    Returns:
        Extracted text content
    """
    logger.info(f"Extracting text from TXT: {txt_path}")
    try:
        with open(txt_path, 'r', encoding='utf-8') as file:
            text = file.read()
        logger.info(f"Successfully extracted {len(text)} characters from text file")
        return text
    except Exception as e:
        logger.error(f"Error extracting text from TXT file: {e}")
        raise


def chunk_text(text: str, chunk_size: int = 700, overlap: int = 200) -> List[Dict[str, Any]]:
    """
    Split text into overlapping chunks with metadata.

    Args:
        text: Text to be chunked
        chunk_size: Target size of each chunk (in characters)
        overlap: Overlap between chunks (in characters)

    Returns:
        List of chunk dictionaries with metadata
    """
    logger.info(f"Chunking text of {len(text)} characters (size: {chunk_size}, overlap: {overlap})")

    chunks = []
    start = 0

    # Extract page breaks if they exist in the text
    page_breaks = []
    import re
    page_pattern = r'<PAGE (\d+)>(.*?)</PAGE \1>'
    matches = list(re.finditer(page_pattern, text, re.DOTALL))

    if matches:
        # Handle text with page information
        for i, match in enumerate(matches):
            page_num = int(match.group(1))
            page_text = match.group(2)

            # Chunk the page text
            page_start = 0
            while page_start < len(page_text):
                end = min(page_start + chunk_size, len(page_text))
                chunk_text = page_text[page_start:end]

                # Add overlap if possible
                if end < len(page_text) and overlap > 0:
                    overlap_end = min(end + overlap, len(page_text))
                    chunk_text += page_text[end:overlap_end]

                chunks.append({
                    "text": chunk_text.strip(),
                    "page_number": page_num,
                    "chunk_index": len(chunks),
                    "source_file": "",  # Will be set later
                    "text_length": len(chunk_text)
                })

                page_start += chunk_size
    else:
        # Handle plain text without page information
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk_text = text[start:end]

            # Add overlap if possible
            if end < len(text) and overlap > 0:
                overlap_end = min(end + overlap, len(text))
                chunk_text += text[end:overlap_end]

            chunks.append({
                "text": chunk_text.strip(),
                "page_number": 1,  # Default page number
                "chunk_index": len(chunks),
                "source_file": "",  # Will be set later
                "text_length": len(chunk_text)
            })

            start += chunk_size

    logger.info(f"Created {len(chunks)} chunks")
    return chunks


def create_embeddings(chunks: List[Dict[str, Any]], cohere_client: Any, max_retries: int = 3) -> List[List[float]]:
    """
    Create embeddings for text chunks using Cohere with retry logic.

    Args:
        chunks: List of text chunks with metadata
        cohere_client: Initialized Cohere client
        max_retries: Maximum number of retry attempts for each batch

    Returns:
        List of embedding vectors
    """
    logger.info(f"Creating embeddings for {len(chunks)} chunks")

    # Extract just the text parts for embedding
    texts = [chunk["text"] for chunk in chunks]

    # Cohere has a limit on the number of texts per request, so we'll batch them
    batch_size = 96  # Stay under Cohere's limit
    all_embeddings = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Creating embeddings"):
        batch = texts[i:i + batch_size]

        # Retry logic for embedding creation
        retry_count = 0
        while retry_count < max_retries:
            try:
                response = cohere_client.embed(
                    texts=batch,
                    model="embed-english-v3.0",  # Using the latest English embedding model
                    input_type="search_document"  # Optimize for search
                )
                batch_embeddings = response.embeddings
                all_embeddings.extend(batch_embeddings)
                break  # Success, exit retry loop
            except Exception as e:
                retry_count += 1
                logger.warning(f"Error creating embeddings for batch {i//batch_size + 1} (attempt {retry_count}): {e}")
                if retry_count >= max_retries:
                    logger.error(f"Failed to create embeddings for batch {i//batch_size + 1} after {max_retries} attempts")
                    raise
                else:
                    # Wait before retrying (exponential backoff)
                    import time
                    time.sleep(2 ** retry_count)

    logger.info(f"Successfully created {len(all_embeddings)} embeddings")
    return all_embeddings


def generate_chunk_id(chunk: Dict[str, Any], source_file: str) -> str:
    """
    Generate a deterministic ID for a chunk based on its content and source.
    This ensures idempotency - same chunk will always have the same ID.

    Args:
        chunk: The chunk dictionary
        source_file: Path to the source file

    Returns:
        Deterministic ID for the chunk
    """
    import hashlib

    # Create a unique but deterministic identifier based on content, source, and position
    content = f"{source_file}|{chunk['page_number']}|{chunk['chunk_index']}|{chunk['text'][:100]}"
    chunk_id = hashlib.md5(content.encode()).hexdigest()
    return chunk_id


def check_existing_chunks(qdrant_client: Any, collection_name: str, source_file: str, chunks: List[Dict[str, Any]]) -> List[int]:
    """
    Check which chunks already exist in Qdrant to support resumable ingestion.

    Args:
        qdrant_client: Initialized Qdrant client
        collection_name: Name of the Qdrant collection
        source_file: Path to the source file being ingested
        chunks: List of all chunks to be ingested

    Returns:
        List of indices of chunks that already exist (and can be skipped)
    """
    try:
        # Find all chunks from this source file
        existing_points = qdrant_client.scroll(
            collection_name=collection_name,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="source_file",
                        match=models.MatchValue(value=source_file)
                    )
                ]
            ),
            limit=len(chunks)  # We expect at most this many chunks from this file
        )[0]

        # Extract the source file and chunk index from existing points to identify which ones we have
        existing_chunks = set()
        for point in existing_points:
            if "source_file" in point.payload and "chunk_index" in point.payload:
                if point.payload["source_file"] == source_file:
                    existing_chunks.add(point.payload["chunk_index"])

        # Return indices of chunks that already exist
        existing_indices = []
        for i, chunk in enumerate(chunks):
            if chunk["chunk_index"] in existing_chunks:
                existing_indices.append(i)

        logger.info(f"Found {len(existing_indices)} existing chunks out of {len(chunks)} total")
        return existing_indices

    except Exception as e:
        logger.warning(f"Could not check for existing chunks: {e}. Proceeding with full ingestion.")
        return []


def upsert_to_qdrant(vectors: List[List[float]], chunks: List[Dict[str, Any]],
                     qdrant_client: Any, collection_name: str = "book_chunks", max_retries: int = 3, resume: bool = False):
    """
    Upsert vectors and payloads to Qdrant collection with retry logic, idempotency, and resume capability.

    Args:
        vectors: List of embedding vectors
        chunks: List of text chunks with metadata
        qdrant_client: Initialized Qdrant client
        collection_name: Name of the Qdrant collection
        max_retries: Maximum number of retry attempts for each batch
        resume: Whether to check for and skip existing chunks
    """
    logger.info(f"Upserting {len(vectors)} vectors to Qdrant collection: {collection_name}")

    # Create the collection if it doesn't exist
    try:
        qdrant_client.get_collection(collection_name=collection_name)
        logger.info(f"Collection {collection_name} already exists")
    except:
        logger.info(f"Creating new collection: {collection_name}")
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(size=1024, distance=models.Distance.COSINE),
        )
        logger.info(f"Created collection {collection_name} with 1024-dim vectors")

    # Determine which chunks to process (skip existing ones if resuming)
    chunks_to_process = []
    vectors_to_process = []

    if resume and len(chunks) > 0:
        # Determine source file from the first chunk
        source_file = chunks[0].get("source_file", "")
        if source_file:
            existing_indices = check_existing_chunks(qdrant_client, collection_name, source_file, chunks)
            for i, (vector, chunk) in enumerate(zip(vectors, chunks)):
                if i not in existing_indices:
                    # Generate deterministic ID for idempotency
                    chunk_id = generate_chunk_id(chunk, source_file)
                    chunks_to_process.append((chunk_id, vector, chunk))
        else:
            # If no source file info, process all chunks
            for i, (vector, chunk) in enumerate(zip(vectors, chunks)):
                # Generate a random ID if we don't have source file info
                chunk_id = str(uuid.uuid4())
                chunks_to_process.append((chunk_id, vector, chunk))
    else:
        # Process all chunks
        source_file = chunks[0].get("source_file", "") if chunks else ""
        for i, (vector, chunk) in enumerate(zip(vectors, chunks)):
            chunk_id = generate_chunk_id(chunk, source_file) if source_file else str(uuid.uuid4())
            chunks_to_process.append((chunk_id, vector, chunk))

    logger.info(f"Processing {len(chunks_to_process)} out of {len(chunks)} chunks")

    # Prepare points for upsert
    points = []
    for chunk_id, vector, chunk in chunks_to_process:
        payload = {
            "text": chunk["text"],
            "page_number": chunk["page_number"],
            "chunk_index": chunk["chunk_index"],
            "source_file": chunk["source_file"],
            "text_length": chunk["text_length"]
        }

        points.append(
            models.PointStruct(
                id=chunk_id,  # Use deterministic ID for idempotency
                vector=vector,
                payload=payload
            )
        )

    # Upsert in batches with retry logic
    batch_size = 64  # Qdrant recommended batch size
    for i in tqdm(range(0, len(points), batch_size), desc="Upserting to Qdrant"):
        batch = points[i:i + batch_size]

        # Retry logic for upsert operation
        retry_count = 0
        while retry_count < max_retries:
            try:
                qdrant_client.upsert(
                    collection_name=collection_name,
                    points=batch
                )
                break  # Success, exit retry loop
            except Exception as e:
                retry_count += 1
                logger.warning(f"Error upserting batch {i//batch_size + 1} to Qdrant (attempt {retry_count}): {e}")
                if retry_count >= max_retries:
                    logger.error(f"Failed to upsert batch {i//batch_size + 1} after {max_retries} attempts")
                    raise
                else:
                    # Wait before retrying (exponential backoff)
                    import time
                    time.sleep(2 ** retry_count)

    logger.info(f"Successfully upserted {len(points)} points to Qdrant")


async def main():
    """
    Main ingestion function.
    """
    parser = argparse.ArgumentParser(description="Ingest book content for RAG chatbot")
    parser.add_argument("input_file", help="Path to the input file (PDF or TXT)")
    parser.add_argument("--collection", default="book_chunks", help="Qdrant collection name")
    parser.add_argument("--chunk-size", type=int, default=700, help="Size of text chunks")
    parser.add_argument("--overlap", type=int, default=200, help="Overlap between chunks")
    parser.add_argument("--dry-run", action="store_true", help="Run without actually ingesting")
    parser.add_argument("--resume", action="store_true", help="Resume from a previous partial ingestion")

    args = parser.parse_args()

    # Validate input file
    input_path = Path(args.input_file)
    if not input_path.exists():
        logger.error(f"Input file does not exist: {args.input_file}")
        sys.exit(1)

    if input_path.suffix.lower() not in ['.pdf', '.txt']:
        logger.error(f"Unsupported file type: {input_path.suffix}. Only PDF and TXT files are supported.")
        sys.exit(1)

    # Validate environment variables
    cohere_api_key = os.getenv("COHERE_API_KEY")
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")

    if not all([cohere_api_key, qdrant_url, qdrant_api_key]):
        logger.error("Missing required environment variables. Please set COHERE_API_KEY, QDRANT_URL, and QDRANT_API_KEY")
        sys.exit(1)

    if args.dry_run:
        logger.info("DRY RUN MODE: No actual ingestion will occur")

    try:
        # Initialize clients
        logger.info("Initializing Cohere client...")
        cohere_client = cohere.Client(api_key=cohere_api_key)

        logger.info("Initializing Qdrant client...")
        qdrant_client = QdrantClient(
            url=qdrant_url,
            api_key=qdrant_api_key,
            prefer_grpc=False  # Using HTTP for better compatibility
        )

        # Extract text based on file type
        if input_path.suffix.lower() == '.pdf':
            text = extract_text_from_pdf(str(input_path))
        else:  # .txt
            text = extract_text_from_txt(str(input_path))

        # Set source file in chunk metadata
        # For now, we'll process the text and add source file info to each chunk
        chunks = chunk_text(text, chunk_size=args.chunk_size, overlap=args.overlap)
        for chunk in chunks:
            chunk["source_file"] = str(input_path)

        # Create embeddings
        embeddings = create_embeddings(chunks, cohere_client)

        # Upsert to Qdrant if not in dry-run mode
        if not args.dry_run:
            upsert_to_qdrant(embeddings, chunks, qdrant_client, args.collection, resume=args.resume)
            logger.info(f"Successfully ingested {len(chunks)} chunks to collection '{args.collection}'")
        else:
            logger.info(f"DRY RUN: Would have ingested {len(chunks)} chunks to collection '{args.collection}'")

    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        log_error_with_context(
            logger_instance=logger,
            error_msg=str(e),
            context={
                "input_file": str(input_path),
                "collection": args.collection,
                "chunk_size": args.chunk_size,
                "overlap": args.overlap,
                "operation": "ingestion"
            },
            error_type="ingestion_error"
        )
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())