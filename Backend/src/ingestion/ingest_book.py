import os
import cohere
import uuid
import asyncio
import time
from tqdm import tqdm
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models

load_dotenv()

async def ingest_from_text(file_path, collection="book_chunks"):
    if not os.path.exists(file_path):
        print(f"❌ Error: {file_path} nahi mili!")
        return

    print(f"📖 Reading {file_path}...")
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    
    # 1. Chunking
    chunks = [text[i:i+500] for i in range(0, len(text), 400)]
    print(f"✂️ Created {len(chunks)} chunks.")
    
    co = cohere.Client(os.getenv("COHERE_API_KEY"))
    # Timeout barhane ke liye client configuration
    q_client = QdrantClient(
        url=os.getenv("QDRANT_URL"), 
        api_key=os.getenv("QDRANT_API_KEY"),
        timeout=60 # Connection timeout ko 60 seconds tak barha diya
    )

    # Collection check/create
    try:
        q_client.get_collection(collection)
    except:
        q_client.create_collection(
            collection_name=collection,
            vectors_config=models.VectorParams(size=1024, distance=models.Distance.COSINE)
        )

    all_embeddings = []
    c_batch_size = 90 

    print("🧠 Step 1: Generating Embeddings (Cohere)...")
    for i in tqdm(range(0, len(chunks), c_batch_size), desc="Cohere Batches"):
        batch = chunks[i : i + c_batch_size]
        try:
            response = co.embed(texts=batch, model="embed-english-v3.0", input_type="search_document")
            all_embeddings.extend(response.embeddings)
            if i + c_batch_size < len(chunks):
                time.sleep(10) # Trial key limit delay
        except Exception as e:
            print(f"Wait... Rate limit hit. Sleeping 30s.")
            time.sleep(30)
            response = co.embed(texts=batch, model="embed-english-v3.0", input_type="search_document")
            all_embeddings.extend(response.embeddings)

    # 2. Batched Upload to Qdrant
    print(f"📤 Step 2: Uploading {len(all_embeddings)} vectors to Qdrant (Batched)...")
    q_batch_size = 100 # Ek waqt mein sirf 100 points jayenge
    
    for i in tqdm(range(0, len(all_embeddings), q_batch_size), desc="Qdrant Upload"):
        batch_vectors = all_embeddings[i : i + q_batch_size]
        batch_chunks = chunks[i : i + q_batch_size]
        
        points = [
            models.PointStruct(
                id=str(uuid.uuid4()), 
                vector=v, 
                payload={"text": c, "source_file": file_path}
            ) for v, c in zip(batch_vectors, batch_chunks)
        ]
        
        try:
            q_client.upsert(collection_name=collection, points=points)
        except Exception as e:
            print(f"\n⚠️ Upload retry: {e}")
            time.sleep(5)
            q_client.upsert(collection_name=collection, points=points)

    print(f"✅ Mubarak Ho! Poora data Qdrant mein chala gaya hai.")

if __name__ == "__main__":
    asyncio.run(ingest_from_text("book.txt"))