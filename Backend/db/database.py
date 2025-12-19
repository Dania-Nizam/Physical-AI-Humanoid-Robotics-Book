from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Database URL from environment
DATABASE_URL = os.getenv(
    "NEON_DATABASE_URL",
    "postgresql+asyncpg://neondb_owner:npg_gICNdrwY5pO7@ep-still-meadow-a4lcind5-pooler.us-east-1.aws.neon.tech/neondb?sslmode=require&channel_binding=require"
)

# For testing purposes, we'll use a SQLite database
import os
if os.getenv("TESTING", "false").lower() == "true":
    DATABASE_URL = "sqlite+aiosqlite:///./test.db"

# Create async engine with connection pooling
engine = create_async_engine(
    DATABASE_URL,
    pool_size=5,
    max_overflow=10,
    pool_pre_ping=True,  # Verify connections before use
    pool_recycle=300,    # Recycle connections after 5 minutes
    echo=False           # Set to True for SQL query logging during development
)

# Create async session maker
AsyncSessionLocal = sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False
)

# Dependency to get DB session
async def get_db_session():
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()