from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
import os
from dotenv import load_dotenv
import ssl

# Load environment variables
load_dotenv()

# 1. DATABASE_URL ko .env se uthaein (Default mein sslmode hata diya hai)
DATABASE_URL = os.getenv(
    "NEON_DATABASE_URL",
    "postgresql+asyncpg://neondb_owner:npg_gICNdrwY5pO7@ep-still-meadow-a4lcind5-pooler.us-east-1.aws.neon.tech/neondb"
)

# Testing check
IS_TESTING = os.getenv("TESTING", "false").lower() == "true"
if IS_TESTING:
    DATABASE_URL = "sqlite+aiosqlite:///./test.db"

# 2. SSL Context sirf PostgreSQL/Neon ke liye banayein
connect_args = {}

if not IS_TESTING and "postgresql" in DATABASE_URL:
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE
    connect_args["ssl"] = ssl_context # Neon ke liye SSL lazmi hai

# 3. Create engine
engine = create_async_engine(
    DATABASE_URL,
    connect_args=connect_args
)

# Async session maker
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