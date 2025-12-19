import asyncio
from db.database import engine
from db.models import Base
from sqlalchemy.ext.asyncio import AsyncEngine
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def init_db():
    """
    Initialize the database by creating all tables
    """
    logger.info("Initializing database...")

    async with engine.begin() as conn:
        # Create all tables
        await conn.run_sync(Base.metadata.create_all)

    logger.info("Database initialized successfully!")

if __name__ == "__main__":
    asyncio.run(init_db())