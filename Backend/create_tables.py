import asyncio
import os
from sqlalchemy.ext.asyncio import create_async_engine
from db.database import DATABASE_URL, connect_args
# Apne models ka 'Base' import karein (check path in your project)
from db.models import Base 

async def init_db():
    print(f"Connecting to: {DATABASE_URL}")
    engine = create_async_engine(DATABASE_URL, connect_args=connect_args)
    
    async with engine.begin() as conn:
        print("Creating tables in Neon Database...")
        # Ye command chat_sessions aur chat_messages tables banaye gi
        await conn.run_sync(Base.metadata.create_all)
        print("Tables created successfully!")
    
    await engine.dispose()

if __name__ == "__main__":
    asyncio.run(init_db())