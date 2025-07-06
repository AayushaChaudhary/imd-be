"""Initialize database tables."""

import asyncio
from sqlalchemy.ext.asyncio import create_async_engine
from app.core.config import settings
from app.db.database import Base
from app.models.song import Song  # Import to register model
import structlog

logger = structlog.get_logger()

async def init_db():
    """Create database tables."""
    logger.info("Initializing database...")
    
    engine = create_async_engine(settings.database_url, echo=True)
    
    async with engine.begin() as conn:
        # Drop all tables (use with caution)
        # await conn.run_sync(Base.metadata.drop_all)
        
        # Create all tables
        await conn.run_sync(Base.metadata.create_all)
    
    await engine.dispose()
    logger.info("Database initialized successfully")

if __name__ == "__main__":
    asyncio.run(init_db())