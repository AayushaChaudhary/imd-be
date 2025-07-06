from contextlib import asynccontextmanager
from fastapi import FastAPI
import structlog
from app.core.config import settings
from app.services.recommender import RecommenderService
from app.utils.logger import setup_logging

logger = structlog.get_logger()

# Global service instance
recommender_service: RecommenderService = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management."""
    # Startup
    logger.info("Starting up Music Recommender API", version=settings.version)
    
    # Setup logging
    setup_logging()
    
    # Initialize services
    global recommender_service
    recommender_service = RecommenderService(
        data_source=settings.data_source,
        db_url=settings.database_url if settings.data_source == "db" else None,
        csv_path=settings.csv_file_path,
        max_rows=settings.max_csv_rows
    )
    
    # Load and build model
    await recommender_service.initialize()
    
    logger.info("Startup complete")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Music Recommender API")
    if recommender_service:
        await recommender_service.cleanup()
    logger.info("Shutdown complete")

def get_recommender_service() -> RecommenderService:
    """Dependency to get the recommender service."""
    if recommender_service is None:
        raise RuntimeError("Recommender service not initialized")
    return recommender_service