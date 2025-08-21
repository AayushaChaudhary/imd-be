from contextlib import asynccontextmanager
from fastapi import FastAPI
import structlog

from app.db.database import Base, engine
from app.models.song import Song
from app.models.user import User
from app.models.interaction import UserSongInteraction


from app.core.config import settings
from app.services.recommender import RecommenderService
from app.utils.logger import setup_logging

logger = structlog.get_logger()
recommender_service: RecommenderService = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up Music Recommender API", version=settings.version)
    setup_logging()

    logger.info("Initializing database and creating tables...")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database tables created.")

    global recommender_service
    recommender_service = RecommenderService(
        data_source=settings.data_source,
        db_url=settings.database_url,
        csv_path=settings.csv_file_path,
        max_rows=settings.max_csv_rows,
    )
    await recommender_service.initialize()
    logger.info("Startup complete")

    yield

    logger.info("Shutting down Music Recommender API")
    if recommender_service:
        await recommender_service.cleanup()
    logger.info("Shutdown complete")


def get_recommender_service() -> RecommenderService:
    if recommender_service is None:
        raise RuntimeError("Recommender service not initialized")
    return recommender_service
