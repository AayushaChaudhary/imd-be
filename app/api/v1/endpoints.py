import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List

import aiofiles
import structlog
from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    Form,
    HTTPException,
    UploadFile,
)

from app.core.config import settings
from app.core.lifespan import get_recommender_service
from app.schemas.recommendation import (
    HealthCheck,
    RecommendationRequest,
    RecommendationResponse,
    RecommendedSong,
)
from app.schemas.song import SongInfo
from app.services.recommender import RecommenderService

logger = structlog.get_logger()
router = APIRouter()


@router.get("/health", response_model=HealthCheck)
async def health_check():
    """Health check endpoint."""
    return HealthCheck(
        status="healthy",
        version=settings.version,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@router.get("/songs", response_model=List[SongInfo])
async def get_songs(recommender: RecommenderService = Depends(get_recommender_service)):
    """Get list of all available songs with their artists."""
    try:
        # We need a method in the service that returns the structured list
        songs = await recommender.get_song_list_with_artists()
        return songs
    except Exception as e:
        logger.error("Error fetching songs", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to fetch songs")


@router.post("/recommend/metadata", response_model=RecommendationResponse)
async def recommend_by_metadata(
    request: RecommendationRequest,
    recommender: RecommenderService = Depends(get_recommender_service),
):
    """Get recommendations based on song metadata."""
    try:
        # Service now returns a tuple (recommendations, features)
        recommendations, _ = await recommender.recommend_by_song_name(
            song_name=request.song_name, num_recommendations=request.num_recommendations
        )

        recommended_songs = [RecommendedSong(**rec) for rec in recommendations]

        return RecommendationResponse(
            recommendations=recommended_songs,
            total_count=len(recommended_songs),
            query_song=request.song_name,
            method="content_based",
            query_audio_features=None,  # No audio features for metadata search
        )

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error("Error in metadata recommendation", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to get recommendations")


# This is the complete function to replace your existing one
@router.post("/recommend/audio", response_model=RecommendationResponse)
async def recommend_by_audio(
    background_tasks: BackgroundTasks,
    recommender: RecommenderService = Depends(get_recommender_service),
    file: UploadFile = File(...),
    num_recommendations: int = Form(default=10, ge=1, le=50),
):
    """
    Get recommendations based on an uploaded audio file.
    This endpoint validates the file, saves it temporarily, calls the
    recommendation service to get audio-based recommendations and feature
    analysis, and then cleans up the temporary file.
    """
    # 1. Validate the uploaded file
    if not file.filename or not file.filename.lower().endswith(
        (".mp3", ".wav", ".flac")
    ):
        raise HTTPException(
            status_code=400, detail="Only MP3, WAV, or FLAC files are supported"
        )

    if file.size and file.size > settings.max_file_size:
        raise HTTPException(
            status_code=413,  # Payload Too Large
            detail=f"File size is too large. Maximum allowed size is {settings.max_file_size / (1024*1024):.1f}MB.",
        )

    # 2. Prepare a unique, temporary path to save the file
    file_id = str(uuid.uuid4())
    file_extension = Path(file.filename).suffix
    temp_filename = f"{file_id}{file_extension}"
    temp_path = Path(settings.upload_dir) / temp_filename

    try:
        # 3. Save the uploaded file to the temporary path asynchronously
        async with aiofiles.open(temp_path, "wb") as f:
            while content := await file.read(
                1024 * 1024
            ):  # Read in 1MB chunks for efficiency
                await f.write(content)

        # 4. Call the recommender service to get both recommendations and features
        # The service now returns a tuple: (list_of_recs, features_dict)
        recommendations, features = await recommender.recommend_by_audio(
            audio_file_path=str(temp_path), num_recommendations=num_recommendations
        )

        # 5. Format the recommendations into Pydantic models for the response
        # This list comprehension now works because `recommendations` is a list of dicts.
        recommended_songs = [RecommendedSong(**rec) for rec in recommendations]

        # 6. Return the full, structured response
        return RecommendationResponse(
            recommendations=recommended_songs,
            total_count=len(recommended_songs),
            query_song=file.filename,
            method="audio_based",
            query_audio_features=features,
        )

    except ValueError as e:
        # Handle specific errors from the service, like feature extraction failure
        logger.warning("Value error during audio recommendation", error=str(e))
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        # Handle all other unexpected errors
        logger.error(
            "Unhandled error in audio recommendation", error=str(e), exc_info=True
        )
        raise HTTPException(
            status_code=500,
            detail="An internal error occurred while processing the audio file.",
        )
    finally:
        # 7. Schedule the temporary file to be deleted after the response is sent
        background_tasks.add_task(cleanup_temp_file, temp_path)


async def cleanup_temp_file(file_path: Path):
    """
    Safely removes a file from the filesystem. To be run in the background.
    """
    try:
        if file_path.exists():
            file_path.unlink()
            logger.info("Temporary file cleaned up", file=str(file_path))
    except Exception as e:
        logger.error(
            "Failed to clean up temporary file", error=str(e), file=str(file_path)
        )


async def cleanup_temp_file(file_path: Path):
    """Clean up temporary file."""
    try:
        if file_path.exists():
            file_path.unlink()
            logger.info("Temp file cleaned up", file=str(file_path))
    except Exception as e:
        logger.error("Failed to cleanup temp file", error=str(e), file=str(file_path))
