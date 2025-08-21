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
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import IntegrityError
from sqlalchemy import select

from app.core.config import settings
from app.core.lifespan import get_recommender_service
from app.core import security
from app.db.database import get_db
from app.models.user import User
from app.models import interaction as interaction_model
from app.schemas.recommendation import (
    HealthCheck,
    RecommendationRequest,
    RecommendationResponse,
    RecommendedSong,
)
from app.schemas.song import SongInfo
from app.schemas import user as user_schema
from app.schemas import token as token_schema
from app.services.recommender import RecommenderService
from sqlalchemy import select
from sqlalchemy import delete


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


@router.post("/signup", response_model=user_schema.User, status_code=201)
async def signup(user_in: user_schema.UserCreate, db: AsyncSession = Depends(get_db)):
    """
    Create a new user.
    """
    # Check if user already exists
    user = await db.execute(select(User).where(User.email == user_in.email))
    if user.scalars().first():
        raise HTTPException(
            status_code=400,
            detail="The user with this email already exists in the system.",
        )

    hashed_password = security.get_password_hash(user_in.password)
    db_user = User(email=user_in.email, hashed_password=hashed_password)
    db.add(db_user)
    await db.commit()
    await db.refresh(db_user)
    return db_user


@router.post("/login", response_model=token_schema.Token)
async def login(
    db: AsyncSession = Depends(get_db),
    form_data: OAuth2PasswordRequestForm = Depends(),
):
    """
    OAuth2 compatible token login, get an access token for future requests.
    """
    user_result = await db.execute(select(User).where(User.email == form_data.username))
    user = user_result.scalars().first()
    if not user or not security.verify_password(
        form_data.password, user.hashed_password
    ):
        raise HTTPException(status_code=400, detail="Incorrect email or password")
    elif not user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")

    access_token = security.create_access_token(subject=user.id)
    return {"access_token": access_token, "token_type": "bearer"}


@router.get("/users/me", response_model=user_schema.User)
async def read_users_me(
    db: AsyncSession = Depends(get_db),
    current_user_id: int = Depends(
        security.get_current_user_id
    ),  # This now uses the new function
):
    # async def read_users_me(
    #     db: AsyncSession = Depends(get_db),
    #     current_user_id: int = Depends(
    #         security.get_current_user_id
    #     ),  # We will create this dependency
    # ):
    """
    Get the current logged in user.
    """
    user = await db.get(User, current_user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return user


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


@router.post("/recommend/like/{track_name}", status_code=201)
async def like_a_song(
    track_name: str,
    db: AsyncSession = Depends(get_db),
    recommender: RecommenderService = Depends(get_recommender_service),
    current_user_id: int = Depends(security.get_current_user_id),
):
    """
    Mark a song as 'liked' by the current user.
    This is a protected endpoint.
    """
    # Check if song exists
    if track_name not in recommender.df.index:
        raise HTTPException(status_code=404, detail=f"Song '{track_name}' not found.")

    # Check if interaction already exists
    interaction = await db.execute(
        select(interaction_model.UserSongInteraction).where(
            interaction_model.UserSongInteraction.user_id == current_user_id,
            interaction_model.UserSongInteraction.track_name == track_name,
        )
    )
    if interaction.scalars().first():
        return {"detail": f"You already liked '{track_name}'."}

    # Create the 'like' interaction
    like_interaction = interaction_model.UserSongInteraction(
        user_id=current_user_id, track_name=track_name
    )
    db.add(like_interaction)
    try:
        await db.commit()
    except IntegrityError:  # Handles race conditions
        await db.rollback()
        raise HTTPException(status_code=409, detail="Interaction already exists.")

    return {"detail": f"Successfully liked '{track_name}'."}


@router.get("/users/me/likes", response_model=List[SongInfo])
async def read_user_likes(
    db: AsyncSession = Depends(get_db),
    recommender: RecommenderService = Depends(get_recommender_service),
    current_user_id: int = Depends(security.get_current_user_id),
):
    """
    Get all songs liked by the current user.
    """
    # 1. Get the track names of liked songs from the interaction table
    interactions_result = await db.execute(
        select(interaction_model.UserSongInteraction.track_name).where(
            interaction_model.UserSongInteraction.user_id == current_user_id
        )
    )
    liked_track_names = interactions_result.scalars().all()

    if not liked_track_names:
        return []

    # 2. Fetch the full song details from the main DataFrame
    # Note: df.loc can take a list of index labels
    liked_songs_df = recommender.df.loc[liked_track_names]

    # 3. Format the data to match the SongInfo schema
    songs_to_return = []
    for _, row in liked_songs_df.iterrows():
        songs_to_return.append(
            SongInfo(track_name=row["track_name"], artist_name=row["artist_name"])
        )

    return songs_to_return


@router.delete("/recommend/like/{track_name}", status_code=200)
async def dislike_a_song(
    track_name: str,
    db: AsyncSession = Depends(get_db),
    current_user_id: int = Depends(security.get_current_user_id),
):
    """
    Remove a 'like' interaction for a song by the current user.
    This is a protected endpoint.
    """
    # Define the statement to find the interaction to delete
    stmt = (
        delete(interaction_model.UserSongInteraction)
        .where(
            interaction_model.UserSongInteraction.user_id == current_user_id,
            interaction_model.UserSongInteraction.track_name == track_name,
        )
        .returning(
            interaction_model.UserSongInteraction.id
        )  # To check if something was deleted
    )

    result = await db.execute(stmt)
    deleted_id = result.scalars().first()

    # If nothing was deleted, the user hadn't liked this song
    if deleted_id is None:
        raise HTTPException(
            status_code=404,
            detail=f"Interaction not found: You have not liked '{track_name}'.",
        )

    await db.commit()
    return {"detail": f"Successfully unliked '{track_name}'."}


@router.post("/recommend/for-me", response_model=RecommendationResponse)
async def recommend_for_me(
    recommender: RecommenderService = Depends(get_recommender_service),
    db: AsyncSession = Depends(get_db),
    current_user_id: int = Depends(security.get_current_user_id),
    num_recommendations: int = Form(default=10, ge=1, le=50),
):
    """
    Get personalized recommendations for the current user based on their
    liked songs using collaborative filtering. This is a protected endpoint.
    """
    try:
        recommendations_data = await recommender.recommend_for_user(
            user_id=current_user_id, db=db, num_recommendations=num_recommendations
        )

        # Even if no recs are found, we return a successful empty response
        recommended_songs = [RecommendedSong(**rec) for rec in recommendations_data]

        return RecommendationResponse(
            recommendations=recommended_songs,
            total_count=len(recommended_songs),
            query_song="For You",  # A special name for this query type
            method="collaborative_filtering",
            query_audio_features=None,
        )
    except Exception as e:
        logger.error(
            "Error in collaborative filtering recommendation",
            error=str(e),
            exc_info=True,
        )
        raise HTTPException(
            status_code=500, detail="Failed to get personalized recommendations."
        )


async def cleanup_temp_file(file_path: Path):
    """Clean up temporary file."""
    try:
        if file_path.exists():
            file_path.unlink()
            logger.info("Temp file cleaned up", file=str(file_path))
    except Exception as e:
        logger.error("Failed to cleanup temp file", error=str(e), file=str(file_path))
