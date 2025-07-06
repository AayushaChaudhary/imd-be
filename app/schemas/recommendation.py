from pydantic import BaseModel, Field
from typing import List, Optional


class AudioFeatures(BaseModel):
    tempo: float
    spectral_centroid: float
    rms_energy: float
    chroma: List[float]
    mfccs: List[float]


class RecommendationRequest(BaseModel):
    song_name: str = Field(..., min_length=1, max_length=200)
    num_recommendations: int = Field(default=10, ge=1, le=50)


class AudioUploadRequest(BaseModel):
    num_recommendations: int = Field(default=10, ge=1, le=50)


class RecommendedSong(BaseModel):
    rank: int = Field(..., ge=1)
    track_name: str
    artist_name: str
    similarity_score: Optional[float] = Field(None, ge=0, le=1)
    genre: Optional[str] = None


class RecommendationResponse(BaseModel):
    recommendations: List[RecommendedSong]
    total_count: int
    query_song: Optional[str] = None
    method: str = Field(..., description="Method used for recommendation")
    query_audio_features: Optional[AudioFeatures] = None


class HealthCheck(BaseModel):
    status: str
    version: str
    timestamp: str
