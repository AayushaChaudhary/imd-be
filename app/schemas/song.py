from pydantic import BaseModel, Field
from typing import Optional

class SongBase(BaseModel):
    track_name: str = Field(..., min_length=1, max_length=200)
    artist_name: str = Field(..., min_length=1, max_length=200)
    genre: Optional[str] = None
    
    # Audio features
    acousticness: Optional[float] = Field(None, ge=0, le=1)
    danceability: Optional[float] = Field(None, ge=0, le=1)
    energy: Optional[float] = Field(None, ge=0, le=1)
    instrumentalness: Optional[float] = Field(None, ge=0, le=1)
    liveness: Optional[float] = Field(None, ge=0, le=1)
    loudness: Optional[float] = None
    speechiness: Optional[float] = Field(None, ge=0, le=1)
    tempo: Optional[float] = Field(None, gt=0)
    valence: Optional[float] = Field(None, ge=0, le=1)
    popularity: Optional[float] = Field(None, ge=0, le=100)

class Song(SongBase):
    id: int
    
    class Config:
        from_attributes = True

class SongCreate(SongBase):
    pass

class SongUpdate(BaseModel):
    track_name: Optional[str] = Field(None, min_length=1, max_length=200)
    artist_name: Optional[str] = Field(None, min_length=1, max_length=200)
    genre: Optional[str] = None

class SongInfo(BaseModel):
    track_name: str
    artist_name: str