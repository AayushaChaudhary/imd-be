from sqlalchemy import Column, Integer, String, Float, Text
from app.db.database import Base

class Song(Base):
    __tablename__ = "songs"
    
    id = Column(Integer, primary_key=True, index=True)
    track_name = Column(String, index=True, nullable=False)
    artist_name = Column(String, index=True, nullable=False)
    genre = Column(String, index=True)
    
    # Audio features
    acousticness = Column(Float)
    danceability = Column(Float)
    energy = Column(Float)
    instrumentalness = Column(Float)
    liveness = Column(Float)
    loudness = Column(Float)
    speechiness = Column(Float)
    tempo = Column(Float)
    valence = Column(Float)
    popularity = Column(Float)
    
    # Additional metadata
    album_name = Column(String)
    release_date = Column(String)
    duration_ms = Column(Integer)