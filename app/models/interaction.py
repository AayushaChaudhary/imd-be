from sqlalchemy import Column, DateTime, ForeignKey, Integer, String
from sqlalchemy.sql import func
from app.db.database import Base


class UserSongInteraction(Base):
    __tablename__ = "user_song_interactions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    # Using track_name as the link, since that's the primary song identifier in the service
    track_name = Column(String, ForeignKey("songs.track_name"), nullable=False)
    liked_at = Column(DateTime(timezone=True), server_default=func.now())
