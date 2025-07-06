"""Load data from CSV to database."""

import asyncio
import pandas as pd
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from app.core.config import settings
from app.models.song import Song
import structlog

logger = structlog.get_logger()

async def load_csv_to_db():
    """Load CSV data to database."""
    logger.info("Starting data load process...")
    
    # Create async engine and session
    engine = create_async_engine(settings.database_url, echo=False)
    AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    
    try:
        # Load CSV
        logger.info(f"Loading CSV file: {settings.csv_file_path}")
        df = pd.read_csv(settings.csv_file_path)
        
        # Clean column names
        df.columns = [col.lower().replace(' ', '_') for col in df.columns]
        
        # Basic cleaning
        df.dropna(subset=['track_name', 'artist_name'], inplace=True)
        df.drop_duplicates(subset=['track_name'], keep='first', inplace=True)
        
        logger.info(f"Loaded {len(df)} records from CSV")
        
        # Insert data in batches
        batch_size = 1000
        total_inserted = 0
        
        async with AsyncSessionLocal() as session:
            for i in range(0, len(df), batch_size):
                batch = df.iloc[i:i + batch_size]
                
                songs = []
                for _, row in batch.iterrows():
                    song = Song(
                        track_name=row.get('track_name'),
                        artist_name=row.get('artist_name'),
                        genre=row.get('genre'),
                        acousticness=row.get('acousticness'),
                        danceability=row.get('danceability'),
                        energy=row.get('energy'),
                        instrumentalness=row.get('instrumentalness'),
                        liveness=row.get('liveness'),
                        loudness=row.get('loudness'),
                        speechiness=row.get('speechiness'),
                        tempo=row.get('tempo'),
                        valence=row.get('valence'),
                        popularity=row.get('popularity')
                    )
                    songs.append(song)
                
                session.add_all(songs)
                await session.commit()
                
                total_inserted += len(songs)
                logger.info(f"Inserted batch {i//batch_size + 1}, total: {total_inserted}")
        
        logger.info(f"Data load completed! Total records inserted: {total_inserted}")
        
    except Exception as e:
        logger.error("Error loading data", error=str(e))
        raise
    finally:
        await engine.dispose()

if __name__ == "__main__":
    asyncio.run(load_csv_to_db())