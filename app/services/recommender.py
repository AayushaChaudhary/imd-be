# app/services/recommender.py

import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import structlog
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler, normalize

from app.core.config import settings
from app.services.audio_processor import AudioProcessor

logger = structlog.get_logger()


class RecommenderService:
    def __init__(
        self,
        data_source: str = "csv",
        db_url: Optional[str] = None,
        csv_path: str = "data/dataset.csv",
        max_rows: Optional[int] = None,
    ):
        self.data_source = data_source
        self.db_url = db_url
        self.csv_path = csv_path
        self.max_rows = max_rows

        self.df: Optional[pd.DataFrame] = None
        self.similarity_matrix: Optional[np.ndarray] = None
        self.song_vectorizer: Optional[CountVectorizer] = None
        self.scaler: Optional[StandardScaler] = None
        self.audio_features_dict: Dict[
            str, Dict
        ] = {}  # To store features of all dataset songs

        self.audio_processor = AudioProcessor()
        self.features_for_similarity = settings.similarity_features
        self.pseudo_users: List[Dict[str, Any]] = []
        self._initialized = False

    async def initialize(self):
        """Initialize the recommender service."""
        if self._initialized:
            return

        logger.info("Initializing recommender service", data_source=self.data_source)
        await self._load_data()
        await self._build_model()
        await self._precompute_audio_features()  # New step
        self._initialized = True
        logger.info("Recommender service initialized successfully")

    async def _load_data(self):
        """Load data from the configured source."""
        # For simplicity, using the CSV loader you had
        logger.info(f"Loading data from CSV: {self.csv_path}")
        csv_path = Path(self.csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        loop = asyncio.get_event_loop()
        self.df = await loop.run_in_executor(
            None, lambda: pd.read_csv(csv_path, nrows=self.max_rows)
        )

        # Cleaning
        self.df.columns = [col.lower().replace(" ", "_") for col in self.df.columns]
        self.df.dropna(subset=["track_name", "artist_name"], inplace=True)
        self.df.drop_duplicates(subset=["track_name"], keep="first", inplace=True)
        numeric_features = [
            col for col in self.features_for_similarity if col in self.df.columns
        ]
        self.df[numeric_features] = self.df[numeric_features].fillna(
            self.df[numeric_features].mean()
        )
        self.df["genre"] = self.df["genre"].fillna("Unknown")
        self.df.set_index("track_name", inplace=True, drop=False)
        logger.info("Data loaded and cleaned", shape=self.df.shape)

    async def _build_model(self):
        """Build the metadata-based recommendation model."""
        if self.df is None:
            raise ValueError("Data not loaded")
        logger.info("Building metadata recommendation model...")
        self.song_vectorizer = CountVectorizer(max_features=50)
        genre_features = self.song_vectorizer.fit_transform(self.df["genre"]).toarray()
        self.scaler = StandardScaler()
        numeric_array = self.scaler.fit_transform(
            self.df[[f for f in self.features_for_similarity if f in self.df.columns]]
        )
        genre_features = normalize(genre_features, axis=1)
        numeric_array = normalize(numeric_array, axis=1)
        combined_features = np.concatenate(
            [genre_features * 0.3, numeric_array * 0.7], axis=1
        )
        self.similarity_matrix = cosine_similarity(combined_features)
        logger.info("Metadata model built successfully")

    async def _precompute_audio_features(self):
        """
        Pre-compute and store audio features for all songs in the dataset.
        NOTE: This is a simplified version. For a large dataset, this would be
        done offline and features would be loaded from a file or database.
        """
        if self.df is None:
            return
        logger.info("Pre-computing audio features for the dataset...")
        for track_name, row in self.df.iterrows():
            # This is a mock feature generation. A real implementation would
            # process actual audio files for each track.
            # We'll create mock features based on the DataFrame values.
            self.audio_features_dict[track_name] = {
                "tempo": row.get("tempo", 120.0),
                "spectral_centroid": row.get("acousticness", 0.5) * 4000,  # Mocking
                "rms_energy": row.get("energy", 0.5),
                # Mocking chroma and mfccs as vectors of the same value
                "chroma": [row.get("danceability", 0.5)] * 12,
                "mfccs": [row.get("valence", 0.5)] * 13,
            }
        logger.info(f"Pre-computed features for {len(self.audio_features_dict)} songs.")

    def _calculate_audio_similarity(self, features1: Dict, features2: Dict) -> float:
        """Calculate similarity between two sets of audio features."""
        if not features1 or not features2:
            return 0.0

        # Scalar feature similarity (e.g., tempo)
        scalar_sim = 0
        s_features = ["tempo", "spectral_centroid", "rms_energy"]
        for feat in s_features:
            val1, val2 = features1.get(feat), features2.get(feat)
            if val1 and val2 and max(val1, val2) > 0:
                scalar_sim += min(val1, val2) / max(val1, val2)
        scalar_sim /= len(s_features)

        # Vector feature similarity (chroma, mfccs)
        vector_sim = 0
        v_features = ["chroma", "mfccs"]
        for feat in v_features:
            vec1, vec2 = np.array(features1.get(feat)), np.array(features2.get(feat))
            if vec1.size > 0 and vec2.size > 0:
                vector_sim += cosine_similarity([vec1], [vec2])[0][0]
        vector_sim /= len(v_features)

        # Weighted average
        return (scalar_sim * 0.4) + (vector_sim * 0.6)

    async def get_song_list_with_artists(self) -> List[Dict[str, str]]:
        # ... (same as before) ...
        if not self._initialized:
            await self.initialize()
        return (
            self.df[["track_name", "artist_name"]]
            .reset_index(drop=True)
            .to_dict(orient="records")
        )

    async def recommend_by_song_name(
        self, song_name: str, num_recommendations: int = 10
    ) -> Tuple[List[Dict[str, Any]], None]:  # <-- Return a tuple for consistency
        """
        Get recommendations based on song metadata.
        Returns a tuple of (recommendations, None) as there are no audio features.
        """
        if not self._initialized:
            await self.initialize()

        if song_name not in self.df.index:
            raise ValueError(f"Song '{song_name}' not found in dataset")

        song_idx = self.df.index.get_loc(song_name)
        sim_scores = self.similarity_matrix[song_idx]
        similar_indices = np.argsort(sim_scores)[::-1][1 : num_recommendations + 1]

        recommendations = []
        for rank, idx in enumerate(similar_indices, 1):
            song_data = self.df.iloc[idx]
            recommendations.append(
                {
                    "rank": rank,
                    "track_name": song_data["track_name"],
                    "artist_name": song_data["artist_name"],
                    "genre": song_data.get("genre", "Unknown"),
                    "similarity_score": float(sim_scores[idx]),
                }
            )

        return recommendations, None

    async def recommend_by_audio(
        self, audio_file_path: str, num_recommendations: int = 10
    ) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """
        Get recommendations based on uploaded audio file.
        Returns a tuple of (recommendations, audio_features).
        """
        if not self._initialized:
            await self.initialize()

        logger.info("Extracting features from uploaded audio", file=audio_file_path)
        input_features = await self.audio_processor.extract_features(audio_file_path)

        if not input_features:
            raise ValueError("Could not extract features from the uploaded audio file")

        logger.info("Calculating audio similarities against dataset...")
        all_similarities = []
        for track_name, track_features in self.audio_features_dict.items():
            similarity = self._calculate_audio_similarity(
                input_features, track_features
            )
            all_similarities.append(
                {"track_name": track_name, "similarity": similarity}
            )

        if not all_similarities:
            logger.warning("Could not calculate any similarities.")
            return [], input_features

        sorted_songs = sorted(
            all_similarities, key=lambda x: x["similarity"], reverse=True
        )
        top_songs = sorted_songs[:num_recommendations]

        recommendations = []
        for rank, song_sim in enumerate(top_songs, 1):
            track_name = song_sim["track_name"]
            if track_name in self.df.index:
                song_data = self.df.loc[track_name]
                recommendations.append(
                    {
                        "rank": rank,
                        "track_name": song_data["track_name"],
                        "artist_name": song_data["artist_name"],
                        "genre": song_data.get("genre", "Unknown"),
                        "similarity_score": float(song_sim["similarity"]),
                    }
                )

        logger.info("Audio-based recommendations generated", count=len(recommendations))
        return recommendations, input_features

    async def cleanup(self):
        logger.info("Cleaning up recommender service")
        pass
