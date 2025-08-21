import asyncio
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import structlog
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler, normalize
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models.interaction import UserSongInteraction
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
        self.audio_features_dict: Dict[str, Dict] = {}
        self.audio_processor = AudioProcessor()
        self.features_for_similarity = settings.similarity_features
        self._initialized = False

    async def initialize(self):
        """Initialize the recommender service."""
        if self._initialized:
            return
        logger.info("Initializing recommender service", data_source=self.data_source)
        await self._load_data()
        await self._build_model()
        await self._precompute_audio_features()
        self._initialized = True
        logger.info("Recommender service initialized successfully")

    async def _load_data(self):
        """Load data from the configured source."""
        logger.info(f"Loading data from CSV: {self.csv_path}")
        csv_path = Path(self.csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        loop = asyncio.get_event_loop()
        self.df = await loop.run_in_executor(
            None, lambda: pd.read_csv(csv_path, nrows=self.max_rows)
        )
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
        """Build the metadata-based (content-based) recommendation model."""
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
        """Pre-compute mock audio features for all songs in the dataset."""
        if self.df is None:
            return
        logger.info("Pre-computing audio features for the dataset...")
        for track_name, row in self.df.iterrows():
            self.audio_features_dict[track_name] = {
                "tempo": row.get("tempo", 120.0),
                "spectral_centroid": row.get("acousticness", 0.5) * 4000,
                "rms_energy": row.get("energy", 0.5),
                "chroma": [row.get("danceability", 0.5)] * 12,
                "mfccs": [row.get("valence", 0.5)] * 13,
            }
        logger.info(f"Pre-computed features for {len(self.audio_features_dict)} songs.")

    def _calculate_audio_similarity(self, features1: Dict, features2: Dict) -> float:
        """Calculate similarity between two sets of audio features."""
        if not features1 or not features2:
            return 0.0
        scalar_sim, s_features = 0, ["tempo", "spectral_centroid", "rms_energy"]
        for feat in s_features:
            val1, val2 = features1.get(feat), features2.get(feat)
            if val1 and val2 and max(val1, val2) > 0:
                scalar_sim += min(val1, val2) / max(val1, val2)
        scalar_sim /= len(s_features)
        vector_sim, v_features = 0, ["chroma", "mfccs"]
        for feat in v_features:
            vec1, vec2 = np.array(features1.get(feat)), np.array(features2.get(feat))
            if vec1.size > 0 and vec2.size > 0:
                vector_sim += cosine_similarity([vec1], [vec2])[0][0]
        vector_sim /= len(v_features)
        return (scalar_sim * 0.4) + (vector_sim * 0.6)

    async def get_song_list_with_artists(self) -> List[Dict[str, str]]:
        if not self._initialized:
            await self.initialize()
        return (
            self.df[["track_name", "artist_name"]]
            .reset_index(drop=True)
            .to_dict(orient="records")
        )

    async def recommend_for_user(
        self, user_id: int, db: AsyncSession, num_recommendations: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Generate recommendations for a specific user based on collaborative filtering.
        """
        logger.info(
            "Generating collaborative filtering recommendations", user_id=user_id
        )

        # 1. Get the target user's liked songs
        target_likes_result = await db.execute(
            select(UserSongInteraction.track_name).where(
                UserSongInteraction.user_id == user_id
            )
        )
        target_liked_songs = set(target_likes_result.scalars().all())

        if not target_liked_songs:
            logger.warning("Target user has no liked songs.", user_id=user_id)
            return []

        # 2. Find other users who liked the same songs
        similar_interactions_result = await db.execute(
            select(UserSongInteraction.user_id, UserSongInteraction.track_name).where(
                UserSongInteraction.user_id != user_id,
                UserSongInteraction.track_name.in_(target_liked_songs),
            )
        )

        # 3. Score neighbors by counting common likes
        neighbor_scores = defaultdict(int)
        for neighbor_id, _ in similar_interactions_result.all():
            neighbor_scores[neighbor_id] += 1

        if not neighbor_scores:
            logger.warning("No users with similar tastes found.", user_id=user_id)
            return []

        # Sort neighbors by similarity score
        sorted_neighbors = sorted(
            neighbor_scores.items(), key=lambda item: item[1], reverse=True
        )

        # 4. Gather recommendations from top neighbors
        potential_recs = set()
        for neighbor_id, score in sorted_neighbors[:10]:  # Limit to top 10 neighbors
            neighbor_likes_result = await db.execute(
                select(UserSongInteraction.track_name).where(
                    UserSongInteraction.user_id == neighbor_id
                )
            )
            neighbor_liked_songs = set(neighbor_likes_result.scalars().all())
            new_recs = neighbor_liked_songs - target_liked_songs
            potential_recs.update(new_recs)
            if len(potential_recs) >= num_recommendations:
                break

        if not potential_recs:
            logger.warning(
                "Found similar users, but no new song recommendations.", user_id=user_id
            )
            return []

        # 5. Format and return final recommendations
        final_recs = list(potential_recs)[:num_recommendations]
        recommendations = []
        for rank, track_name in enumerate(final_recs, 1):
            if track_name in self.df.index:
                song_data = self.df.loc[track_name]
                recommendations.append(
                    {
                        "rank": rank,
                        "track_name": song_data["track_name"],
                        "artist_name": song_data["artist_name"],
                        "genre": song_data.get("genre", "Unknown"),
                        "similarity_score": None,  # Not applicable for this method
                    }
                )

        logger.info(
            "Successfully generated collaborative recommendations",
            count=len(recommendations),
        )
        return recommendations

    async def recommend_by_song_name(
        self, song_name: str, num_recommendations: int = 10
    ) -> Tuple[List[Dict[str, Any]], None]:
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
        if not self._initialized:
            await self.initialize()
        input_features = await self.audio_processor.extract_features(audio_file_path)
        if not input_features:
            raise ValueError("Could not extract features from the uploaded audio file")
        all_similarities = []
        for track_name, track_features in self.audio_features_dict.items():
            similarity = self._calculate_audio_similarity(
                input_features, track_features
            )
            all_similarities.append(
                {"track_name": track_name, "similarity": similarity}
            )
        if not all_similarities:
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
