import ast
from pathlib import Path
from typing import List, Optional, Union

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # App settings
    app_name: str = "Music Recommender API"
    version: str = "1.0.0"
    debug: bool = Field(default=False, env="DEBUG")

    # Server settings
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=10000, env="PORT")

    # Database settings
    database_url: str = Field(env="DATABASE_URL")
    database_echo: bool = Field(default=False, env="DATABASE_ECHO")

    # Data source settings
    data_source: str = Field(default="csv", env="DATA_SOURCE")
    csv_file_path: str = Field(default="data/dataset.csv", env="CSV_FILE_PATH")
    max_csv_rows: Optional[int] = Field(default=20000, env="MAX_CSV_ROWS")

    # CORS settings
    cors_origins: Union[List[str], str] = Field(
        default=[
            "https://karanmahato44.github.io",
            "https://aayushachaudhary.github.io",
            "http://localhost:3000",
            "http://localhost:5173",
        ],
        env="CORS_ORIGINS",
    )

    @field_validator("cors_origins")
    @classmethod
    def validate_cors_origins(cls, v):
        if isinstance(v, str):
            if v.strip() == "*":
                return ["*"]
            try:
                parsed = ast.literal_eval(v)
                if isinstance(parsed, list):
                    return parsed
                else:
                    return [parsed]
            except (ValueError, SyntaxError):
                return [origin.strip() for origin in v.split(",") if origin.strip()]
        return v

    # ML Model settings
    similarity_features: List[str] = Field(
        default=[
            "acousticness",
            "danceability",
            "energy",
            "instrumentalness",
            "liveness",
            "loudness",
            "speechiness",
            "tempo",
            "valence",
            "popularity",
        ]
    )

    # File upload settings
    upload_dir: str = Field(default="temp_uploads", env="UPLOAD_DIR")
    max_file_size: int = Field(default=10 * 1024 * 1024, env="MAX_FILE_SIZE")

    # Logging
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_format: str = Field(default="json", env="LOG_FORMAT")

    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()

# Ensure upload directory exists
Path(settings.upload_dir).mkdir(parents=True, exist_ok=True)
