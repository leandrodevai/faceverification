from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    vector_db_distance_metric: str = "l2"
    vector_db_collection: str = "face_embeddings"
    vector_db_persist_directory: str | None = None
    vector_db_n_results: int = 5
    face_match_threshold: float = 1.08

    device: Literal["auto", "cpu", "cuda"] = "auto"
    mtcnn_thresholds: tuple[float, float, float] = (0.6, 0.7, 0.95)
    facenet_pretrained: str = "vggface2"

    demo_username: str = "demo"
    demo_password: str = "demo123"
    jwt_secret_key: str = "change-me-in-production-demo-secret-32-bytes-min"
    jwt_algorithm: str = "HS256"
    jwt_access_token_expire_minutes: int = 60
    max_upload_bytes: int = 5 * 1024 * 1024

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="FACEVERIFICATION_",
        extra="ignore",
    )


settings = Settings()
