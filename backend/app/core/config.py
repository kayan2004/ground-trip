from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Smart Travel Assistant API"
    app_env: str = "development"
    app_debug: bool = True
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/smart_travel_assistant"
    database_echo: bool = False
    jwt_secret_key: str = "change-this-development-secret-to-32-plus-chars"
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 60
    rag_seed_documents_path: str = "data/rag_seed_documents.json"
    rag_chunk_size: int = 800
    rag_chunk_overlap: int = 120

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
