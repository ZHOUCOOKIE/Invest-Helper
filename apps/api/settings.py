from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/investpulse"
    openai_api_key: str = ""
    openai_model: str = "gpt-4.1-mini"
    extractor_mode: str = "auto"
    openai_base_url: str = "https://api.openai.com/v1"
    openai_timeout_seconds: float = 30.0
    extraction_max_content_chars: int = 4000
    reextract_rate_limit_window_seconds: int = 60
    reextract_rate_limit_max_attempts: int = 3


@lru_cache
def get_settings() -> Settings:
    return Settings()
