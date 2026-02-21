from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/investpulse"
    openai_api_key: str = ""
    openai_model: str = "qwen/qwen-2.5-72b-instruct"
    extractor_mode: str = "auto"
    openai_base_url: str = "https://openrouter.ai/api/v1"
    openai_timeout_seconds: float = 30.0
    openai_max_output_tokens: int = 800
    openai_call_budget: int = 3
    openrouter_site_url: str = ""
    openrouter_app_name: str = ""
    extraction_max_content_chars: int = 4000
    reextract_rate_limit_window_seconds: int = 60
    reextract_rate_limit_max_attempts: int = 3
    prompt_version: str = "extract_v1"
    max_assets_in_prompt: int = 50


@lru_cache
def get_settings() -> Settings:
    return Settings()
