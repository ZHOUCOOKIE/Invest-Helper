from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    env: str = "local"
    debug: bool = False
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/investpulse"
    database_url_test: str | None = None
    openai_api_key: str = ""
    openai_model: str = "minimax/minimax-m2.5"
    extractor_mode: str = "auto"
    openai_base_url: str = "https://openrouter.ai/api/v1"
    dummy_fallback: bool = False
    openai_timeout_seconds: float = 30.0
    openai_max_output_tokens: int = 800
    openai_call_budget: int | None = None
    openrouter_site_url: str = ""
    openrouter_app_name: str = ""
    extraction_max_content_chars: int = 4000
    reextract_rate_limit_window_seconds: int = 60
    reextract_rate_limit_max_attempts: int = 3
    prompt_version: str = "extract_v1"
    max_assets_in_prompt: int = 50
    auto_approve_enabled: bool = True
    auto_approve_confidence_threshold: int = 70
    auto_approve_max_views: int = 10
    auto_reject_confidence_threshold: int = 50
    extract_max_concurrency_default: int = 4
    extract_max_rpm_default: int = 120
    extract_batch_size_default: int = 50
    extract_batch_sleep_ms_default: int = 50
    extract_retry_max: int = 5
    extract_retry_backoff_base_ms: int = 800
    extract_retry_backoff_max_ms: int = 20000
    extract_job_max_concurrency: int = 3

    def resolved_database_url(self) -> str:
        if self.env.lower() == "test" and self.database_url_test:
            return self.database_url_test
        return self.database_url


@lru_cache
def get_settings() -> Settings:
    return Settings()
