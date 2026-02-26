from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    env: str = "local"
    debug: bool = False
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/investpulse"
    openai_api_key: str = ""
    openai_model: str = "qwen/qwen-2.5-72b-instruct"
    extractor_mode: str = "auto"
    openai_base_url: str = "https://openrouter.ai/api/v1"
    dummy_fallback: bool = False
    openai_timeout_seconds: float = 30.0
    openai_max_output_tokens: int = 800
    openai_call_budget: int | None = None
    call_budget_per_hour: int = 30
    call_budget_window_minutes: int = 60
    openrouter_site_url: str = ""
    openrouter_app_name: str = ""
    extraction_max_content_chars: int = 4000
    reextract_rate_limit_window_seconds: int = 60
    reextract_rate_limit_max_attempts: int = 3
    prompt_version: str = "extract_v1"
    max_assets_in_prompt: int = 50
    auto_approve_enabled: bool = True
    auto_approve_confidence_threshold: int = 70
    auto_approve_min_display_confidence: int = 50
    auto_approve_max_views: int = 10
    auto_reject_confidence_threshold: int = 50
    extract_max_concurrency_default: int = 4
    extract_max_rpm_default: int = 120
    extract_batch_size_default: int = 50
    extract_batch_sleep_ms_default: int = 50
    extract_retry_max: int = 5
    extract_retry_backoff_base_ms: int = 800
    extract_retry_backoff_max_ms: int = 20000
    extract_max_concurrency_max: int = 8
    extract_max_rpm_max: int = 240
    extract_batch_size_max: int = 100
    extract_batch_sleep_ms_min: int = 20
    extract_adaptive_throttle_enabled: bool = True
    extract_adaptive_recovery_successes: int = 12


@lru_cache
def get_settings() -> Settings:
    return Settings()
