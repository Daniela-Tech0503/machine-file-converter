from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    deepseek_api_key: str | None = None
    openrouter_api_key: str | None = None
    deepseek_model: str = "deepseek-chat"
    openrouter_gemini_model: str = "google/gemini-2.5-flash"
    openrouter_ocr_model: str = "google/gemini-2.5-flash"
    max_upload_mb: int = 10
    max_ocr_pages: int = 4
    allowed_origins: str = "http://localhost:5173,http://127.0.0.1:5173"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def cors_origins(self) -> list[str]:
        return [origin.strip() for origin in self.allowed_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
