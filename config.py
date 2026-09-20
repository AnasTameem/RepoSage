import os
from pydantic_settings import BaseSettings, SettingsConfigDict

env = os.getenv("ENVIRONMENT", "dev")

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=f".env.{env}", extra="ignore")

    llm_base_url: str
    llm_api_key: str
    llm_model: str

    llm_fallback_base_url: str | None = None
    llm_fallback_api_key: str | None = None
    llm_fallback_model: str | None = None

    embedding_mode: str = "local"
    embedding_base_url: str | None = None
    embedding_api_key: str | None = None
    embedding_model: str = "BAAI/bge-m3"

    weaviate_url: str
    weaviate_api_key: str | None = None

    session_query_limit: int = 100

settings = Settings()