from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuracion tipada de la aplicacion cargada desde .env o variables de entorno."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Servidor y Seguridad
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    max_upload_mb: int = Field(default=50, ge=1, le=500)
    cors_origins: str = "*"

    # Rutas y ChromaDB
    documents_dir: str = "./documents"
    chroma_dir: str = "./chroma_db"
    chroma_collection: str = "rag_conductor"

    # Chunking y RAG
    chunk_size: int = Field(default=1000, ge=100, le=8000)
    chunk_overlap: int = Field(default=200, ge=0, le=2000)
    top_k: int = Field(default=5, ge=1, le=20)
    min_similarity_score: float = Field(default=0.30, ge=0.0, le=1.0)

    # Orquestacion de Proveedores y Fallbacks
    llm_primary: Literal["ollama", "together"] = "ollama"
    llm_fallback_enabled: bool = True
    embedding_fallback_enabled: bool = False
    llm_timeout_seconds: float = Field(default=60.0, gt=1.0)
    llm_max_retries: int = Field(default=2, ge=0, le=5)

    # Ollama
    ollama_host: str = "http://localhost:11434"
    embedding_model: str = "nomic-embed-text"
    chat_model: str = "llama3.1:8b"

    # Together.ai
    together_api_key: str = ""
    together_base_url: str = "https://api.together.xyz/v1"
    together_chat_model: str = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
    together_embedding_model: str = "togethercomputer/m2-bert-80M-8k-retrieval"
