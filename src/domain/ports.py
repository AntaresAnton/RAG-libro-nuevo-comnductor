from abc import ABC, abstractmethod
from collections.abc import Iterator
from pathlib import Path

from src.domain.models import DocumentChunk, LLMResult


class DocumentLoaderPort(ABC):
    """Puerto para cargar y fragmentar documentos locales."""

    @abstractmethod
    def load_directory(
        self,
        directory: Path,
        existing_hashes: dict[str, str] | None = None,
    ) -> tuple[list[DocumentChunk], int, list[str]]:
        """Lee los PDFs de un directorio omitiendo los que no cambiaron su hash.

        Retorna (chunks, cantidad_omitida, lista_de_errores).
        """

    @abstractmethod
    def load_file(self, file_path: Path) -> list[DocumentChunk]:
        """Lee y fragmenta un archivo PDF individual."""


class LLMPort(ABC):
    """Puerto para generacion de embeddings y respuestas de chat."""

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        """Genera el vector de embedding para un texto."""

    @abstractmethod
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Genera vectores de embedding para multiples textos."""

    @abstractmethod
    def generate(self, prompt: str, system_prompt: str = "") -> LLMResult:
        """Genera una respuesta completa dado un prompt y un system prompt."""

    @abstractmethod
    def generate_stream(
        self,
        prompt: str,
        system_prompt: str = "",
    ) -> Iterator[tuple[str, str, bool]]:
        """Genera la respuesta en streaming.

        Produce tuplas (token_delta, proveedor_usado, fallback_activado).
        """

    @abstractmethod
    def health_check(self) -> dict[str, dict]:
        """Verifica el estado de disponibilidad de los proveedores LLM."""

    def chat_with_tools(
        self,
        messages: list[dict],
        tools: list[dict],
    ) -> tuple[dict, str, bool]:
        """Ejecuta un turno de chat con soporte de tool calling.

        Retorna (assistant_message_dict, provider_name, fallback_used).
        """
        raise NotImplementedError("Este proveedor no implementa tool calling.")

    def generate_stream_messages(
        self,
        messages: list[dict],
    ) -> Iterator[tuple[str, str, bool]]:
        """Genera una respuesta en streaming a partir de una lista de mensajes."""
        raise NotImplementedError("Este proveedor no implementa streaming de mensajes.")


class VectorStorePort(ABC):
    """Puerto para almacenamiento y busqueda vectorial."""

    @abstractmethod
    def upsert(self, chunks: list[DocumentChunk], embeddings: list[list[float]]) -> int:
        """Inserta o actualiza chunks y sus embeddings. Retorna cantidad procesada."""

    @abstractmethod
    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        min_score: float = 0.0,
        source_filter: str | None = None,
    ) -> list[DocumentChunk]:
        """Busca los chunks mas similares filtrando por min_score (0..1) y fuente opcional."""

    def get_page_chunks(self, source: str, page: int) -> list[DocumentChunk]:
        """Retorna todos los chunks de una pagina especifica de un documento."""
        return []

    @abstractmethod
    def clear(self) -> None:
        """Elimina todos los documentos de la coleccion."""

    @abstractmethod
    def count(self) -> int:
        """Retorna el numero total de chunks indexados."""

    @abstractmethod
    def get_indexed_sources(self) -> dict[str, dict]:
        """Retorna un mapa {source: {'file_hash': str, 'chunk_count': int}}."""

    @abstractmethod
    def delete_by_source(self, source: str) -> None:
        """Elimina todos los chunks pertenecientes a un archivo fuente."""
