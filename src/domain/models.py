from dataclasses import dataclass, field


@dataclass(frozen=True)
class DocumentChunk:
    """Fragmento de texto extraido de un PDF con sus metadatos de origen."""

    id: str
    content: str
    source: str
    page: int
    file_hash: str = ""
    score: float | None = None


@dataclass(frozen=True)
class Citation:
    """Referencia verificable al documento y pagina de origen."""

    source: str
    page: int
    excerpt: str
    score: float | None = None


@dataclass(frozen=True)
class LLMResult:
    """Resultado de generacion del LLM incluyendo trazabilidad del proveedor."""

    content: str
    provider: str
    fallback_used: bool = False


@dataclass(frozen=True)
class RagResponse:
    """Respuesta final del sistema RAG con citas y metadatos del proveedor."""

    answer: str
    citations: list[Citation] = field(default_factory=list)
    provider_used: str = "none"
    fallback_triggered: bool = False


@dataclass(frozen=True)
class IngestResult:
    """Resumen de una operacion de ingesta o re-indexacion de PDFs."""

    files_processed: int
    files_skipped: int
    chunks_upserted: int
    total_chunks_in_store: int
    errors: list[str] = field(default_factory=list)
