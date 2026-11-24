from pathlib import Path

from src.domain.models import DocumentChunk, IngestResult
from src.domain.ports import DocumentLoaderPort, LLMPort, VectorStorePort


class IngestUseCase:
    """Caso de uso para ingestar o re-indexar PDFs locales en el vector store."""

    def __init__(
        self,
        document_loader: DocumentLoaderPort,
        llm: LLMPort,
        vector_store: VectorStorePort,
        documents_dir: Path,
        batch_size: int = 32,
    ) -> None:
        self._loader = document_loader
        self._llm = llm
        self._vector_store = vector_store
        self._documents_dir = documents_dir
        self._batch_size = batch_size

    def execute(self, reindex: bool = False) -> IngestResult:
        """Ejecuta la ingesta incremental o completa de los PDFs en documents_dir."""
        if reindex:
            self._vector_store.clear()
            existing_hashes: dict[str, str] = {}
        else:
            indexed = self._vector_store.get_indexed_sources()
            existing_hashes = {
                source: info.get("file_hash", "")
                for source, info in indexed.items()
                if info.get("file_hash")
            }

        chunks, skipped, errors = self._loader.load_directory(
            self._documents_dir,
            existing_hashes=existing_hashes,
        )

        if not chunks:
            return IngestResult(
                files_processed=0,
                files_skipped=skipped,
                chunks_upserted=0,
                total_chunks_in_store=self._vector_store.count(),
                errors=errors,
            )

        sources_to_update = sorted({chunk.source for chunk in chunks})
        if not reindex:
            # Limpia chunks previos de archivos modificados para evitar duplicados obsoletos.
            for source in sources_to_update:
                if source in existing_hashes:
                    self._vector_store.delete_by_source(source)

        upserted = self._embed_and_upsert(chunks)

        return IngestResult(
            files_processed=len(sources_to_update),
            files_skipped=skipped,
            chunks_upserted=upserted,
            total_chunks_in_store=self._vector_store.count(),
            errors=errors,
        )

    def _embed_and_upsert(self, chunks: list[DocumentChunk]) -> int:
        total_upserted = 0
        for start in range(0, len(chunks), self._batch_size):
            batch = chunks[start : start + self._batch_size]
            texts = [chunk.content for chunk in batch]
            embeddings = self._llm.embed_batch(texts)
            total_upserted += self._vector_store.upsert(batch, embeddings)
        return total_upserted
