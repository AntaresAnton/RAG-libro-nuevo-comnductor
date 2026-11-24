import hashlib
import re
from pathlib import Path

import fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.domain.models import DocumentChunk
from src.domain.ports import DocumentLoaderPort


class PyMuPDFLoader(DocumentLoaderPort):
    """Carga PDFs pagina por pagina con PyMuPDF, limpia saltos de linea y aplica chunking."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200) -> None:
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def load_directory(
        self,
        directory: Path,
        existing_hashes: dict[str, str] | None = None,
    ) -> tuple[list[DocumentChunk], int, list[str]]:
        if not directory.exists():
            directory.mkdir(parents=True, exist_ok=True)
            return [], 0, []

        known_hashes = existing_hashes or {}
        pdf_files = sorted(directory.glob("*.pdf"))
        all_chunks: list[DocumentChunk] = []
        skipped = 0
        errors: list[str] = []

        for pdf_path in pdf_files:
            try:
                file_hash = self._compute_sha256(pdf_path)
                if known_hashes.get(pdf_path.name) == file_hash:
                    skipped += 1
                    continue
                chunks = self._load_file_with_hash(pdf_path, file_hash)
                all_chunks.extend(chunks)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{pdf_path.name}: {exc}")

        return all_chunks, skipped, errors

    def load_file(self, file_path: Path) -> list[DocumentChunk]:
        file_hash = self._compute_sha256(file_path)
        return self._load_file_with_hash(file_path, file_hash)

    def _load_file_with_hash(self, file_path: Path, file_hash: str) -> list[DocumentChunk]:
        chunks: list[DocumentChunk] = []
        with fitz.open(file_path) as doc:
            for page_idx, page in enumerate(doc, start=1):
                raw_text = page.get_text("text")
                cleaned_text = self._clean_text(raw_text)
                if not cleaned_text:
                    continue

                split_texts = self._splitter.split_text(cleaned_text)
                for chunk_idx, text in enumerate(split_texts):
                    chunk_id = f"{file_path.name}::p{page_idx}::c{chunk_idx}"
                    chunks.append(
                        DocumentChunk(
                            id=chunk_id,
                            content=text,
                            source=file_path.name,
                            page=page_idx,
                            file_hash=file_hash,
                        )
                    )
        return chunks

    @staticmethod
    def _clean_text(text: str) -> str:
        # Une palabras cortadas por guion al final de linea
        text = re.sub(r"-\n(\w)", r"\1", text)
        # Normaliza saltos de linea multiples a doble salto (parrafos)
        text = re.sub(r"\n{3,}", "\n\n", text)
        # Convierte saltos de linea simples dentro de un parrafo en espacios
        text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
        # Colapsa espacios repetidos
        text = re.sub(r"[ \t]{2,}", " ", text)
        return text.strip()

    @staticmethod
    def _compute_sha256(file_path: Path) -> str:
        hasher = hashlib.sha256()
        with file_path.open("rb") as f:
            for block in iter(lambda: f.read(65536), b""):
                hasher.update(block)
        return hasher.hexdigest()
