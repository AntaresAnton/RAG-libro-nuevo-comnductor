from pathlib import Path

import fitz

from src.infrastructure.loaders.pymupdf_loader import PyMuPDFLoader


def _create_sample_pdf(path: Path, pages: list[str]) -> None:
    doc = fitz.open()
    for text in pages:
        page = doc.new_page()
        if text:
            page.insert_text((72, 72), text)
    doc.save(path)
    doc.close()


def test_pymupdf_loader_preserves_page_numbers_and_cleans_text(tmp_path: Path):
    pdf_path = tmp_path / "manual.pdf"
    _create_sample_pdf(
        pdf_path,
        [
            "En zona urbana la velo-\ncidad maxima es\n50 km/h.",
            "",
            "Distancia minima de seguridad en carretera.",
        ],
    )

    loader = PyMuPDFLoader(chunk_size=500, chunk_overlap=50)
    chunks = loader.load_file(pdf_path)

    assert len(chunks) == 2
    assert chunks[0].page == 1
    assert chunks[0].source == "manual.pdf"
    assert "velocidad maxima es 50 km/h." in chunks[0].content
    assert chunks[1].page == 3
    assert chunks[0].file_hash == chunks[1].file_hash


def test_pymupdf_loader_skips_unchanged_and_reports_corrupted(tmp_path: Path):
    valid_pdf = tmp_path / "valid.pdf"
    _create_sample_pdf(valid_pdf, ["Contenido valido de prueba."])

    broken_pdf = tmp_path / "broken.pdf"
    broken_pdf.write_bytes(b"not a real pdf")

    loader = PyMuPDFLoader(chunk_size=500, chunk_overlap=50)
    first_chunks, skipped_first, errors_first = loader.load_directory(tmp_path)

    assert len(first_chunks) == 1
    assert skipped_first == 0
    assert len(errors_first) == 1
    assert "broken.pdf" in errors_first[0]

    existing_hashes = {"valid.pdf": first_chunks[0].file_hash}
    second_chunks, skipped_second, errors_second = loader.load_directory(
        tmp_path, existing_hashes=existing_hashes
    )
    assert second_chunks == []
    assert skipped_second == 1
    assert len(errors_second) == 1
