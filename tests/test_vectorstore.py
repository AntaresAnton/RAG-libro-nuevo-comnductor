from pathlib import Path

from src.domain.models import DocumentChunk
from src.infrastructure.vectorstores.chroma_adapter import ChromaVectorStoreAdapter


def test_chroma_adapter_upsert_search_and_delete(tmp_path: Path):
    store = ChromaVectorStoreAdapter(persist_dir=tmp_path / "chroma", collection_name="test_col")

    chunks = [
        DocumentChunk(
            id="doc1::p1::c0",
            content="Velocidad maxima urbana 50 km/h",
            source="doc1.pdf",
            page=1,
            file_hash="h1",
        ),
        DocumentChunk(
            id="doc2::p4::c0",
            content="Uso obligatorio del cinturon de seguridad",
            source="doc2.pdf",
            page=4,
            file_hash="h2",
        ),
    ]
    embeddings = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ]

    assert store.upsert(chunks, embeddings) == 2
    assert store.count() == 2

    # Consulta cercana al primer vector
    hits = store.search(query_embedding=[0.95, 0.05, 0.0], top_k=2, min_score=0.80)
    assert len(hits) == 1
    assert hits[0].source == "doc1.pdf"
    assert hits[0].page == 1
    assert hits[0].score is not None and hits[0].score >= 0.80

    sources = store.get_indexed_sources()
    assert sources["doc1.pdf"] == {"file_hash": "h1", "chunk_count": 1}
    assert sources["doc2.pdf"] == {"file_hash": "h2", "chunk_count": 1}

    store.delete_by_source("doc1.pdf")
    assert store.count() == 1
    assert "doc1.pdf" not in store.get_indexed_sources()

    store.clear()
    assert store.count() == 0


def test_chroma_adapter_source_filter_and_get_page_chunks(tmp_path: Path):
    store = ChromaVectorStoreAdapter(persist_dir=tmp_path / "chroma_pages", collection_name="test_pages")
    chunks = [
        DocumentChunk(id="a::p10::c1", content="Parte 2", source="a.pdf", page=10, file_hash="h1"),
        DocumentChunk(id="a::p10::c0", content="Parte 1", source="a.pdf", page=10, file_hash="h1"),
        DocumentChunk(id="b::p10::c0", content="Otro doc", source="b.pdf", page=10, file_hash="h2"),
    ]
    store.upsert(chunks, [[1.0, 0.0], [0.9, 0.1], [0.95, 0.05]])

    filtered = store.search(query_embedding=[1.0, 0.0], top_k=5, min_score=0.0, source_filter="b.pdf")
    assert len(filtered) == 1
    assert filtered[0].source == "b.pdf"

    page_chunks = store.get_page_chunks("a.pdf", 10)
    assert [c.id for c in page_chunks] == ["a::p10::c0", "a::p10::c1"]
