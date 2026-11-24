import threading
from pathlib import Path

from fastapi.testclient import TestClient

from main import AppContainer, create_app
from src.application.agent_use_case import AgentUseCase
from src.application.ask_use_case import AskUseCase
from src.application.ingest_use_case import IngestUseCase
from src.config import Settings
from src.domain.models import DocumentChunk
from tests.test_use_cases import FakeLLM, FakeLoader, FakeToolLLM, FakeVectorStore


def _make_test_client(tmp_path: Path, max_upload_mb: int = 1) -> tuple[TestClient, AppContainer]:
    docs_dir = tmp_path / "documents"
    docs_dir.mkdir(parents=True, exist_ok=True)

    settings = Settings(
        documents_dir=str(docs_dir),
        chroma_dir=str(tmp_path / "chroma"),
        max_upload_mb=max_upload_mb,
    )
    llm = FakeLLM("ollama")
    chunk = DocumentChunk(
        id="manual.pdf::p2::c0",
        content="La velocidad maxima en zona urbana es 50 km/h.",
        source="manual.pdf",
        page=2,
        file_hash="abc",
        score=0.91,
    )
    store = FakeVectorStore([chunk])
    loader = FakeLoader([chunk])

    container = AppContainer(
        settings=settings,
        documents_dir=docs_dir,
        llm=llm,
        vector_store=store,
        document_loader=loader,
        ingest_use_case=IngestUseCase(loader, llm, store, docs_dir),
        ask_use_case=AskUseCase(llm, store, top_k=4, min_similarity_score=0.3),
        agent_use_case=AgentUseCase(llm, store, default_top_k=4, min_similarity_score=0.3),
        ingest_lock=threading.Lock(),
    )
    app = create_app(settings=settings, container=container)
    return TestClient(app), container


def test_health_and_security_headers(tmp_path: Path):
    client, _ = _make_test_client(tmp_path)
    with client:
        res = client.get("/api/health")
        assert res.status_code == 200
        assert res.headers["X-Content-Type-Options"] == "nosniff"
        assert res.headers["X-Frame-Options"] == "DENY"
        data = res.json()
        assert data["status"] == "ok"
        assert data["total_chunks"] == 1


def test_upload_validates_extension_magic_bytes_and_size(tmp_path: Path):
    client, container = _make_test_client(tmp_path, max_upload_mb=1)
    with client:
        # Extension invalida
        r_ext = client.post(
            "/api/documents/upload",
            files={"file": ("notas.txt", b"%PDF-1.4 content", "text/plain")},
        )
        assert r_ext.status_code == 400

        # Cabecera PDF falsa
        r_magic = client.post(
            "/api/documents/upload",
            files={"file": ("fake.pdf", b"NOT_A_PDF_HEADER", "application/pdf")},
        )
        assert r_magic.status_code == 400

        # Archivo demasiado grande (> 1 MB)
        oversized = b"%PDF-1.4\n" + b"0" * (1024 * 1024 + 10)
        r_big = client.post(
            "/api/documents/upload",
            files={"file": ("big.pdf", oversized, "application/pdf")},
        )
        assert r_big.status_code == 413
        assert not (container.documents_dir / "big.pdf").exists()

        # PDF valido
        r_ok = client.post(
            "/api/documents/upload",
            files={"file": ("manual.pdf", b"%PDF-1.4\ncontenido", "application/pdf")},
        )
        assert r_ok.status_code == 200
        assert (container.documents_dir / "manual.pdf").exists()

        # Listar y eliminar
        r_list = client.get("/api/documents")
        assert r_list.status_code == 200
        assert len(r_list.json()["documents"]) == 1

        r_del = client.delete("/api/documents/manual.pdf")
        assert r_del.status_code == 200
        assert not (container.documents_dir / "manual.pdf").exists()


def test_ingest_conflict_when_locked(tmp_path: Path):
    client, container = _make_test_client(tmp_path)
    with client:
        assert container.ingest_lock.acquire(blocking=False)
        try:
            res = client.post("/api/ingest", json={"reindex": False})
            assert res.status_code == 409
        finally:
            container.ingest_lock.release()

        res_ok = client.post("/api/ingest", json={"reindex": False})
        assert res_ok.status_code == 200
        assert res_ok.json()["files_processed"] == 1


def test_ask_and_ask_stream(tmp_path: Path):
    client, _ = _make_test_client(tmp_path)
    with client:
        res = client.post("/api/ask", json={"question": "¿Velocidad maxima?"})
        assert res.status_code == 200
        body = res.json()
        assert body["provider_used"] == "ollama"
        assert len(body["citations"]) == 1

        stream_res = client.post("/api/ask/stream", json={"question": "¿Velocidad maxima?"})
        assert stream_res.status_code == 200
        text = stream_res.text
        assert '"type": "citations"' in text


def test_agent_and_agent_stream(tmp_path: Path):
    client, container = _make_test_client(tmp_path)
    container.agent_use_case = AgentUseCase(
        llm=FakeToolLLM(),
        vector_store=container.vector_store,
        default_top_k=4,
        min_similarity_score=0.3,
    )
    with client:
        res = client.post("/api/agent", json={"message": "¿Velocidad maxima?", "history": []})
        assert res.status_code == 200
        body = res.json()
        assert body["provider_used"] == "together"
        assert "página 12" in body["answer"]
        assert len(body["citations"]) == 1

        container.agent_use_case = AgentUseCase(
            llm=FakeToolLLM(),
            vector_store=container.vector_store,
            default_top_k=4,
            min_similarity_score=0.3,
        )
        stream_res = client.post(
            "/api/agent/stream",
            json={"message": "¿Velocidad maxima?", "history": []},
        )
        assert stream_res.status_code == 200
        text = stream_res.text
        assert '"type": "tool"' in text
        assert '"type": "citations"' in text
        assert '"type": "done"' in text
        assert '"type": "token"' in text
        assert '"type": "done"' in text
