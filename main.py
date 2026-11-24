import json
import logging
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.application.agent_use_case import AgentUseCase
from src.application.ask_use_case import AskUseCase
from src.application.ingest_use_case import IngestUseCase
from src.config import Settings
from src.domain.ports import DocumentLoaderPort, LLMPort, VectorStorePort
from src.infrastructure.llm.errors import LLMProviderError
from src.infrastructure.llm.fallback_llm_adapter import FallbackLLMAdapter
from src.infrastructure.llm.ollama_adapter import OllamaAdapter
from src.infrastructure.llm.together_adapter import TogetherAdapter
from src.infrastructure.loaders.pymupdf_loader import PyMuPDFLoader
from src.infrastructure.vectorstores.chroma_adapter import ChromaVectorStoreAdapter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("rag_conductor.api")

BASE_DIR = Path(__file__).resolve().parent
WEB_DIR = BASE_DIR / "web"


@dataclass
class AppContainer:
    """Contenedor de dependencias de la aplicacion."""

    settings: Settings
    documents_dir: Path
    llm: LLMPort
    vector_store: VectorStorePort
    document_loader: DocumentLoaderPort
    ingest_use_case: IngestUseCase
    ask_use_case: AskUseCase
    agent_use_case: AgentUseCase
    ingest_lock: threading.Lock


def build_llm_adapter(cfg: Settings) -> FallbackLLMAdapter:
    """Construye la cadena de proveedores LLM (primario + fallback)."""
    ollama = OllamaAdapter(
        host=cfg.ollama_host,
        embedding_model=cfg.embedding_model,
        chat_model=cfg.chat_model,
        timeout=cfg.llm_timeout_seconds,
        max_retries=cfg.llm_max_retries,
    )
    together = TogetherAdapter(
        api_key=cfg.together_api_key,
        base_url=cfg.together_base_url,
        chat_model=cfg.together_chat_model,
        embedding_model=cfg.together_embedding_model,
        timeout=cfg.llm_timeout_seconds,
        max_retries=cfg.llm_max_retries,
    )

    registry = {"ollama": ollama, "together": together}
    primary_name = cfg.llm_primary
    secondary_name = "together" if primary_name == "ollama" else "ollama"

    providers = [(primary_name, registry[primary_name])]
    if cfg.llm_fallback_enabled:
        providers.append((secondary_name, registry[secondary_name]))

    return FallbackLLMAdapter(
        providers=providers,
        embedding_fallback_enabled=cfg.embedding_fallback_enabled,
        embedding_provider=registry[cfg.embedding_provider],
    )


def build_container(cfg: Settings) -> AppContainer:
    """Crea e inyecta todas las dependencias de produccion."""
    documents_dir = (BASE_DIR / cfg.documents_dir).resolve()
    chroma_dir = (BASE_DIR / cfg.chroma_dir).resolve()
    documents_dir.mkdir(parents=True, exist_ok=True)
    chroma_dir.mkdir(parents=True, exist_ok=True)

    llm_adapter = build_llm_adapter(cfg)
    vector_store = ChromaVectorStoreAdapter(
        persist_dir=chroma_dir,
        collection_name=cfg.chroma_collection,
    )
    document_loader = PyMuPDFLoader(
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
    )
    ingest_use_case = IngestUseCase(
        document_loader=document_loader,
        llm=llm_adapter,
        vector_store=vector_store,
        documents_dir=documents_dir,
    )
    ask_use_case = AskUseCase(
        llm=llm_adapter,
        vector_store=vector_store,
        top_k=cfg.top_k,
        min_similarity_score=cfg.min_similarity_score,
    )
    agent_use_case = AgentUseCase(
        llm=llm_adapter,
        vector_store=vector_store,
        default_top_k=cfg.top_k,
        min_similarity_score=cfg.min_similarity_score,
    )
    return AppContainer(
        settings=cfg,
        documents_dir=documents_dir,
        llm=llm_adapter,
        vector_store=vector_store,
        document_loader=document_loader,
        ingest_use_case=ingest_use_case,
        ask_use_case=ask_use_case,
        agent_use_case=agent_use_case,
        ingest_lock=threading.Lock(),
    )


class IngestRequest(BaseModel):
    reindex: bool = False


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)


class AgentMessage(BaseModel):
    role: str
    content: str


class AgentRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)
    history: list[AgentMessage] = Field(default_factory=list)


def create_app(
    settings: Settings | None = None,
    container: AppContainer | None = None,
) -> FastAPI:
    """Factory de la aplicacion FastAPI para produccion y testing."""
    cfg = settings or (container.settings if container else Settings())

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        if container is not None:
            application.state.container = container
        else:
            application.state.container = build_container(cfg)
        yield
        close_fn = getattr(application.state.container.llm, "close", None)
        if callable(close_fn):
            close_fn()

    application = FastAPI(
        title="RAG Conductor",
        description="Sistema RAG local con Arquitectura Hexagonal, Ollama y fallback a Together.ai",
        version="1.0.0",
        lifespan=lifespan,
    )

    origins = [o.strip() for o in cfg.cors_origins.split(",") if o.strip()] or ["*"]
    application.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["*"],
    )

    @application.middleware("http")
    async def security_and_logging_middleware(request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - start) * 1000
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        logger.info(
            "%s %s -> %s (%.1f ms)",
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
        )
        return response

    @application.exception_handler(Exception)
    async def unhandled_exception_handler(_request: Request, exc: Exception):
        logger.exception("Error no controlado: %s", exc)
        return JSONResponse(
            status_code=500,
            content={"detail": "Error interno del servidor."},
        )

    if WEB_DIR.exists():
        application.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")

    def get_c(request: Request) -> AppContainer:
        return request.app.state.container

    @application.get("/", include_in_schema=False)
    def serve_dashboard():
        index_path = WEB_DIR / "index.html"
        if not index_path.exists():
            raise HTTPException(status_code=404, detail="Dashboard no encontrado.")
        return FileResponse(index_path)

    @application.get("/api/health")
    def health(request: Request):
        c = get_c(request)
        return {
            "status": "ok",
            "primary_provider": c.settings.llm_primary,
            "fallback_enabled": c.settings.llm_fallback_enabled,
            "total_chunks": c.vector_store.count(),
            "providers": c.llm.health_check(),
        }

    @application.get("/api/documents")
    def list_documents(request: Request):
        c = get_c(request)
        indexed = c.vector_store.get_indexed_sources()
        pdf_files = sorted(c.documents_dir.glob("*.pdf"))
        items = []
        for pdf in pdf_files:
            info = indexed.get(pdf.name)
            items.append(
                {
                    "name": pdf.name,
                    "size_bytes": pdf.stat().st_size,
                    "indexed": info is not None,
                    "chunk_count": info["chunk_count"] if info else 0,
                }
            )
        return {"documents": items, "total_chunks": c.vector_store.count()}

    @application.post("/api/documents/upload")
    def upload_document(request: Request, file: UploadFile = File(...)):
        c = get_c(request)
        if not file.filename:
            raise HTTPException(status_code=400, detail="Nombre de archivo invalido.")

        safe_name = Path(file.filename).name
        if not safe_name.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Solo se permiten archivos PDF.")

        max_bytes = c.settings.max_upload_mb * 1024 * 1024
        first_chunk = file.file.read(65536)
        if not first_chunk.startswith(b"%PDF-"):
            raise HTTPException(
                status_code=400,
                detail="El archivo no tiene una cabecera PDF valida.",
            )

        dest_path = c.documents_dir / safe_name
        total_written = 0
        try:
            with dest_path.open("wb") as buffer:
                total_written += len(first_chunk)
                if total_written > max_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=f"El archivo excede el limite de {c.settings.max_upload_mb} MB.",
                    )
                buffer.write(first_chunk)
                while chunk := file.file.read(65536):
                    total_written += len(chunk)
                    if total_written > max_bytes:
                        raise HTTPException(
                            status_code=413,
                            detail=f"El archivo excede el limite de {c.settings.max_upload_mb} MB.",
                        )
                    buffer.write(chunk)
        except HTTPException:
            if dest_path.exists():
                dest_path.unlink()
            raise

        return {"filename": safe_name, "size_bytes": total_written}

    @application.delete("/api/documents/{filename}")
    def delete_document(request: Request, filename: str):
        c = get_c(request)
        safe_name = Path(filename).name
        target = c.documents_dir / safe_name
        if not target.exists():
            raise HTTPException(status_code=404, detail="Archivo no encontrado.")

        target.unlink()
        c.vector_store.delete_by_source(safe_name)
        return {"deleted": safe_name, "total_chunks": c.vector_store.count()}

    @application.post("/api/ingest")
    def ingest_documents(request: Request, payload: IngestRequest):
        c = get_c(request)
        if not c.ingest_lock.acquire(blocking=False):
            raise HTTPException(
                status_code=409,
                detail="Ya hay una ingesta en progreso. Intenta nuevamente en unos segundos.",
            )
        try:
            result = c.ingest_use_case.execute(reindex=payload.reindex)
            return asdict(result)
        except LLMProviderError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        finally:
            c.ingest_lock.release()

    @application.post("/api/ask")
    def ask_question(request: Request, payload: AskRequest):
        c = get_c(request)
        try:
            response = c.ask_use_case.execute(payload.question)
            return asdict(response)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except LLMProviderError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @application.post("/api/ask/stream")
    def ask_question_stream(request: Request, payload: AskRequest):
        c = get_c(request)

        def event_generator():
            try:
                for event in c.ask_use_case.execute_stream(payload.question):
                    yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
            except ValueError as exc:
                yield f"data: {json.dumps({'type': 'error', 'detail': str(exc)}, ensure_ascii=False)}\n\n"
            except LLMProviderError as exc:
                yield f"data: {json.dumps({'type': 'error', 'detail': str(exc)}, ensure_ascii=False)}\n\n"
            except Exception as exc:  # noqa: BLE001
                logger.exception("Error en stream SSE: %s", exc)
                yield f"data: {json.dumps({'type': 'error', 'detail': 'Error interno durante el streaming.'}, ensure_ascii=False)}\n\n"

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    @application.post("/api/agent")
    def agent_chat(request: Request, payload: AgentRequest):
        c = get_c(request)
        try:
            history = [m.model_dump() for m in payload.history]
            response = c.agent_use_case.execute(payload.message, history=history)
            return asdict(response)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except LLMProviderError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @application.post("/api/agent/stream")
    def agent_chat_stream(request: Request, payload: AgentRequest):
        c = get_c(request)
        history = [m.model_dump() for m in payload.history]

        def event_generator():
            try:
                for event in c.agent_use_case.execute_stream(payload.message, history=history):
                    yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
            except ValueError as exc:
                yield f"data: {json.dumps({'type': 'error', 'detail': str(exc)}, ensure_ascii=False)}\n\n"
            except LLMProviderError as exc:
                yield f"data: {json.dumps({'type': 'error', 'detail': str(exc)}, ensure_ascii=False)}\n\n"
            except Exception as exc:  # noqa: BLE001
                logger.exception("Error en stream SSE del agente: %s", exc)
                yield f"data: {json.dumps({'type': 'error', 'detail': 'Error interno durante el streaming.'}, ensure_ascii=False)}\n\n"

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    return application


app = create_app()

if __name__ == "__main__":
    cfg = Settings()
    uvicorn.run("main:app", host=cfg.app_host, port=cfg.app_port, reload=False)
