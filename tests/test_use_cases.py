from collections.abc import Iterator
from pathlib import Path

import pytest

from src.application.agent_use_case import AgentUseCase
from src.application.ask_use_case import AskUseCase
from src.application.ingest_use_case import IngestUseCase
from src.domain.models import DocumentChunk, LLMResult
from src.domain.ports import DocumentLoaderPort, LLMPort, VectorStorePort
from src.infrastructure.llm.errors import LLMProviderError
from src.infrastructure.llm.fallback_llm_adapter import FallbackLLMAdapter


class FakeLLM(LLMPort):
    def __init__(self, name: str, should_fail: bool = False) -> None:
        self.name = name
        self.should_fail = should_fail

    def embed(self, text: str) -> list[float]:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if self.should_fail:
            raise LLMProviderError(self.name, "service down")
        return [[0.1, 0.2, 0.3] for _ in texts]

    def generate(self, prompt: str, system_prompt: str = "") -> LLMResult:
        if self.should_fail:
            raise LLMProviderError(self.name, "service down")
        return LLMResult(content=f"respuesta desde {self.name}", provider=self.name)

    def generate_stream(
        self,
        prompt: str,
        system_prompt: str = "",
    ) -> Iterator[tuple[str, str, bool]]:
        if self.should_fail:
            raise LLMProviderError(self.name, "stream down")
        yield "hola ", self.name, False
        yield "mundo", self.name, False

    def health_check(self) -> dict[str, dict]:
        return {self.name: {"available": not self.should_fail}}


class FakeVectorStore(VectorStorePort):
    def __init__(self, chunks: list[DocumentChunk] | None = None) -> None:
        self.chunks = list(chunks or [])
        self.cleared = False
        self.deleted_sources: list[str] = []

    def upsert(self, chunks: list[DocumentChunk], embeddings: list[list[float]]) -> int:
        self.chunks.extend(chunks)
        return len(chunks)

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        min_score: float = 0.0,
        source_filter: str | None = None,
    ) -> list[DocumentChunk]:
        return [
            c
            for c in self.chunks
            if (c.score or 0.0) >= min_score and (source_filter is None or c.source == source_filter)
        ][:top_k]

    def get_page_chunks(self, source: str, page: int) -> list[DocumentChunk]:
        return [c for c in self.chunks if c.source == source and c.page == page]

    def clear(self) -> None:
        self.chunks.clear()
        self.cleared = True

    def count(self) -> int:
        return len(self.chunks)

    def get_indexed_sources(self) -> dict[str, dict]:
        sources: dict[str, dict] = {}
        for c in self.chunks:
            if c.source not in sources:
                sources[c.source] = {"file_hash": c.file_hash, "chunk_count": 0}
            sources[c.source]["chunk_count"] += 1
        return sources

    def delete_by_source(self, source: str) -> None:
        self.deleted_sources.append(source)
        self.chunks = [c for c in self.chunks if c.source != source]


class FakeLoader(DocumentLoaderPort):
    def __init__(self, chunks: list[DocumentChunk], skipped: int = 0) -> None:
        self._chunks = chunks
        self._skipped = skipped

    def load_directory(
        self,
        directory: Path,
        existing_hashes: dict[str, str] | None = None,
    ) -> tuple[list[DocumentChunk], int, list[str]]:
        return self._chunks, self._skipped, []

    def load_file(self, file_path: Path) -> list[DocumentChunk]:
        return self._chunks


def test_fallback_llm_uses_primary_when_healthy():
    adapter = FallbackLLMAdapter(
        providers=[("ollama", FakeLLM("ollama")), ("together", FakeLLM("together"))]
    )
    res = adapter.generate("hola")
    assert res.provider == "ollama"
    assert res.fallback_used is False


def test_fallback_llm_switches_to_secondary_on_failure():
    adapter = FallbackLLMAdapter(
        providers=[
            ("ollama", FakeLLM("ollama", should_fail=True)),
            ("together", FakeLLM("together")),
        ]
    )
    res = adapter.generate("hola")
    assert res.provider == "together"
    assert res.fallback_used is True
    assert "together" in res.content


def test_fallback_llm_stream_switches_on_connection_error():
    adapter = FallbackLLMAdapter(
        providers=[
            ("ollama", FakeLLM("ollama", should_fail=True)),
            ("together", FakeLLM("together")),
        ]
    )
    events = list(adapter.generate_stream("hola"))
    assert events == [("hola ", "together", True), ("mundo", "together", True)]


def test_fallback_llm_blocks_embedding_fallback_by_default():
    adapter = FallbackLLMAdapter(
        providers=[
            ("ollama", FakeLLM("ollama", should_fail=True)),
            ("together", FakeLLM("together")),
        ],
        embedding_fallback_enabled=False,
    )
    with pytest.raises(LLMProviderError):
        adapter.embed("consulta")


def test_ask_use_case_empty_store():
    use_case = AskUseCase(llm=FakeLLM("ollama"), vector_store=FakeVectorStore())
    res = use_case.execute("¿Limite de velocidad?")
    assert res.citations == []
    assert res.provider_used == "none"


def test_ask_use_case_filters_low_similarity_chunks():
    low_score_chunk = DocumentChunk(
        id="1", content="irrelevant", source="manual.pdf", page=3, score=0.12
    )
    use_case = AskUseCase(
        llm=FakeLLM("ollama"),
        vector_store=FakeVectorStore([low_score_chunk]),
        min_similarity_score=0.30,
    )
    res = use_case.execute("¿Limite de velocidad?")
    assert res.citations == []
    assert "No encontré fragmentos" in res.answer


def test_ask_use_case_returns_citations_and_provider_info():
    chunk = DocumentChunk(
        id="1",
        content="En zona urbana la velocidad maxima es 50 km/h.",
        source="libro.pdf",
        page=12,
        score=0.88,
    )
    fallback_llm = FallbackLLMAdapter(
        providers=[
            ("ollama", FakeLLM("ollama", should_fail=True)),
            ("together", FakeLLM("together")),
        ],
        embedding_fallback_enabled=True,
    )
    use_case = AskUseCase(
        llm=fallback_llm,
        vector_store=FakeVectorStore([chunk]),
        min_similarity_score=0.30,
    )
    res = use_case.execute("¿Velocidad maxima urbana?")
    assert res.provider_used == "together"
    assert res.fallback_triggered is True
    assert len(res.citations) == 1
    assert res.citations[0].source == "libro.pdf"
    assert res.citations[0].page == 12


def test_ingest_use_case_replaces_updated_source_and_skips_unchanged():
    existing = DocumentChunk(
        id="libro.pdf::p1::c0",
        content="viejo",
        source="libro.pdf",
        page=1,
        file_hash="hash_v1",
    )
    updated = DocumentChunk(
        id="libro.pdf::p1::c0",
        content="nuevo",
        source="libro.pdf",
        page=1,
        file_hash="hash_v2",
    )
    store = FakeVectorStore([existing])
    loader = FakeLoader([updated], skipped=1)
    use_case = IngestUseCase(
        document_loader=loader,
        llm=FakeLLM("ollama"),
        vector_store=store,
        documents_dir=Path("./documents"),
    )
    result = use_case.execute(reindex=False)
    assert result.files_processed == 1
    assert result.files_skipped == 1
    assert result.chunks_upserted == 1
    assert store.deleted_sources == ["libro.pdf"]


class FakeToolLLM(FakeLLM):
    def __init__(self) -> None:
        super().__init__("together")
        self.calls = 0

    def chat_with_tools(self, messages: list[dict], tools: list[dict]) -> tuple[dict, str, bool]:
        self.calls += 1
        if self.calls == 1:
            return (
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "buscar_en_documentos",
                                "arguments": '{"query": "velocidad urbana"}',
                            },
                        }
                    ],
                },
                "together",
                False,
            )
        return (
            {
                "role": "assistant",
                "content": "En la página 12 se indica que la velocidad máxima urbana es 50 km/h.",
            },
            "together",
            False,
        )


def test_agent_use_case_runs_tool_call_and_returns_citations():
    chunk = DocumentChunk(
        id="1",
        content="En zona urbana la velocidad maxima es 50 km/h.",
        source="libro.pdf",
        page=12,
        score=0.91,
    )
    agent = AgentUseCase(
        llm=FakeToolLLM(),
        vector_store=FakeVectorStore([chunk]),
        min_similarity_score=0.30,
    )
    res = agent.execute("¿Cuál es la velocidad máxima urbana?")
    assert res.provider_used == "together"
    assert "página 12" in res.answer
    assert len(res.citations) == 1
    assert res.citations[0].page == 12


def test_agent_use_case_tools_and_history():
    chunks = [
        DocumentChunk(
            id="libro.pdf::p50::c0",
            content="Sanciones por alcoholemia entre 0,31 y 0,79.",
            source="libro.pdf",
            page=50,
            score=0.88,
        ),
        DocumentChunk(
            id="libro.pdf::p53::c0",
            content="Estado de ebriedad: 0,80 o más gramos por mil.",
            source="libro.pdf",
            page=53,
            score=0.85,
        ),
    ]
    store = FakeVectorStore(chunks)
    agent = AgentUseCase(llm=FakeToolLLM(), vector_store=store, min_similarity_score=0.30)

    out_list, _ = agent._run_tool("listar_documentos", {})
    assert "libro.pdf" in out_list

    out_page, page_chunks = agent._run_tool("leer_pagina", {"documento": "libro.pdf", "pagina": 53})
    assert "Estado de ebriedad" in out_page
    assert len(page_chunks) == 1

    out_cmp, cmp_chunks = agent._run_tool(
        "comparar_temas",
        {"tema_a": "alcoholemia", "tema_b": "ebriedad"},
    )
    assert "TEMA A" in out_cmp and "TEMA B" in out_cmp
    assert len(cmp_chunks) >= 2

    msgs = agent._build_messages(
        "¿Y en qué página estaba?",
        history=[{"role": "user", "content": "Hola"}, {"role": "assistant", "content": "Hola, dime."}],
    )
    assert len(msgs) == 4
    assert msgs[1]["content"] == "Hola"
