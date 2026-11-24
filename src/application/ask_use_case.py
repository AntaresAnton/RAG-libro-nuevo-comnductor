from collections.abc import Iterator
from dataclasses import asdict

from src.domain.models import Citation, DocumentChunk, RagResponse
from src.domain.ports import LLMPort, VectorStorePort

SYSTEM_PROMPT = """Eres un asistente cercano, claro y experto que responde preguntas EXCLUSIVAMENTE usando los fragmentos de contexto proporcionados.

Reglas estrictas:
1. Responde en español de forma natural, fluida y conversacional, como si le explicaras directamente a una persona, usando Markdown ligero solo cuando ayude a ordenar la lectura.
2. Usa SOLO la información presente en el contexto. No inventes datos, cifras ni normas.
3. Si el contexto no contiene suficiente información para responder, indícalo explícitamente de forma amable.
4. Integra las referencias a las páginas de manera orgánica dentro del relato, explicando cómo se complementan entre sí (por ejemplo: "En la página X aparece esta información, complementada con lo que se indica en la página Y, además de lo señalado en la página Z..."). Evita citas mecánicas entre corchetes como [archivo.pdf, pág. X]; menciona el nombre del archivo solo si hay varios documentos distintos y hace falta diferenciarlos."""


class AskUseCase:
    """Caso de uso para responder consultas mediante recuperacion semantica y LLM."""

    def __init__(
        self,
        llm: LLMPort,
        vector_store: VectorStorePort,
        top_k: int = 5,
        min_similarity_score: float = 0.30,
    ) -> None:
        self._llm = llm
        self._vector_store = vector_store
        self._top_k = top_k
        self._min_similarity_score = min_similarity_score

    def execute(self, question: str) -> RagResponse:
        """Ejecuta el flujo RAG completo de forma sincrona."""
        clean_question = question.strip()
        if not clean_question:
            raise ValueError("La pregunta no puede estar vacía.")

        if self._vector_store.count() == 0:
            return RagResponse(
                answer="No hay documentos indexados todavía. Sube o coloca un PDF en `documents/` y presiona **Ingerir documentos**.",
                citations=[],
                provider_used="none",
                fallback_triggered=False,
            )

        chunks = self._retrieve_chunks(clean_question)
        if not chunks:
            return RagResponse(
                answer="No encontré fragmentos suficientemente relevantes en los documentos indexados para responder esa pregunta con certeza.",
                citations=[],
                provider_used="none",
                fallback_triggered=False,
            )

        prompt = self._build_user_prompt(clean_question, chunks)
        result = self._llm.generate(prompt=prompt, system_prompt=SYSTEM_PROMPT)

        return RagResponse(
            answer=result.content,
            citations=self._build_citations(chunks),
            provider_used=result.provider,
            fallback_triggered=result.fallback_used,
        )

    def execute_stream(self, question: str) -> Iterator[dict]:
        """Ejecuta el flujo RAG emitiendo eventos estructurados para streaming SSE."""
        clean_question = question.strip()
        if not clean_question:
            raise ValueError("La pregunta no puede estar vacía.")

        if self._vector_store.count() == 0:
            msg = "No hay documentos indexados todavía. Sube o coloca un PDF en `documents/` y presiona **Ingerir documentos**."
            yield {"type": "citations", "citations": []}
            yield {"type": "token", "content": msg, "provider": "none", "fallback": False}
            yield {"type": "done", "provider": "none", "fallback": False}
            return

        chunks = self._retrieve_chunks(clean_question)
        if not chunks:
            msg = "No encontré fragmentos suficientemente relevantes en los documentos indexados para responder esa pregunta con certeza."
            yield {"type": "citations", "citations": []}
            yield {"type": "token", "content": msg, "provider": "none", "fallback": False}
            yield {"type": "done", "provider": "none", "fallback": False}
            return

        citations = [asdict(c) for c in self._build_citations(chunks)]
        yield {"type": "citations", "citations": citations}

        prompt = self._build_user_prompt(clean_question, chunks)
        last_provider = "unknown"
        last_fallback = False

        for token, provider, fallback in self._llm.generate_stream(
            prompt=prompt,
            system_prompt=SYSTEM_PROMPT,
        ):
            last_provider = provider
            last_fallback = fallback
            yield {
                "type": "token",
                "content": token,
                "provider": provider,
                "fallback": fallback,
            }

        yield {"type": "done", "provider": last_provider, "fallback": last_fallback}

    def _retrieve_chunks(self, question: str) -> list[DocumentChunk]:
        query_embedding = self._llm.embed(question)
        return self._vector_store.search(
            query_embedding=query_embedding,
            top_k=self._top_k,
            min_score=self._min_similarity_score,
        )

    @staticmethod
    def _build_citations(chunks: list[DocumentChunk]) -> list[Citation]:
        return [
            Citation(
                source=chunk.source,
                page=chunk.page,
                excerpt=chunk.content,
                score=chunk.score,
            )
            for chunk in chunks
        ]

    @staticmethod
    def _build_user_prompt(question: str, chunks: list[DocumentChunk]) -> str:
        context_blocks = [
            f"[{i}] Documento: {chunk.source} | Página: {chunk.page}\n{chunk.content}"
            for i, chunk in enumerate(chunks, start=1)
        ]
        joined_context = "\n\n---\n\n".join(context_blocks)
        return (
            f"CONTEXTO RECUPERADO:\n{joined_context}\n\n"
            f"PREGUNTA DEL USUARIO:\n{question}\n\n"
            "RESPUESTA NATURAL (indicando de forma fluida en qué página o páginas aparece cada información y cómo se complementan):"
        )
