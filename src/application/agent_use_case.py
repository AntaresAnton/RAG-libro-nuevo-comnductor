import json
from collections.abc import Iterator
from dataclasses import asdict

from src.domain.models import Citation, DocumentChunk, RagResponse
from src.domain.ports import LLMPort, VectorStorePort

AGENT_SYSTEM_PROMPT = """Eres un agente asistente experto, cercano y conversacional sobre los documentos cargados en la base vectorial (como el Libro del Nuevo Conductor).

Tienes herramientas para consultar directamente los PDFs indexados. Úsalas siempre que necesites datos, cifras, normas, sanciones o contenido de los documentos.

Reglas:
1. Responde en español de forma natural, clara y fluida, como si conversaras directamente con la persona.
2. Usa SOLO información obtenida mediante las herramientas. Nunca inventes normas, cifras ni artículos.
3. Integra las páginas de forma natural en tu explicación, conectando cómo se complementan entre sí (por ejemplo: "En la página X aparece esta información, complementada con lo que se indica en la página Y, además de lo señalado en la página Z..."). Evita citas mecánicas entre corchetes.
4. Si te piden un quiz o examen práctico, busca primero en los documentos, formula preguntas claras con alternativas (A, B, C) y espera la respuesta del usuario antes de revelar las soluciones (a menos que pida verlas de inmediato).
5. Si te piden comparar temas o analizar un escenario práctico, realiza las búsquedas necesarias antes de responder.
6. Si no encuentras suficiente información en los documentos, dilo con honestidad."""

AGENT_TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "buscar_en_documentos",
            "description": (
                "Busca fragmentos relevantes en la base vectorial usando similitud semántica. "
                "Úsala para responder preguntas, preparar resúmenes o buscar normas específicas."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Consulta o tema a buscar en los documentos.",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Cantidad máxima de fragmentos a recuperar (por defecto 5).",
                    },
                    "documento": {
                        "type": "string",
                        "description": "Nombre exacto del PDF si quieres filtrar por un archivo específico.",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "leer_pagina",
            "description": (
                "Lee el contenido completo de una página específica de un PDF indexado. "
                "Úsala cuando necesites ver toda la información de una página sin cortes."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "documento": {
                        "type": "string",
                        "description": "Nombre del archivo PDF (ej. LIBRO-DEL-NUEVO-CONDUCTOR-AUTOMOVILISTAS_2026.pdf).",
                    },
                    "pagina": {
                        "type": "integer",
                        "description": "Número de página a leer.",
                    },
                },
                "required": ["documento", "pagina"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "listar_documentos",
            "description": "Lista los documentos PDF actualmente indexados en la base vectorial y cuántos chunks tiene cada uno.",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "comparar_temas",
            "description": (
                "Busca información sobre dos temas en paralelo para compararlos o analizar cómo se complementan "
                "(por ejemplo, velocidades en zona urbana vs rural, o infracción vs sanción)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "tema_a": {"type": "string", "description": "Primer tema o norma a buscar."},
                    "tema_b": {"type": "string", "description": "Segundo tema o norma a buscar."},
                },
                "required": ["tema_a", "tema_b"],
            },
        },
    },
]


class AgentUseCase:
    """Agente ReAct con Tool Calling sobre ChromaDB y soporte de historial."""

    def __init__(
        self,
        llm: LLMPort,
        vector_store: VectorStorePort,
        default_top_k: int = 5,
        min_similarity_score: float = 0.25,
        max_iterations: int = 5,
    ) -> None:
        self._llm = llm
        self._vector_store = vector_store
        self._default_top_k = default_top_k
        self._min_similarity_score = min_similarity_score
        self._max_iterations = max_iterations

    def execute_stream(
        self,
        message: str,
        history: list[dict] | None = None,
    ) -> Iterator[dict]:
        """Ejecuta el ciclo del agente emitiendo eventos SSE (tool, citations, token, done)."""
        clean_msg = message.strip()
        if not clean_msg:
            raise ValueError("El mensaje no puede estar vacío.")

        if self._vector_store.count() == 0:
            msg = "No hay documentos indexados todavía. Sube un PDF en la barra lateral y presiona **Ingerir nuevos**."
            yield {"type": "citations", "citations": []}
            yield {"type": "token", "content": msg, "provider": "none", "fallback": False}
            yield {"type": "done", "provider": "none", "fallback": False}
            return

        messages = self._build_messages(clean_msg, history)
        collected_chunks: list[DocumentChunk] = []
        last_provider = "unknown"
        last_fallback = False

        for _ in range(self._max_iterations):
            assistant_msg, provider, fallback = self._llm.chat_with_tools(
                messages=messages,
                tools=AGENT_TOOLS,
            )
            last_provider = provider
            last_fallback = fallback
            messages.append(assistant_msg)

            tool_calls = assistant_msg.get("tool_calls") or []
            if not tool_calls:
                citations = [asdict(c) for c in self._dedup_citations(collected_chunks)]
                yield {"type": "citations", "citations": citations}

                content = assistant_msg.get("content") or ""
                if content:
                    yield {
                        "type": "token",
                        "content": content,
                        "provider": provider,
                        "fallback": fallback,
                    }
                yield {"type": "done", "provider": provider, "fallback": fallback}
                return

            for tc in tool_calls:
                fn = tc.get("function", {})
                tool_name = fn.get("name", "")
                args = self._parse_args(fn.get("arguments"))
                yield {"type": "tool", "name": tool_name, "args": args}

                tool_output, new_chunks = self._run_tool(tool_name, args)
                collected_chunks.extend(new_chunks)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.get("id", "call_0"),
                        "content": tool_output,
                    }
                )

        # Si alcanzo max_iterations con tool_calls pendientes, fuerza sintesis final en streaming
        citations = [asdict(c) for c in self._dedup_citations(collected_chunks)]
        yield {"type": "citations", "citations": citations}

        for token, provider, fallback in self._llm.generate_stream_messages(messages=messages):
            last_provider = provider
            last_fallback = fallback
            yield {
                "type": "token",
                "content": token,
                "provider": provider,
                "fallback": fallback,
            }

        yield {"type": "done", "provider": last_provider, "fallback": last_fallback}

    def execute(self, message: str, history: list[dict] | None = None) -> RagResponse:
        """Version sincrona del agente."""
        answer_parts: list[str] = []
        citations: list[Citation] = []
        provider_used = "none"
        fallback_triggered = False

        for event in self.execute_stream(message=message, history=history):
            etype = event.get("type")
            if etype == "citations":
                citations = [Citation(**c) for c in event.get("citations", [])]
            elif etype == "token":
                answer_parts.append(event.get("content", ""))
            elif etype == "done":
                provider_used = event.get("provider", "none")
                fallback_triggered = bool(event.get("fallback", False))

        return RagResponse(
            answer="".join(answer_parts),
            citations=citations,
            provider_used=provider_used,
            fallback_triggered=fallback_triggered,
        )

    def _run_tool(self, name: str, args: dict) -> tuple[str, list[DocumentChunk]]:
        if name == "buscar_en_documentos":
            query = str(args.get("query", "")).strip()
            top_k = max(1, min(int(args.get("top_k", self._default_top_k)), 10))
            documento = args.get("documento") or None
            if not query:
                return "Error: falta el parámetro 'query'.", []
            chunks = self._search(query, top_k=top_k, source_filter=documento)
            return self._format_chunks(chunks), chunks

        if name == "leer_pagina":
            documento = str(args.get("documento", "")).strip()
            pagina = int(args.get("pagina", 0))
            if not documento or pagina <= 0:
                return "Error: especifica 'documento' y un número de 'pagina' válido.", []
            chunks = self._vector_store.get_page_chunks(source=documento, page=pagina)
            if not chunks:
                return f"No se encontró contenido para {documento} en la página {pagina}.", []
            return self._format_chunks(chunks), chunks

        if name == "listar_documentos":
            sources = self._vector_store.get_indexed_sources()
            if not sources:
                return "No hay documentos indexados.", []
            lines = [
                f"- {src}: {info.get('chunk_count', 0)} fragmentos indexados"
                for src, info in sources.items()
            ]
            return "Documentos indexados:\n" + "\n".join(lines), []

        if name == "comparar_temas":
            tema_a = str(args.get("tema_a", "")).strip()
            tema_b = str(args.get("tema_b", "")).strip()
            chunks_a = self._search(tema_a, top_k=4) if tema_a else []
            chunks_b = self._search(tema_b, top_k=4) if tema_b else []
            text_a = self._format_chunks(chunks_a)
            text_b = self._format_chunks(chunks_b)
            combined = f"=== TEMA A ({tema_a}) ===\n{text_a}\n\n=== TEMA B ({tema_b}) ===\n{text_b}"
            return combined, chunks_a + chunks_b

        return f"Herramienta desconocida: {name}", []

    def _search(
        self,
        query: str,
        top_k: int,
        source_filter: str | None = None,
    ) -> list[DocumentChunk]:
        embedding = self._llm.embed(query)
        return self._vector_store.search(
            query_embedding=embedding,
            top_k=top_k,
            min_score=self._min_similarity_score,
            source_filter=source_filter,
        )

    @staticmethod
    def _format_chunks(chunks: list[DocumentChunk]) -> str:
        if not chunks:
            return "No se encontraron fragmentos relevantes para esa búsqueda."
        blocks = [
            f"[{i}] Documento: {c.source} | Página: {c.page}\n{c.content}"
            for i, c in enumerate(chunks, start=1)
        ]
        return "\n\n---\n\n".join(blocks)

    @staticmethod
    def _parse_args(raw_args) -> dict:
        if isinstance(raw_args, dict):
            return raw_args
        if isinstance(raw_args, str) and raw_args.strip():
            try:
                return json.loads(raw_args)
            except json.JSONDecodeError:
                return {}
        return {}

    @staticmethod
    def _build_messages(message: str, history: list[dict] | None) -> list[dict]:
        messages: list[dict] = [{"role": "system", "content": AGENT_SYSTEM_PROMPT}]
        if history:
            for item in history[-10:]:
                role = item.get("role")
                content = item.get("content")
                if role in ("user", "assistant") and isinstance(content, str) and content.strip():
                    messages.append({"role": role, "content": content})
        messages.append({"role": "user", "content": message})
        return messages

    @staticmethod
    def _dedup_citations(chunks: list[DocumentChunk]) -> list[Citation]:
        seen: set[tuple[str, int, str]] = set()
        citations: list[Citation] = []
        for c in chunks:
            key = (c.source, c.page, c.id)
            if key in seen:
                continue
            seen.add(key)
            citations.append(
                Citation(
                    source=c.source,
                    page=c.page,
                    excerpt=c.content,
                    score=c.score,
                )
            )
        return citations
