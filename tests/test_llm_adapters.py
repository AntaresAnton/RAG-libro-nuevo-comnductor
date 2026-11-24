import json

import httpx
import pytest

from src.infrastructure.llm.errors import LLMProviderError
from src.infrastructure.llm.fallback_llm_adapter import FallbackLLMAdapter
from src.infrastructure.llm.ollama_adapter import OllamaAdapter
from src.infrastructure.llm.together_adapter import TogetherAdapter


def test_ollama_adapter_retries_and_succeeds():
    attempts = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["count"] += 1
        if request.url.path == "/api/chat" and attempts["count"] == 1:
            return httpx.Response(503, json={"error": "busy"})
        if request.url.path == "/api/chat":
            return httpx.Response(200, json={"message": {"content": "ok ollama"}})
        if request.url.path == "/api/embed":
            return httpx.Response(200, json={"embeddings": [[0.1, 0.2]]})
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "llama3.1:8b"}, {"name": "nomic-embed-text:latest"}]})
        return httpx.Response(404)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    adapter = OllamaAdapter(max_retries=1, client=client)

    res = adapter.generate("hola")
    assert res.content == "ok ollama"
    assert attempts["count"] == 2
    assert adapter.embed("texto") == [0.1, 0.2]

    health = adapter.health_check()["ollama"]
    assert health["available"] is True
    assert health["chat_model_ready"] is True
    assert health["embedding_model_ready"] is True


def test_ollama_adapter_stream():
    ndjson = "\n".join(
        [
            json.dumps({"message": {"content": "Hola "}, "done": False}),
            json.dumps({"message": {"content": "conductor"}, "done": True}),
        ]
    )

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=ndjson.encode("utf-8"))

    client = httpx.Client(transport=httpx.MockTransport(handler))
    adapter = OllamaAdapter(client=client)
    tokens = list(adapter.generate_stream("pregunta"))
    assert tokens == [("Hola ", "ollama", False), ("conductor", "ollama", False)]


def test_together_adapter_requires_api_key():
    adapter = TogetherAdapter(api_key="")
    with pytest.raises(LLMProviderError):
        adapter.generate("hola")
    health = adapter.health_check()["together"]
    assert health["available"] is False
    assert health["configured"] is False


def test_together_adapter_chat_and_stream():
    sse = (
        'data: {"choices":[{"delta":{"content":"Respuesta "}}]}\n\n'
        'data: {"choices":[{"delta":{"content":"Together"}}]}\n\n'
        "data: [DONE]\n\n"
    )

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/chat/completions"):
            body = json.loads(request.content.decode("utf-8"))
            if body.get("stream"):
                return httpx.Response(200, content=sse.encode("utf-8"))
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": "sync together"}}]},
            )
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": []})
        return httpx.Response(404)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    adapter = TogetherAdapter(api_key="test-key", client=client)

    assert adapter.generate("hola").content == "sync together"
    tokens = list(adapter.generate_stream("hola"))
    assert tokens == [("Respuesta ", "together", False), ("Together", "together", False)]
    assert adapter.health_check()["together"]["available"] is True


def test_together_adapter_chat_with_tools():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/chat/completions"):
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": None,
                                "reasoning_content": "pensando...",
                                "tool_calls": [
                                    {
                                        "id": "call_abc",
                                        "type": "function",
                                        "function": {
                                            "name": "buscar_en_documentos",
                                            "arguments": '{"query": "alcohol"}',
                                        },
                                    }
                                ],
                            }
                        }
                    ]
                },
            )
        return httpx.Response(404)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    adapter = TogetherAdapter(api_key="test-key", client=client)
    msg, provider, fallback = adapter.chat_with_tools(
        messages=[{"role": "user", "content": "alcohol"}],
        tools=[{"type": "function", "function": {"name": "buscar_en_documentos"}}],
    )
    assert provider == "together"
    assert fallback is False
    assert msg["content"] == ""
    assert "reasoning_content" not in msg
    assert msg["tool_calls"][0]["function"]["name"] == "buscar_en_documentos"


def test_ollama_adapter_chat_with_tools_normalizes_arguments():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/chat":
            return httpx.Response(
                200,
                json={
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "leer_pagina",
                                    "arguments": {"documento": "libro.pdf", "pagina": 50},
                                }
                            }
                        ],
                    }
                },
            )
        return httpx.Response(404)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    adapter = OllamaAdapter(client=client)
    msg, provider, fallback = adapter.chat_with_tools(
        messages=[{"role": "user", "content": "lee pag 50"}],
        tools=[],
    )
    assert provider == "ollama"
    assert fallback is False
    assert msg["tool_calls"][0]["id"] == "call_0"
    assert json.loads(msg["tool_calls"][0]["function"]["arguments"]) == {
        "documento": "libro.pdf",
        "pagina": 50,
    }


def test_fallback_adapter_uses_separate_embedding_provider():
    def ollama_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/embed":
            return httpx.Response(200, json={"embeddings": [[0.5, 0.5]]})
        return httpx.Response(404)

    def together_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/chat/completions"):
            return httpx.Response(200, json={"choices": [{"message": {"content": "ok together"}}]})
        return httpx.Response(500)

    ollama = OllamaAdapter(client=httpx.Client(transport=httpx.MockTransport(ollama_handler)))
    together = TogetherAdapter(
        api_key="test-key",
        client=httpx.Client(transport=httpx.MockTransport(together_handler)),
    )
    fallback = FallbackLLMAdapter(
        providers=[("together", together), ("ollama", ollama)],
        embedding_fallback_enabled=False,
        embedding_provider=ollama,
    )
    assert fallback.embed("hola") == [0.5, 0.5]
    assert fallback.generate("hola").provider == "together"
