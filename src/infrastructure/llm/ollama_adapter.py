import json
from collections.abc import Iterator

import httpx

from src.domain.models import LLMResult
from src.domain.ports import LLMPort
from src.infrastructure.llm.errors import LLMProviderError
from src.infrastructure.llm.http_utils import request_with_retry


class OllamaAdapter(LLMPort):
    """Adaptador para Ollama local con connection pooling y reintentos."""

    PROVIDER_NAME = "ollama"

    def __init__(
        self,
        host: str = "http://localhost:11434",
        embedding_model: str = "nomic-embed-text",
        chat_model: str = "llama3.1:8b",
        timeout: float = 60.0,
        max_retries: int = 2,
        client: httpx.Client | None = None,
    ) -> None:
        self._host = host.rstrip("/")
        self._embedding_model = embedding_model
        self._chat_model = chat_model
        self._timeout = timeout
        self._max_retries = max_retries
        self._owns_client = client is None
        self._client = client or httpx.Client(timeout=timeout)

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def embed(self, text: str) -> list[float]:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            response = request_with_retry(
                lambda: self._client.post(
                    f"{self._host}/api/embed",
                    json={"model": self._embedding_model, "input": texts},
                ),
                max_retries=self._max_retries,
            )
            data = response.json()
            embeddings = data.get("embeddings")
            if not embeddings:
                raise LLMProviderError(
                    self.PROVIDER_NAME,
                    "Ollama devolvio una respuesta vacia para embeddings.",
                )
            return embeddings
        except httpx.ConnectError as exc:
            raise LLMProviderError(
                self.PROVIDER_NAME,
                f"No se pudo conectar a Ollama en {self._host}. ¿Esta ejecutandose el servicio?",
            ) from exc
        except httpx.HTTPError as exc:
            raise LLMProviderError(self.PROVIDER_NAME, f"Error HTTP en embeddings: {exc}") from exc

    def generate(self, prompt: str, system_prompt: str = "") -> LLMResult:
        messages = self._build_messages(prompt, system_prompt)
        try:
            response = request_with_retry(
                lambda: self._client.post(
                    f"{self._host}/api/chat",
                    json={
                        "model": self._chat_model,
                        "messages": messages,
                        "stream": False,
                    },
                ),
                max_retries=self._max_retries,
            )
            data = response.json()
            content = data.get("message", {}).get("content", "")
            return LLMResult(content=content, provider=self.PROVIDER_NAME, fallback_used=False)
        except httpx.ConnectError as exc:
            raise LLMProviderError(
                self.PROVIDER_NAME,
                f"No se pudo conectar a Ollama en {self._host}. ¿Esta ejecutandose el servicio?",
            ) from exc
        except httpx.HTTPError as exc:
            raise LLMProviderError(self.PROVIDER_NAME, f"Error HTTP en chat: {exc}") from exc

    def generate_stream(
        self,
        prompt: str,
        system_prompt: str = "",
    ) -> Iterator[tuple[str, str, bool]]:
        messages = self._build_messages(prompt, system_prompt)
        try:
            with self._client.stream(
                "POST",
                f"{self._host}/api/chat",
                json={
                    "model": self._chat_model,
                    "messages": messages,
                    "stream": True,
                },
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line:
                        continue
                    payload = json.loads(line)
                    delta = payload.get("message", {}).get("content", "")
                    if delta:
                        yield delta, self.PROVIDER_NAME, False
                    if payload.get("done"):
                        break
        except httpx.ConnectError as exc:
            raise LLMProviderError(
                self.PROVIDER_NAME,
                f"No se pudo conectar a Ollama en {self._host}.",
            ) from exc
        except httpx.HTTPError as exc:
            raise LLMProviderError(self.PROVIDER_NAME, f"Error HTTP en stream: {exc}") from exc

    def health_check(self) -> dict[str, dict]:
        try:
            response = self._client.get(f"{self._host}/api/tags", timeout=5.0)
            response.raise_for_status()
            models = [m.get("name", "") for m in response.json().get("models", [])]
            return {
                self.PROVIDER_NAME: {
                    "available": True,
                    "host": self._host,
                    "chat_model": self._chat_model,
                    "embedding_model": self._embedding_model,
                    "chat_model_ready": self._model_present(self._chat_model, models),
                    "embedding_model_ready": self._model_present(self._embedding_model, models),
                }
            }
        except Exception as exc:  # noqa: BLE001
            return {
                self.PROVIDER_NAME: {
                    "available": False,
                    "host": self._host,
                    "error": str(exc),
                }
            }

    @staticmethod
    def _build_messages(prompt: str, system_prompt: str) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return messages

    @staticmethod
    def _model_present(target: str, installed: list[str]) -> bool:
        return any(m == target or m.startswith(f"{target}:") for m in installed)
