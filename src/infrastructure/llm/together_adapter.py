import json
from collections.abc import Iterator

import httpx

from src.domain.models import LLMResult
from src.domain.ports import LLMPort
from src.infrastructure.llm.errors import LLMProviderError
from src.infrastructure.llm.http_utils import request_with_retry


class TogetherAdapter(LLMPort):
    """Adaptador para Together.ai usando la API REST compatible con OpenAI."""

    PROVIDER_NAME = "together"

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.together.xyz/v1",
        chat_model: str = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        embedding_model: str = "togethercomputer/m2-bert-80M-8k-retrieval",
        timeout: float = 60.0,
        max_retries: int = 2,
        client: httpx.Client | None = None,
    ) -> None:
        self._api_key = api_key.strip()
        self._base_url = base_url.rstrip("/")
        self._chat_model = chat_model
        self._embedding_model = embedding_model
        self._timeout = timeout
        self._max_retries = max_retries
        self._owns_client = client is None
        self._client = client or httpx.Client(timeout=timeout)

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def _headers(self) -> dict[str, str]:
        if not self._api_key:
            raise LLMProviderError(
                self.PROVIDER_NAME,
                "TOGETHER_API_KEY no esta configurada.",
            )
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

    def embed(self, text: str) -> list[float]:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            headers = self._headers()
            response = request_with_retry(
                lambda: self._client.post(
                    f"{self._base_url}/embeddings",
                    headers=headers,
                    json={"model": self._embedding_model, "input": texts},
                ),
                max_retries=self._max_retries,
            )
            items = response.json().get("data", [])
            return [item["embedding"] for item in items]
        except LLMProviderError:
            raise
        except httpx.HTTPError as exc:
            raise LLMProviderError(self.PROVIDER_NAME, f"Error HTTP en embeddings: {exc}") from exc

    def generate(self, prompt: str, system_prompt: str = "") -> LLMResult:
        messages = self._build_messages(prompt, system_prompt)
        try:
            headers = self._headers()
            response = request_with_retry(
                lambda: self._client.post(
                    f"{self._base_url}/chat/completions",
                    headers=headers,
                    json={
                        "model": self._chat_model,
                        "messages": messages,
                        "temperature": 0.2,
                        "stream": False,
                    },
                ),
                max_retries=self._max_retries,
            )
            data = response.json()
            content = data["choices"][0]["message"]["content"]
            return LLMResult(content=content, provider=self.PROVIDER_NAME, fallback_used=False)
        except LLMProviderError:
            raise
        except (httpx.HTTPError, KeyError, IndexError) as exc:
            raise LLMProviderError(self.PROVIDER_NAME, f"Error en chat: {exc}") from exc

    def generate_stream(
        self,
        prompt: str,
        system_prompt: str = "",
    ) -> Iterator[tuple[str, str, bool]]:
        yield from self.generate_stream_messages(self._build_messages(prompt, system_prompt))

    def generate_stream_messages(
        self,
        messages: list[dict],
    ) -> Iterator[tuple[str, str, bool]]:
        try:
            headers = self._headers()
            with self._client.stream(
                "POST",
                f"{self._base_url}/chat/completions",
                headers=headers,
                json={
                    "model": self._chat_model,
                    "messages": messages,
                    "temperature": 0.2,
                    "stream": True,
                },
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    raw_data = line.removeprefix("data:").strip()
                    if raw_data == "[DONE]":
                        break
                    payload = json.loads(raw_data)
                    delta = payload["choices"][0].get("delta", {}).get("content", "")
                    if delta:
                        yield delta, self.PROVIDER_NAME, False
        except LLMProviderError:
            raise
        except (httpx.HTTPError, KeyError, IndexError, json.JSONDecodeError) as exc:
            raise LLMProviderError(self.PROVIDER_NAME, f"Error en stream: {exc}") from exc

    def chat_with_tools(
        self,
        messages: list[dict],
        tools: list[dict],
    ) -> tuple[dict, str, bool]:
        try:
            headers = self._headers()
            body: dict = {
                "model": self._chat_model,
                "messages": messages,
                "temperature": 0.2,
                "stream": False,
            }
            if tools:
                body["tools"] = tools
                body["tool_choice"] = "auto"

            response = request_with_retry(
                lambda: self._client.post(
                    f"{self._base_url}/chat/completions",
                    headers=headers,
                    json=body,
                ),
                max_retries=self._max_retries,
            )
            data = response.json()
            raw_msg = data["choices"][0]["message"]
            clean_msg: dict = {
                "role": raw_msg.get("role", "assistant"),
                "content": raw_msg.get("content") or "",
            }
            if raw_msg.get("tool_calls"):
                clean_msg["tool_calls"] = raw_msg["tool_calls"]
            return clean_msg, self.PROVIDER_NAME, False
        except LLMProviderError:
            raise
        except (httpx.HTTPError, KeyError, IndexError) as exc:
            raise LLMProviderError(self.PROVIDER_NAME, f"Error en tool-calling: {exc}") from exc

    def health_check(self) -> dict[str, dict]:
        if not self._api_key:
            return {
                self.PROVIDER_NAME: {
                    "available": False,
                    "configured": False,
                    "error": "TOGETHER_API_KEY no configurada",
                }
            }
        try:
            response = self._client.get(
                f"{self._base_url}/models",
                headers=self._headers(),
                timeout=5.0,
            )
            response.raise_for_status()
            return {
                self.PROVIDER_NAME: {
                    "available": True,
                    "configured": True,
                    "chat_model": self._chat_model,
                    "embedding_model": self._embedding_model,
                }
            }
        except Exception as exc:  # noqa: BLE001
            return {
                self.PROVIDER_NAME: {
                    "available": False,
                    "configured": True,
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
