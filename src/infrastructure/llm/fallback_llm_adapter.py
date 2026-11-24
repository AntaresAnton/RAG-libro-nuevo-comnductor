import logging
from collections.abc import Iterator

from src.domain.models import LLMResult
from src.domain.ports import LLMPort
from src.infrastructure.llm.errors import LLMProviderError

logger = logging.getLogger(__name__)


class FallbackLLMAdapter(LLMPort):
    """Orquestador de proveedores LLM con soporte de fallback automatico.

    Para generacion de texto (sync y streaming), intenta los proveedores en orden.
    Para embeddings, por defecto usa solo el primario para evitar inconsistencias
    de dimensiones en ChromaDB, salvo que embedding_fallback_enabled=True.
    """

    def __init__(
        self,
        providers: list[tuple[str, LLMPort]],
        embedding_fallback_enabled: bool = False,
        embedding_provider: LLMPort | None = None,
    ) -> None:
        if not providers:
            raise ValueError("Debe configurarse al menos un proveedor LLM.")
        self._providers = providers
        self._embedding_fallback_enabled = embedding_fallback_enabled
        self._embedding_provider = embedding_provider

    def embed(self, text: str) -> list[float]:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if self._embedding_provider is not None and not self._embedding_fallback_enabled:
            return self._embedding_provider.embed_batch(texts)

        candidates = self._providers if self._embedding_fallback_enabled else self._providers[:1]
        if self._embedding_provider is not None:
            candidates = [("embedding_primary", self._embedding_provider)] + [
                p for p in self._providers if p[1] is not self._embedding_provider
            ]
        errors: list[str] = []
        for idx, (name, provider) in enumerate(candidates):
            try:
                return provider.embed_batch(texts)
            except LLMProviderError as exc:
                logger.warning("Fallo embeddings en '%s': %s", name, exc)
                errors.append(str(exc))
                if idx == len(candidates) - 1:
                    break
        raise LLMProviderError("fallback", " | ".join(errors))

    def generate(self, prompt: str, system_prompt: str = "") -> LLMResult:
        errors: list[str] = []
        for idx, (name, provider) in enumerate(self._providers):
            try:
                result = provider.generate(prompt=prompt, system_prompt=system_prompt)
                return LLMResult(
                    content=result.content,
                    provider=name,
                    fallback_used=idx > 0,
                )
            except LLMProviderError as exc:
                logger.warning("Fallo generacion en '%s', intentando fallback: %s", name, exc)
                errors.append(str(exc))
        raise LLMProviderError("fallback", "Todos los proveedores fallaron: " + " | ".join(errors))

    def generate_stream(
        self,
        prompt: str,
        system_prompt: str = "",
    ) -> Iterator[tuple[str, str, bool]]:
        errors: list[str] = []
        for idx, (name, provider) in enumerate(self._providers):
            yielded_any = False
            try:
                for token, _, _ in provider.generate_stream(
                    prompt=prompt,
                    system_prompt=system_prompt,
                ):
                    yielded_any = True
                    yield token, name, idx > 0
                return
            except LLMProviderError as exc:
                logger.warning("Fallo stream en '%s': %s", name, exc)
                errors.append(str(exc))
                # Si ya emitimos tokens al cliente, no podemos reiniciar el stream con otro proveedor
                if yielded_any:
                    raise LLMProviderError(
                        name,
                        f"El stream se interrumpio a mitad de la respuesta: {exc}",
                    ) from exc
        raise LLMProviderError("fallback", "Todos los proveedores fallaron: " + " | ".join(errors))

    def chat_with_tools(
        self,
        messages: list[dict],
        tools: list[dict],
    ) -> tuple[dict, str, bool]:
        errors: list[str] = []
        for idx, (name, provider) in enumerate(self._providers):
            try:
                msg, _, _ = provider.chat_with_tools(messages=messages, tools=tools)
                return msg, name, idx > 0
            except (LLMProviderError, NotImplementedError) as exc:
                logger.warning("Fallo tool-calling en '%s', intentando fallback: %s", name, exc)
                errors.append(str(exc))
        raise LLMProviderError("fallback", "Todos los proveedores fallaron: " + " | ".join(errors))

    def generate_stream_messages(
        self,
        messages: list[dict],
    ) -> Iterator[tuple[str, str, bool]]:
        errors: list[str] = []
        for idx, (name, provider) in enumerate(self._providers):
            yielded_any = False
            try:
                for token, _, _ in provider.generate_stream_messages(messages=messages):
                    yielded_any = True
                    yield token, name, idx > 0
                return
            except (LLMProviderError, NotImplementedError) as exc:
                logger.warning("Fallo stream mensajes en '%s': %s", name, exc)
                errors.append(str(exc))
                if yielded_any:
                    raise LLMProviderError(
                        name,
                        f"El stream se interrumpio a mitad de la respuesta: {exc}",
                    ) from exc
        raise LLMProviderError("fallback", "Todos los proveedores fallaron: " + " | ".join(errors))

    def health_check(self) -> dict[str, dict]:
        status: dict[str, dict] = {}
        for _, provider in self._providers:
            status.update(provider.health_check())
        return status

    def close(self) -> None:
        for _, provider in self._providers:
            close_fn = getattr(provider, "close", None)
            if callable(close_fn):
                close_fn()
