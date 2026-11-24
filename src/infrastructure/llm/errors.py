class LLMProviderError(RuntimeError):
    """Error lanzado cuando un proveedor LLM no esta disponible o falla la peticion."""

    def __init__(self, provider: str, message: str) -> None:
        self.provider = provider
        super().__init__(f"[{provider}] {message}")
