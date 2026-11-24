import time
from collections.abc import Callable

import httpx

RETRYABLE_STATUS_CODES = {429, 502, 503, 504}


def request_with_retry(
    call: Callable[[], httpx.Response],
    max_retries: int = 2,
    base_delay: float = 0.5,
) -> httpx.Response:
    """Ejecuta una peticion HTTP reintentando errores transitorios con backoff exponencial."""
    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            response = call()
            if response.status_code in RETRYABLE_STATUS_CODES and attempt < max_retries:
                time.sleep(base_delay * (2**attempt))
                continue
            response.raise_for_status()
            return response
        except httpx.TimeoutException as exc:
            last_exc = exc
            if attempt < max_retries:
                time.sleep(base_delay * (2**attempt))
                continue
            raise
        except httpx.HTTPStatusError as exc:
            last_exc = exc
            if exc.response.status_code in RETRYABLE_STATUS_CODES and attempt < max_retries:
                time.sleep(base_delay * (2**attempt))
                continue
            raise
    assert last_exc is not None
    raise last_exc
