"""Google Gemini text embedding provider using gemini-embedding-001 with rate-limit retry and backoff."""
import asyncio
import logging
from typing import List, Optional
import httpx

from app.rag.embeddings.base import BaseEmbeddingProvider

logger = logging.getLogger(__name__)


class GeminiEmbeddingProvider(BaseEmbeddingProvider):
    """Generates embeddings using Google Gemini REST API with automatic rate-limit retry."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-embedding-001",
        dimension: int = 768,
        version: str = "1.0",
        timeout: float = 30.0,
        max_retries: int = 8,
        base_backoff: float = 4.0,
    ):
        self._api_key = api_key
        self._model_name = model_name
        self._dimension = dimension
        self._version = version
        self._timeout = timeout
        self._max_retries = max_retries
        self._base_backoff = base_backoff

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def version(self) -> str:
        return self._version

    def _get_api_model_path(self) -> str:
        if self._model_name.startswith("models/"):
            return self._model_name
        return f"models/{self._model_name}"

    async def _post_with_retry(self, url: str, json_payload: dict) -> dict:
        """Execute HTTP POST with exponential backoff for rate limits (HTTP 429 / 503)."""
        attempt = 0
        while True:
            try:
                async with httpx.AsyncClient(timeout=self._timeout) as client:
                    response = await client.post(url, json=json_payload)

                    if response.status_code == 429 or response.status_code == 503:
                        attempt += 1
                        if attempt > self._max_retries:
                            response.raise_for_status()
                        wait_time = self._base_backoff * (1.5 ** attempt)
                        logger.warning(
                            "Gemini API rate limit (%d), backing off for %.1fs (attempt %d/%d)...",
                            response.status_code,
                            wait_time,
                            attempt,
                            self._max_retries,
                        )
                        await asyncio.sleep(wait_time)
                        continue

                    response.raise_for_status()
                    return response.json()

            except (httpx.TimeoutException, httpx.NetworkError) as e:
                attempt += 1
                if attempt > self._max_retries:
                    raise e
                wait_time = self._base_backoff * (1.5 ** attempt)
                logger.warning(
                    "Network error (%s), retrying in %.1fs (attempt %d/%d)...",
                    e,
                    wait_time,
                    attempt,
                    self._max_retries,
                )
                await asyncio.sleep(wait_time)

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts using batchEmbedContents endpoint."""
        if not texts:
            return []

        if not self._api_key:
            raise ValueError("Gemini API key is required to generate embeddings.")

        model_path = self._get_api_model_path()
        url = f"https://generativelanguage.googleapis.com/v1beta/{model_path}:batchEmbedContents?key={self._api_key}"

        requests_payload = [
            {
                "model": model_path,
                "content": {"parts": [{"text": text}]},
                "outputDimensionality": self._dimension,
            }
            for text in texts
        ]

        data = await self._post_with_retry(url, {"requests": requests_payload})

        embeddings_data = data.get("embeddings", [])
        if len(embeddings_data) != len(texts):
            raise ValueError(
                f"Embedding response count mismatch: expected {len(texts)}, got {len(embeddings_data)}"
            )

        result: List[List[float]] = []
        for item in embeddings_data:
            vals = item.get("values", [])
            if len(vals) != self._dimension:
                raise ValueError(
                    f"Invalid embedding dimension returned: expected {self._dimension}, got {len(vals)}"
                )
            result.append(vals)

        # Gentle delay to respect RPM limits
        await asyncio.sleep(1.0)
        return result

    async def embed_query(self, text: str) -> List[float]:
        """Generate embedding vector for a single query text."""
        if not text:
            return [0.0] * self._dimension

        if not self._api_key:
            raise ValueError("Gemini API key is required to generate query embedding.")

        model_path = self._get_api_model_path()
        url = f"https://generativelanguage.googleapis.com/v1beta/{model_path}:embedContent?key={self._api_key}"

        payload = {
            "model": model_path,
            "content": {"parts": [{"text": text}]},
            "outputDimensionality": self._dimension,
        }

        data = await self._post_with_retry(url, payload)

        vals = data.get("embedding", {}).get("values", [])
        if len(vals) != self._dimension:
            raise ValueError(
                f"Invalid embedding dimension returned: expected {self._dimension}, got {len(vals)}"
            )

        return vals
