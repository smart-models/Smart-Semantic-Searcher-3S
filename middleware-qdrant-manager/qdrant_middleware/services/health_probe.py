"""Background Qdrant and embedder reachability probe service."""

import asyncio
import logging
import time

import httpx
from qdrant_client import AsyncQdrantClient

from qdrant_middleware.middleware.metrics import qdrant_up

log = logging.getLogger(__name__)

EMBEDDER_PROBE_TIMEOUT = 3.0


class HealthProbe:
    """Cache Qdrant and Smart Embedder embedder health in the background."""

    def __init__(
        self,
        client: AsyncQdrantClient,
        interval: float,
        embedder_url: str | None = None,
        probe_timeout: float = EMBEDDER_PROBE_TIMEOUT,
        embedder_client_token: str | None = None,
    ) -> None:
        """Configure Qdrant and authenticated Smart Embedder probes."""
        self.client = client
        self.interval = interval
        self.embedder_url = embedder_url.rstrip("/") if embedder_url else None
        self.probe_timeout = probe_timeout
        self._embedder_headers = (
            {"Authorization": f"Bearer {embedder_client_token}"}
            if embedder_client_token
            else {}
        )
        self.reachable = False
        self.collections = 0
        self.embedder_reachable = False
        self.embedder_profile: str | None = None
        self.embedder_models: dict[str, str | None] | None = None
        self.embedder_contract: dict | None = None
        self.embedder_version: str | None = None
        self.last_check: float | None = None
        self._task: asyncio.Task[None] | None = None

    async def probe_once(self) -> None:
        """Run one Qdrant + embedder probe and update cached status."""
        try:
            result = await self.client.get_collections()
            self.reachable = True
            self.collections = len(result.collections)
            qdrant_up.set(1)
        except Exception as exc:
            self.reachable = False
            qdrant_up.set(0)
            log.warning("Qdrant health probe failed: %s", exc)
        await self._probe_embedder()
        self.last_check = time.time()

    def _clear_embedder_model_fields(self) -> None:
        """Reset the cached model/contract fields without touching reachability."""
        self.embedder_profile = None
        self.embedder_models = None
        self.embedder_contract = None
        self.embedder_version = None

    async def _probe_embedder(self) -> None:
        """Probe the embedder's /health and /models endpoints and cache both."""
        if not self.embedder_url:
            self.embedder_reachable = False
            self._clear_embedder_model_fields()
            return
        try:
            async with httpx.AsyncClient(
                timeout=self.probe_timeout,
                headers=self._embedder_headers,
            ) as client:
                response = await client.get(f"{self.embedder_url}/health")
                self.embedder_reachable = response.status_code == 200
                if not self.embedder_reachable:
                    self._clear_embedder_model_fields()
                    return
                # /models is informational only (profile name, model list) and
                # every request path already reads it from this cache, not from
                # the embedder — any failure here, network or malformed
                # response alike, must never flip embedder_reachable back to
                # False after /health has already answered 200, or a blip on
                # this secondary call would wrongly gate off every Points/Query
                # operation via require_operational_readiness.
                try:
                    models_response = await client.get(f"{self.embedder_url}/models")
                    if models_response.status_code != 200:
                        self._clear_embedder_model_fields()
                        return
                    data = models_response.json()
                    embedder = data.get("embedder") or {}
                    self.embedder_profile = data.get("profile")
                    self.embedder_models = {
                        "dense": embedder.get("dense"),
                        "sparse": embedder.get("sparse"),
                        "colbert": embedder.get("colbert"),
                        "reranker": data.get("reranker"),
                    }
                    self.embedder_contract = data.get("embedding_contract")
                    self.embedder_version = data.get("version")
                except Exception as exc:
                    self._clear_embedder_model_fields()
                    log.warning("Embedder /models probe failed: %s", exc)
        except Exception as exc:
            self.embedder_reachable = False
            self._clear_embedder_model_fields()
            log.warning("Embedder health probe failed: %s", exc)

    async def _loop(self) -> None:
        """Run :meth:`probe_once` forever at the configured interval."""
        while True:
            await self.probe_once()
            await asyncio.sleep(self.interval)

    def start(self) -> None:
        """Start the periodic probe loop if it is not already running."""
        if self._task and not self._task.done():
            return
        self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        """Cancel and await the periodic probe loop."""
        if not self._task:
            return
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        finally:
            self._task = None
