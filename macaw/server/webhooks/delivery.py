"""Async webhook delivery with HMAC signing and retry."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json

import httpx

from macaw.logging import get_logger

logger = get_logger("server.webhooks")


class WebhookDelivery:
    """Delivers JSON payloads to webhook URLs with HMAC-SHA256 signing."""

    def __init__(self, max_retries: int = 3, retry_delay_s: float = 1.0) -> None:
        self._max_retries = max_retries
        self._retry_delay_s = retry_delay_s

    @staticmethod
    def compute_signature(payload: bytes, secret: str) -> str:
        """Compute HMAC-SHA256 signature for payload."""
        return hmac.new(secret.encode(), payload, hashlib.sha256).hexdigest()

    async def deliver(
        self,
        url: str,
        payload: dict[str, object],
        *,
        secret: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> bool:
        """Deliver JSON payload to webhook URL with retry.

        Fire-and-forget: failures are logged, not raised.
        Returns True if delivered successfully, False otherwise.
        """
        body = json.dumps(payload, default=str).encode()
        headers: dict[str, str] = {"Content-Type": "application/json"}

        if secret:
            sig = self.compute_signature(body, secret)
            headers["X-Macaw-Signature"] = sig

        for attempt in range(self._max_retries + 1):
            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    resp = await client.post(url, content=body, headers=headers)
                    if resp.status_code < 400:
                        logger.info(
                            "webhook_delivered",
                            url=url,
                            status=resp.status_code,
                            attempt=attempt,
                        )
                        return True
                    logger.warning(
                        "webhook_rejected",
                        url=url,
                        status=resp.status_code,
                        attempt=attempt,
                    )
            except Exception as exc:
                logger.warning("webhook_error", url=url, error=str(exc), attempt=attempt)

            if attempt < self._max_retries:
                delay = self._retry_delay_s * (2**attempt)  # exponential backoff
                await asyncio.sleep(delay)

        logger.error("webhook_delivery_failed", url=url, max_retries=self._max_retries)
        return False
