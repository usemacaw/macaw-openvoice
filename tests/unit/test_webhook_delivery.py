"""Tests for webhook configuration and delivery module.

Covers WebhookConfig validation, HMAC-SHA256 signing, retry logic,
exponential backoff, and fire-and-forget semantics.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import ValidationError

from macaw.server.webhooks.config import WebhookConfig
from macaw.server.webhooks.delivery import WebhookDelivery


class TestWebhookConfig:
    """WebhookConfig Pydantic model validation."""

    def test_valid_config(self) -> None:
        config = WebhookConfig(
            url="https://example.com/hook",
            secret="this-is-a-long-secret",  # pragma: allowlist secret
            max_retries=5,
            retry_delay_s=2.0,
        )
        assert config.url == "https://example.com/hook"
        assert config.secret == "this-is-a-long-secret"  # pragma: allowlist secret
        assert config.max_retries == 5
        assert config.retry_delay_s == 2.0

    def test_url_required(self) -> None:
        with pytest.raises(ValidationError):
            WebhookConfig()  # type: ignore[call-arg]

    def test_secret_min_length(self) -> None:
        with pytest.raises(ValidationError):
            WebhookConfig(url="https://example.com", secret="short")  # pragma: allowlist secret

    def test_secret_none_allowed(self) -> None:
        config = WebhookConfig(url="https://example.com")
        assert config.secret is None

    def test_max_retries_bounds(self) -> None:
        with pytest.raises(ValidationError):
            WebhookConfig(url="https://example.com", max_retries=11)

    def test_retry_delay_positive(self) -> None:
        with pytest.raises(ValidationError):
            WebhookConfig(url="https://example.com", retry_delay_s=0)

    def test_defaults(self) -> None:
        config = WebhookConfig(url="https://example.com")
        assert config.max_retries == 3
        assert config.retry_delay_s == 1.0


class TestComputeSignature:
    """WebhookDelivery.compute_signature HMAC-SHA256 correctness."""

    def test_returns_valid_hmac_sha256(self) -> None:
        payload = b'{"text": "hello"}'
        secret = "my-webhook-secret"  # pragma: allowlist secret
        sig = WebhookDelivery.compute_signature(payload, secret)

        expected = hmac.new(secret.encode(), payload, hashlib.sha256).hexdigest()
        assert sig == expected

    def test_different_payloads_different_signatures(self) -> None:
        secret = "my-webhook-secret"  # pragma: allowlist secret
        sig1 = WebhookDelivery.compute_signature(b"payload1", secret)
        sig2 = WebhookDelivery.compute_signature(b"payload2", secret)
        assert sig1 != sig2

    def test_different_secrets_different_signatures(self) -> None:
        payload = b"same-payload"
        sig1 = WebhookDelivery.compute_signature(payload, "secret-one-abcdefg")
        sig2 = WebhookDelivery.compute_signature(payload, "secret-two-abcdefg")
        assert sig1 != sig2


class TestDeliverSuccess:
    """WebhookDelivery.deliver() on successful delivery."""

    async def test_returns_true_on_success(self) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 200

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        delivery = WebhookDelivery(max_retries=0)

        with patch("macaw.server.webhooks.delivery.httpx") as mock_httpx:
            mock_httpx.AsyncClient.return_value = mock_client
            result = await delivery.deliver(
                url="https://example.com/hook",
                payload={"text": "hello"},
            )

        assert result is True

    async def test_sends_correct_content_type(self) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 200

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        delivery = WebhookDelivery(max_retries=0)

        with patch("macaw.server.webhooks.delivery.httpx") as mock_httpx:
            mock_httpx.AsyncClient.return_value = mock_client
            await delivery.deliver(
                url="https://example.com/hook",
                payload={"text": "hello"},
            )

        call_kwargs = mock_client.post.call_args
        headers = call_kwargs.kwargs["headers"]
        assert headers["Content-Type"] == "application/json"

    async def test_includes_signature_header_when_secret_provided(self) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 200

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        delivery = WebhookDelivery(max_retries=0)
        secret = "my-signing-secret-key"  # pragma: allowlist secret

        with patch("macaw.server.webhooks.delivery.httpx") as mock_httpx:
            mock_httpx.AsyncClient.return_value = mock_client
            await delivery.deliver(
                url="https://example.com/hook",
                payload={"text": "hello"},
                secret=secret,
            )

        call_kwargs = mock_client.post.call_args
        headers = call_kwargs.kwargs["headers"]
        assert "X-Macaw-Signature" in headers

        # Verify the signature matches expected HMAC
        body = call_kwargs.kwargs["content"]
        expected_sig = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
        assert headers["X-Macaw-Signature"] == expected_sig

    async def test_no_signature_header_without_secret(self) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 200

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        delivery = WebhookDelivery(max_retries=0)

        with patch("macaw.server.webhooks.delivery.httpx") as mock_httpx:
            mock_httpx.AsyncClient.return_value = mock_client
            await delivery.deliver(
                url="https://example.com/hook",
                payload={"text": "hello"},
            )

        call_kwargs = mock_client.post.call_args
        headers = call_kwargs.kwargs["headers"]
        assert "X-Macaw-Signature" not in headers

    async def test_sends_json_body(self) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 200

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        delivery = WebhookDelivery(max_retries=0)
        payload = {"text": "hello", "status": "completed"}

        with patch("macaw.server.webhooks.delivery.httpx") as mock_httpx:
            mock_httpx.AsyncClient.return_value = mock_client
            await delivery.deliver(
                url="https://example.com/hook",
                payload=payload,
            )

        call_kwargs = mock_client.post.call_args
        body = call_kwargs.kwargs["content"]
        parsed = json.loads(body)
        assert parsed == payload


class TestDeliverRetry:
    """WebhookDelivery.deliver() retry and backoff behavior."""

    async def test_retries_on_server_error(self) -> None:
        error_response = MagicMock()
        error_response.status_code = 500

        success_response = MagicMock()
        success_response.status_code = 200

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=[error_response, success_response])
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        delivery = WebhookDelivery(max_retries=1, retry_delay_s=0.001)

        with patch("macaw.server.webhooks.delivery.httpx") as mock_httpx:
            mock_httpx.AsyncClient.return_value = mock_client
            result = await delivery.deliver(
                url="https://example.com/hook",
                payload={"text": "hello"},
            )

        assert result is True
        assert mock_client.post.call_count == 2

    async def test_retries_on_exception(self) -> None:
        success_response = MagicMock()
        success_response.status_code = 200

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(
            side_effect=[ConnectionError("network error"), success_response]
        )
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        delivery = WebhookDelivery(max_retries=1, retry_delay_s=0.001)

        with patch("macaw.server.webhooks.delivery.httpx") as mock_httpx:
            mock_httpx.AsyncClient.return_value = mock_client
            result = await delivery.deliver(
                url="https://example.com/hook",
                payload={"text": "hello"},
            )

        assert result is True

    async def test_returns_false_after_max_retries(self) -> None:
        error_response = MagicMock()
        error_response.status_code = 500

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=error_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        delivery = WebhookDelivery(max_retries=2, retry_delay_s=0.001)

        with patch("macaw.server.webhooks.delivery.httpx") as mock_httpx:
            mock_httpx.AsyncClient.return_value = mock_client
            result = await delivery.deliver(
                url="https://example.com/hook",
                payload={"text": "hello"},
            )

        assert result is False
        # Initial attempt + 2 retries = 3 total
        assert mock_client.post.call_count == 3

    async def test_fire_and_forget_no_exception_raised(self) -> None:
        """deliver() never raises, even on persistent failures."""
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=ConnectionError("permanent failure"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        delivery = WebhookDelivery(max_retries=1, retry_delay_s=0.001)

        with patch("macaw.server.webhooks.delivery.httpx") as mock_httpx:
            mock_httpx.AsyncClient.return_value = mock_client
            # Should not raise
            result = await delivery.deliver(
                url="https://example.com/hook",
                payload={"text": "hello"},
            )

        assert result is False

    async def test_exponential_backoff(self) -> None:
        """Verify that retry delays increase exponentially."""
        error_response = MagicMock()
        error_response.status_code = 500

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=error_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        delivery = WebhookDelivery(max_retries=2, retry_delay_s=1.0)
        sleep_calls: list[float] = []

        async def mock_sleep(delay: float) -> None:
            sleep_calls.append(delay)

        with (
            patch("macaw.server.webhooks.delivery.httpx") as mock_httpx,
            patch("macaw.server.webhooks.delivery.asyncio.sleep", side_effect=mock_sleep),
        ):
            mock_httpx.AsyncClient.return_value = mock_client
            await delivery.deliver(
                url="https://example.com/hook",
                payload={"text": "hello"},
            )

        # First retry: 1.0 * 2^0 = 1.0
        # Second retry: 1.0 * 2^1 = 2.0
        assert len(sleep_calls) == 2
        assert sleep_calls[0] == pytest.approx(1.0)
        assert sleep_calls[1] == pytest.approx(2.0)


class TestWebhookSettings:
    """WebhookSettings in MacawSettings."""

    def test_webhook_settings_in_macaw_settings(self) -> None:
        from macaw.config.settings import MacawSettings, WebhookSettings

        settings = MacawSettings()
        assert isinstance(settings.webhook, WebhookSettings)
        assert settings.webhook.max_retries == 3
        assert settings.webhook.retry_delay_s == 1.0
        assert settings.webhook.allowed_schemes == "https"
