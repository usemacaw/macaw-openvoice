"""Webhook configuration model."""

from __future__ import annotations

from pydantic import BaseModel, Field


class WebhookConfig(BaseModel):
    """Configuration for webhook delivery."""

    url: str = Field(..., description="Webhook URL (HTTPS required in production)")
    secret: str | None = Field(default=None, min_length=16, description="HMAC signing secret")
    max_retries: int = Field(default=3, ge=0, le=10)
    retry_delay_s: float = Field(default=1.0, gt=0)
