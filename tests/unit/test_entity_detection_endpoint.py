"""Tests for entity detection wiring in the transcription endpoint.

Verifies DetectedEntityResponse model, verbose_json response with entities,
backward compatibility, and category filtering through the formatter.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx

from macaw._types import BatchResult, ResponseFormat, SegmentDetail
from macaw.server.app import create_app
from macaw.server.formatters import format_response
from macaw.server.models.responses import DetectedEntityResponse, VerboseTranscriptionResponse


def _make_mock_registry() -> MagicMock:
    registry = MagicMock()
    registry.has_model.return_value = True
    registry.get_manifest.return_value = MagicMock()
    return registry


def _make_mock_scheduler(text: str = "hello world") -> MagicMock:
    scheduler = MagicMock()
    scheduler.transcribe = AsyncMock(
        return_value=BatchResult(
            text=text,
            language="en",
            duration=2.0,
            segments=(SegmentDetail(id=0, start=0.0, end=2.0, text=text),),
        )
    )
    return scheduler


class TestDetectedEntityResponseModel:
    """DetectedEntityResponse Pydantic model serialization."""

    def test_serializes_all_fields(self) -> None:
        entity = DetectedEntityResponse(
            text="test@example.com",
            entity_type="email_address",
            category="pii",
            start_char=0,
            end_char=16,
        )
        data = entity.model_dump()
        assert data["text"] == "test@example.com"
        assert data["entity_type"] == "email_address"
        assert data["category"] == "pii"
        assert data["start_char"] == 0
        assert data["end_char"] == 16

    def test_roundtrip_from_dict(self) -> None:
        raw = {
            "text": "192.168.1.1",
            "entity_type": "ip_address",
            "category": "pci",
            "start_char": 5,
            "end_char": 16,
        }
        entity = DetectedEntityResponse(**raw)
        assert entity.text == "192.168.1.1"
        assert entity.category == "pci"


class TestVerboseResponseWithEntities:
    """VerboseTranscriptionResponse includes optional entities field."""

    def test_entities_default_is_none(self) -> None:
        resp = VerboseTranscriptionResponse(language="en", duration=1.0, text="hello")
        assert resp.entities is None

    def test_entities_serialized_when_present(self) -> None:
        resp = VerboseTranscriptionResponse(
            language="en",
            duration=1.0,
            text="hello",
            entities=[
                DetectedEntityResponse(
                    text="info@test.com",
                    entity_type="email_address",
                    category="pii",
                    start_char=0,
                    end_char=13,
                )
            ],
        )
        data = resp.model_dump()
        assert len(data["entities"]) == 1
        assert data["entities"][0]["entity_type"] == "email_address"


class TestFormatResponseEntityDetection:
    """format_response wires entity detection into verbose_json."""

    def _make_result(self, text: str) -> BatchResult:
        return BatchResult(
            text=text,
            language="en",
            duration=1.0,
            segments=(SegmentDetail(id=0, start=0.0, end=1.0, text=text),),
        )

    def test_no_entity_detection_omits_entities_key(self) -> None:
        result = self._make_result("hello world")
        response: dict[str, Any] = format_response(
            result, ResponseFormat.VERBOSE_JSON, entity_detection=None
        )
        assert "entities" not in response

    def test_entity_detection_all_includes_entities(self) -> None:
        result = self._make_result("Contact info@test.com")
        response: dict[str, Any] = format_response(
            result, ResponseFormat.VERBOSE_JSON, entity_detection=["all"]
        )
        assert "entities" in response
        assert len(response["entities"]) == 1
        assert response["entities"][0]["entity_type"] == "email_address"

    def test_entity_detection_pii_filters_correctly(self) -> None:
        result = self._make_result("Email info@test.com IP 10.0.0.1")
        response: dict[str, Any] = format_response(
            result, ResponseFormat.VERBOSE_JSON, entity_detection=["pii"]
        )
        assert "entities" in response
        assert all(e["category"] == "pii" for e in response["entities"])

    def test_entity_detection_with_json_format_is_ignored(self) -> None:
        result = self._make_result("Contact info@test.com")
        response: dict[str, Any] = format_response(
            result, ResponseFormat.JSON, entity_detection=["all"]
        )
        assert "entities" not in response

    def test_entity_detection_no_matches_returns_empty_list(self) -> None:
        result = self._make_result("Just plain text here")
        response: dict[str, Any] = format_response(
            result, ResponseFormat.VERBOSE_JSON, entity_detection=["all"]
        )
        assert response["entities"] == []

    def test_entity_detection_multiple_entities(self) -> None:
        result = self._make_result("Email info@test.com SSN 123-45-6789")
        response: dict[str, Any] = format_response(
            result, ResponseFormat.VERBOSE_JSON, entity_detection=["pii"]
        )
        assert len(response["entities"]) == 2
        types = {e["entity_type"] for e in response["entities"]}
        assert "email_address" in types
        assert "ssn" in types


class TestTranscriptionEndpointEntityDetection:
    """Integration test for entity_detection Form param on the endpoint."""

    async def test_endpoint_without_entity_detection_has_no_entities(self) -> None:
        scheduler = _make_mock_scheduler("hello world")
        app = create_app(registry=_make_mock_registry(), scheduler=scheduler)

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            response = await client.post(
                "/v1/audio/transcriptions",
                data={"model": "test-model", "response_format": "verbose_json"},
                files={"file": ("test.wav", b"fake-audio", "audio/wav")},
            )

        assert response.status_code == 200
        data = response.json()
        assert "entities" not in data

    async def test_endpoint_with_entity_detection_returns_entities(self) -> None:
        scheduler = _make_mock_scheduler("Contact info@test.com today")
        app = create_app(registry=_make_mock_registry(), scheduler=scheduler)

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            response = await client.post(
                "/v1/audio/transcriptions",
                data={
                    "model": "test-model",
                    "response_format": "verbose_json",
                    "entity_detection": "all",
                },
                files={"file": ("test.wav", b"fake-audio", "audio/wav")},
            )

        assert response.status_code == 200
        data = response.json()
        assert "entities" in data
        assert len(data["entities"]) == 1
        assert data["entities"][0]["entity_type"] == "email_address"

    async def test_endpoint_entity_detection_comma_separated(self) -> None:
        scheduler = _make_mock_scheduler("Email info@test.com MRN: 12345 IP 10.0.0.1")
        app = create_app(registry=_make_mock_registry(), scheduler=scheduler)

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            response = await client.post(
                "/v1/audio/transcriptions",
                data={
                    "model": "test-model",
                    "response_format": "verbose_json",
                    "entity_detection": "pii,phi",
                },
                files={"file": ("test.wav", b"fake-audio", "audio/wav")},
            )

        assert response.status_code == 200
        data = response.json()
        assert "entities" in data
        categories = {e["category"] for e in data["entities"]}
        assert "pci" not in categories

    async def test_endpoint_json_format_ignores_entity_detection(self) -> None:
        scheduler = _make_mock_scheduler("Contact info@test.com")
        app = create_app(registry=_make_mock_registry(), scheduler=scheduler)

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            response = await client.post(
                "/v1/audio/transcriptions",
                data={
                    "model": "test-model",
                    "response_format": "json",
                    "entity_detection": "all",
                },
                files={"file": ("test.wav", b"fake-audio", "audio/wav")},
            )

        assert response.status_code == 200
        data = response.json()
        assert "entities" not in data
