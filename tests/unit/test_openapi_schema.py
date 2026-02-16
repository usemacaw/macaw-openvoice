"""Tests for OpenAPI schema structural integrity.

Ensures that all expected endpoints are registered and key parameters
are exposed. Prevents accidental removal of endpoints or fields during
refactoring.
"""

from __future__ import annotations

from macaw.server.app import create_app

# Expected API paths that must always be present.
_EXPECTED_PATHS = {
    "/health",
    "/v1/audio/transcriptions",
    "/v1/audio/transcriptions/{request_id}/cancel",
    "/v1/audio/translations",
    "/v1/audio/speech",
    "/v1/realtime",
    "/v1/voices",
    "/v1/voices/{voice_id}",
    "/v1/models",
}


def _get_schema() -> dict[str, object]:
    app = create_app()
    return app.openapi()


class TestOpenAPIEndpoints:
    """All declared routes must appear in the OpenAPI schema."""

    def test_all_expected_paths_present(self) -> None:
        schema = _get_schema()
        paths = set(schema.get("paths", {}).keys())  # type: ignore[union-attr]
        missing = _EXPECTED_PATHS - paths
        assert not missing, f"Missing paths in OpenAPI schema: {sorted(missing)}"

    def test_no_unexpected_path_removals(self) -> None:
        """Guard against accidentally removing routes."""
        schema = _get_schema()
        paths = set(schema.get("paths", {}).keys())  # type: ignore[union-attr]
        # At least the expected paths should exist; additional paths are fine.
        assert len(paths) >= len(_EXPECTED_PATHS)


class TestTranscriptionEndpointParams:
    """POST /v1/audio/transcriptions must expose all OpenAI-compatible params."""

    def test_has_required_parameters(self) -> None:
        schema = _get_schema()
        paths: dict[str, object] = schema.get("paths", {})  # type: ignore[assignment]
        post = paths.get("/v1/audio/transcriptions", {}).get("post", {})  # type: ignore[union-attr]

        # Collect parameter/field names from the request body schema
        body = post.get("requestBody", {})  # type: ignore[union-attr]
        content = body.get("content", {})
        # multipart/form-data for file upload endpoints
        form_schema = content.get("multipart/form-data", {}).get("schema", {})

        # May be direct properties or use $ref — check both patterns
        props = _resolve_properties(schema, form_schema)
        param_names = set(props.keys())

        expected_params = {"file", "model", "language", "prompt", "response_format", "temperature"}
        missing = expected_params - param_names
        assert not missing, f"Missing transcription params: {sorted(missing)}"

    def test_has_timestamp_granularities(self) -> None:
        schema = _get_schema()
        paths: dict[str, object] = schema.get("paths", {})  # type: ignore[assignment]
        post = paths.get("/v1/audio/transcriptions", {}).get("post", {})  # type: ignore[union-attr]
        body = post.get("requestBody", {})  # type: ignore[union-attr]
        content = body.get("content", {})
        form_schema = content.get("multipart/form-data", {}).get("schema", {})
        props = _resolve_properties(schema, form_schema)

        assert "timestamp_granularities[]" in props or "timestamp_granularities" in props, (
            "timestamp_granularities not found in transcription endpoint params"
        )


class TestSpeechEndpointParams:
    """POST /v1/audio/speech must expose TTS parameters."""

    def test_has_required_parameters(self) -> None:
        schema = _get_schema()
        paths: dict[str, object] = schema.get("paths", {})  # type: ignore[assignment]
        post = paths.get("/v1/audio/speech", {}).get("post", {})  # type: ignore[union-attr]
        body = post.get("requestBody", {})  # type: ignore[union-attr]
        content = body.get("content", {})

        # Speech endpoint uses JSON body
        json_schema = content.get("application/json", {}).get("schema", {})
        props = _resolve_properties(schema, json_schema)
        param_names = set(props.keys())

        expected_params = {"model", "input", "voice"}
        missing = expected_params - param_names
        assert not missing, f"Missing speech params: {sorted(missing)}"


class TestOpenAPITags:
    """All routes must be grouped by OpenAPI tags."""

    def test_expected_tags_present(self) -> None:
        schema = _get_schema()
        paths: dict[str, object] = schema.get("paths", {})  # type: ignore[assignment]
        all_tags: set[str] = set()
        for ops in paths.values():
            for method_data in ops.values():  # type: ignore[union-attr]
                if isinstance(method_data, dict):
                    all_tags.update(method_data.get("tags", []))
        expected = {"Audio", "Voices", "System", "Realtime"}
        missing = expected - all_tags
        assert not missing, f"Missing OpenAPI tags: {sorted(missing)}"

    def test_audio_endpoints_tagged(self) -> None:
        schema = _get_schema()
        paths: dict[str, object] = schema.get("paths", {})  # type: ignore[assignment]
        for path in ("/v1/audio/transcriptions", "/v1/audio/translations", "/v1/audio/speech"):
            ops = paths.get(path, {})
            for method_data in ops.values():  # type: ignore[union-attr]
                if isinstance(method_data, dict) and "tags" in method_data:
                    assert "Audio" in method_data["tags"], f"{path} missing Audio tag"


class TestSchemaMetadata:
    """Schema metadata should match project configuration."""

    def test_title(self) -> None:
        schema = _get_schema()
        assert schema.get("info", {}).get("title") == "Macaw OpenVoice"  # type: ignore[union-attr]

    def test_version_present(self) -> None:
        schema = _get_schema()
        version = schema.get("info", {}).get("version")  # type: ignore[union-attr]
        assert version, "Schema version should not be empty"


# --- Helpers ---


def _resolve_properties(
    full_schema: dict[str, object],
    local_schema: dict[str, object],
) -> dict[str, object]:
    """Resolve JSON Schema properties, following $ref if needed."""
    if "properties" in local_schema:
        return local_schema["properties"]  # type: ignore[return-value]

    ref = local_schema.get("$ref")
    if isinstance(ref, str) and ref.startswith("#/"):
        parts = ref.lstrip("#/").split("/")
        node: object = full_schema
        for part in parts:
            if isinstance(node, dict):
                node = node.get(part, {})
            else:
                return {}
        if isinstance(node, dict) and "properties" in node:
            return node["properties"]  # type: ignore[return-value]

    # allOf pattern (Pydantic v2 can emit this)
    all_of = local_schema.get("allOf")
    if isinstance(all_of, list):
        merged: dict[str, object] = {}
        for sub in all_of:
            if isinstance(sub, dict):
                merged.update(_resolve_properties(full_schema, sub))
        return merged

    return {}
