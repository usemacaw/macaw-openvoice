"""Unit tests for pronunciation dictionary CRUD REST endpoints.

Tests the routes defined in ``macaw/server/routes/pronunciation.py`` using
``httpx.AsyncClient`` with ``ASGITransport`` against the real FastAPI app.
The ``FileSystemPronunciationStore`` writes to a temporary directory
(cleaned up automatically by ``tmp_path``), so no mocks are needed for the
store itself.
"""

from __future__ import annotations

import httpx
from httpx import ASGITransport

from macaw.server.app import create_app


def _make_app(tmp_path_str: str) -> create_app:
    """Create an app whose pronunciation store points at *tmp_path_str*."""
    import os

    os.environ["MACAW_PRONUNCIATION_STORE_PATH"] = tmp_path_str
    try:
        from macaw.config.settings import get_settings

        get_settings.cache_clear()
        app = create_app()
    finally:
        os.environ.pop("MACAW_PRONUNCIATION_STORE_PATH", None)
        get_settings.cache_clear()
    return app


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BASE_URL = "http://test"

_ALIAS_RULE = {
    "string_to_match": "IEEE",
    "rule_type": "alias",
    "alias": "I triple E",
}

_PHONEME_RULE = {
    "string_to_match": "tomato",
    "rule_type": "phoneme",
    "phoneme": "t@meItoU",
    "alphabet": "ipa",
}


async def _create_dict(
    client: httpx.AsyncClient,
    *,
    name: str = "Test Dict",
    rules: list[dict[str, object]] | None = None,
    description: str = "",
) -> httpx.Response:
    payload: dict[str, object] = {"name": name, "description": description}
    if rules is not None:
        payload["rules"] = rules
    else:
        payload["rules"] = [_ALIAS_RULE]
    return await client.post(
        "/v1/pronunciation-dictionaries/add-from-rules",
        json=payload,
    )


# ---------------------------------------------------------------------------
# TestCreateDictionary
# ---------------------------------------------------------------------------


class TestCreateDictionary:
    """POST /v1/pronunciation-dictionaries/add-from-rules"""

    async def test_create_with_alias_rules_returns_201(self, tmp_path: object) -> None:
        app = _make_app(str(tmp_path))
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url=BASE_URL) as client:
            response = await _create_dict(client, name="My Dict", rules=[_ALIAS_RULE])

        assert response.status_code == 201
        body = response.json()
        assert body["name"] == "My Dict"
        assert "dictionary_id" in body
        assert "version_id" in body
        assert "created_at" in body
        assert len(body["rules"]) == 1
        assert body["rules"][0]["string_to_match"] == "IEEE"
        assert body["rules"][0]["rule_type"] == "alias"
        assert body["rules"][0]["alias"] == "I triple E"

    async def test_create_without_name_returns_400(self, tmp_path: object) -> None:
        app = _make_app(str(tmp_path))
        transport = ASGITransport(app=app, raise_app_exceptions=False)
        async with httpx.AsyncClient(transport=transport, base_url=BASE_URL) as client:
            response = await client.post(
                "/v1/pronunciation-dictionaries/add-from-rules",
                json={"rules": [_ALIAS_RULE]},
            )

        assert response.status_code == 400
        body = response.json()
        assert "name" in body["error"]["message"].lower()

    async def test_create_with_empty_rules_is_valid(self, tmp_path: object) -> None:
        app = _make_app(str(tmp_path))
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url=BASE_URL) as client:
            response = await _create_dict(client, name="Empty Rules", rules=[])

        assert response.status_code == 201
        body = response.json()
        assert body["name"] == "Empty Rules"
        assert body["rules"] == []

    async def test_create_validates_rule_type(self, tmp_path: object) -> None:
        app = _make_app(str(tmp_path))
        transport = ASGITransport(app=app, raise_app_exceptions=False)
        async with httpx.AsyncClient(transport=transport, base_url=BASE_URL) as client:
            response = await client.post(
                "/v1/pronunciation-dictionaries/add-from-rules",
                json={
                    "name": "Bad Rule Type",
                    "rules": [
                        {
                            "string_to_match": "foo",
                            "rule_type": "invalid_type",
                            "alias": "bar",
                        }
                    ],
                },
            )

        # Pydantic validation error surfaces as 422 or 500 (depending on error
        # handler mapping).  The key assertion is that it does NOT succeed.
        assert response.status_code >= 400

    async def test_create_with_name_exceeding_max_length_returns_400(
        self, tmp_path: object
    ) -> None:
        app = _make_app(str(tmp_path))
        transport = ASGITransport(app=app, raise_app_exceptions=False)
        async with httpx.AsyncClient(transport=transport, base_url=BASE_URL) as client:
            response = await _create_dict(client, name="x" * 201, rules=[_ALIAS_RULE])

        assert response.status_code == 400
        body = response.json()
        assert "200" in body["error"]["message"]


# ---------------------------------------------------------------------------
# TestListDictionaries
# ---------------------------------------------------------------------------


class TestListDictionaries:
    """GET /v1/pronunciation-dictionaries"""

    async def test_list_empty_returns_empty_list(self, tmp_path: object) -> None:
        app = _make_app(str(tmp_path))
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url=BASE_URL) as client:
            response = await client.get("/v1/pronunciation-dictionaries")

        assert response.status_code == 200
        assert response.json() == []

    async def test_list_returns_created_dictionaries(self, tmp_path: object) -> None:
        app = _make_app(str(tmp_path))
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url=BASE_URL) as client:
            await _create_dict(client, name="Dict A", rules=[_ALIAS_RULE])
            await _create_dict(client, name="Dict B", rules=[_PHONEME_RULE])

            response = await client.get("/v1/pronunciation-dictionaries")

        assert response.status_code == 200
        body = response.json()
        assert len(body) == 2
        names = {d["name"] for d in body}
        assert names == {"Dict A", "Dict B"}


# ---------------------------------------------------------------------------
# TestGetDictionary
# ---------------------------------------------------------------------------


class TestGetDictionary:
    """GET /v1/pronunciation-dictionaries/{dictionary_id}"""

    async def test_get_existing_returns_dictionary(self, tmp_path: object) -> None:
        app = _make_app(str(tmp_path))
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url=BASE_URL) as client:
            create_resp = await _create_dict(client, name="Fetch Me")
            dict_id = create_resp.json()["dictionary_id"]

            response = await client.get(f"/v1/pronunciation-dictionaries/{dict_id}")

        assert response.status_code == 200
        body = response.json()
        assert body["dictionary_id"] == dict_id
        assert body["name"] == "Fetch Me"

    async def test_get_nonexistent_returns_404(self, tmp_path: object) -> None:
        app = _make_app(str(tmp_path))
        transport = ASGITransport(app=app, raise_app_exceptions=False)
        async with httpx.AsyncClient(transport=transport, base_url=BASE_URL) as client:
            response = await client.get("/v1/pronunciation-dictionaries/nonexistent-id-000")

        assert response.status_code == 404
        body = response.json()
        assert "nonexistent-id-000" in body["error"]["message"]


# ---------------------------------------------------------------------------
# TestDeleteDictionary
# ---------------------------------------------------------------------------


class TestDeleteDictionary:
    """DELETE /v1/pronunciation-dictionaries/{dictionary_id}"""

    async def test_delete_existing_returns_success(self, tmp_path: object) -> None:
        app = _make_app(str(tmp_path))
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url=BASE_URL) as client:
            create_resp = await _create_dict(client, name="Delete Me")
            dict_id = create_resp.json()["dictionary_id"]

            response = await client.delete(f"/v1/pronunciation-dictionaries/{dict_id}")

        assert response.status_code == 200
        body = response.json()
        assert body["deleted"] is True
        assert body["dictionary_id"] == dict_id

    async def test_delete_nonexistent_returns_404(self, tmp_path: object) -> None:
        app = _make_app(str(tmp_path))
        transport = ASGITransport(app=app, raise_app_exceptions=False)
        async with httpx.AsyncClient(transport=transport, base_url=BASE_URL) as client:
            response = await client.delete("/v1/pronunciation-dictionaries/nonexistent-id-000")

        assert response.status_code == 404
        body = response.json()
        assert "nonexistent-id-000" in body["error"]["message"]

    async def test_delete_removes_dictionary_from_list(self, tmp_path: object) -> None:
        app = _make_app(str(tmp_path))
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url=BASE_URL) as client:
            create_resp = await _create_dict(client, name="Ephemeral")
            dict_id = create_resp.json()["dictionary_id"]

            await client.delete(f"/v1/pronunciation-dictionaries/{dict_id}")

            list_resp = await client.get("/v1/pronunciation-dictionaries")

        assert list_resp.status_code == 200
        assert list_resp.json() == []


# ---------------------------------------------------------------------------
# TestAddRules
# ---------------------------------------------------------------------------


class TestAddRules:
    """POST /v1/pronunciation-dictionaries/{dictionary_id}/rules"""

    async def test_add_rules_appends_to_existing(self, tmp_path: object) -> None:
        app = _make_app(str(tmp_path))
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url=BASE_URL) as client:
            create_resp = await _create_dict(client, name="Grow Me", rules=[_ALIAS_RULE])
            dict_id = create_resp.json()["dictionary_id"]
            original_version = create_resp.json()["version_id"]

            response = await client.post(
                f"/v1/pronunciation-dictionaries/{dict_id}/rules",
                json={"rules": [_PHONEME_RULE]},
            )

        assert response.status_code == 200
        body = response.json()
        assert len(body["rules"]) == 2
        match_strings = [r["string_to_match"] for r in body["rules"]]
        assert "IEEE" in match_strings
        assert "tomato" in match_strings
        # version_id must be regenerated
        assert body["version_id"] != original_version

    async def test_add_rules_nonexistent_returns_404(self, tmp_path: object) -> None:
        app = _make_app(str(tmp_path))
        transport = ASGITransport(app=app, raise_app_exceptions=False)
        async with httpx.AsyncClient(transport=transport, base_url=BASE_URL) as client:
            response = await client.post(
                "/v1/pronunciation-dictionaries/nonexistent-id-000/rules",
                json={"rules": [_ALIAS_RULE]},
            )

        assert response.status_code == 404
        body = response.json()
        assert "nonexistent-id-000" in body["error"]["message"]

    async def test_add_rules_with_empty_list_returns_400(self, tmp_path: object) -> None:
        app = _make_app(str(tmp_path))
        transport = ASGITransport(app=app, raise_app_exceptions=False)
        async with httpx.AsyncClient(transport=transport, base_url=BASE_URL) as client:
            create_resp = await _create_dict(client, name="No Empty")
            dict_id = create_resp.json()["dictionary_id"]

            response = await client.post(
                f"/v1/pronunciation-dictionaries/{dict_id}/rules",
                json={"rules": []},
            )

        assert response.status_code == 400


# ---------------------------------------------------------------------------
# TestRemoveRules
# ---------------------------------------------------------------------------


class TestRemoveRules:
    """DELETE /v1/pronunciation-dictionaries/{dictionary_id}/rules"""

    async def test_remove_rules_by_string_match(self, tmp_path: object) -> None:
        app = _make_app(str(tmp_path))
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url=BASE_URL) as client:
            create_resp = await _create_dict(
                client,
                name="Shrink Me",
                rules=[_ALIAS_RULE, _PHONEME_RULE],
            )
            dict_id = create_resp.json()["dictionary_id"]
            original_version = create_resp.json()["version_id"]

            response = await client.request(
                "DELETE",
                f"/v1/pronunciation-dictionaries/{dict_id}/rules",
                json={"rule_strings": ["IEEE"]},
            )

        assert response.status_code == 200
        body = response.json()
        assert len(body["rules"]) == 1
        assert body["rules"][0]["string_to_match"] == "tomato"
        # version_id must be regenerated
        assert body["version_id"] != original_version

    async def test_remove_rules_nonexistent_returns_404(self, tmp_path: object) -> None:
        app = _make_app(str(tmp_path))
        transport = ASGITransport(app=app, raise_app_exceptions=False)
        async with httpx.AsyncClient(transport=transport, base_url=BASE_URL) as client:
            response = await client.request(
                "DELETE",
                "/v1/pronunciation-dictionaries/nonexistent-id-000/rules",
                json={"rule_strings": ["anything"]},
            )

        assert response.status_code == 404
        body = response.json()
        assert "nonexistent-id-000" in body["error"]["message"]

    async def test_remove_rules_with_empty_list_returns_400(self, tmp_path: object) -> None:
        app = _make_app(str(tmp_path))
        transport = ASGITransport(app=app, raise_app_exceptions=False)
        async with httpx.AsyncClient(transport=transport, base_url=BASE_URL) as client:
            create_resp = await _create_dict(client, name="No Empty Remove")
            dict_id = create_resp.json()["dictionary_id"]

            response = await client.request(
                "DELETE",
                f"/v1/pronunciation-dictionaries/{dict_id}/rules",
                json={"rule_strings": []},
            )

        assert response.status_code == 400
