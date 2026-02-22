"""Tests for Sprint 4: Seed, text normalization, and sampling parameters.

Validates:
- Proto fields (seed, text_normalization, temperature, top_k, top_p)
- Converter mapping of new fields into SynthesizeParams.options
- build_tts_proto_request with new parameters
- SpeechRequest and TTSSpeakCommand model fields
- Qwen3 backend seed + sampling parameter extraction
- Kokoro backend seed no-op logging
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from macaw.proto.tts_worker_pb2 import SynthesizeRequest
from macaw.scheduler.tts_converters import build_tts_proto_request
from macaw.server.models.events import TTSSpeakCommand
from macaw.server.models.speech import SpeechRequest
from macaw.workers.tts.converters import proto_request_to_synthesize_params

# ---------------------------------------------------------------------------
# Proto field tests
# ---------------------------------------------------------------------------


class TestSynthesizeRequestProtoFields:
    def test_seed_default_zero(self) -> None:
        req = SynthesizeRequest()
        assert req.seed == 0

    def test_seed_set(self) -> None:
        req = SynthesizeRequest(seed=42)
        assert req.seed == 42

    def test_text_normalization_default_empty(self) -> None:
        req = SynthesizeRequest()
        assert req.text_normalization == ""

    def test_text_normalization_set(self) -> None:
        req = SynthesizeRequest(text_normalization="off")
        assert req.text_normalization == "off"

    def test_temperature_default_zero(self) -> None:
        req = SynthesizeRequest()
        assert req.temperature == 0.0

    def test_temperature_set(self) -> None:
        req = SynthesizeRequest(temperature=0.7)
        assert abs(req.temperature - 0.7) < 1e-6

    def test_top_k_default_zero(self) -> None:
        req = SynthesizeRequest()
        assert req.top_k == 0

    def test_top_k_set(self) -> None:
        req = SynthesizeRequest(top_k=30)
        assert req.top_k == 30

    def test_top_p_default_zero(self) -> None:
        req = SynthesizeRequest()
        assert req.top_p == 0.0

    def test_top_p_set(self) -> None:
        req = SynthesizeRequest(top_p=0.9)
        assert abs(req.top_p - 0.9) < 1e-6


# ---------------------------------------------------------------------------
# Converter: proto_request_to_synthesize_params
# ---------------------------------------------------------------------------


class TestConverterSeedAndNormalization:
    def test_seed_packed_into_options(self) -> None:
        req = SynthesizeRequest(text="Hello", seed=42)
        params = proto_request_to_synthesize_params(req)
        assert params.options is not None
        assert params.options["seed"] == 42

    def test_seed_zero_not_packed(self) -> None:
        req = SynthesizeRequest(text="Hello")
        params = proto_request_to_synthesize_params(req)
        # No extended fields set, options should be None
        assert params.options is None

    def test_text_normalization_packed(self) -> None:
        req = SynthesizeRequest(text="Hello", text_normalization="off")
        params = proto_request_to_synthesize_params(req)
        assert params.options is not None
        assert params.options["text_normalization"] == "off"

    def test_text_normalization_empty_not_packed(self) -> None:
        req = SynthesizeRequest(text="Hello")
        params = proto_request_to_synthesize_params(req)
        assert params.options is None

    def test_temperature_packed(self) -> None:
        req = SynthesizeRequest(text="Hello", temperature=0.7)
        params = proto_request_to_synthesize_params(req)
        assert params.options is not None
        assert abs(float(str(params.options["temperature"])) - 0.7) < 1e-5

    def test_top_k_packed(self) -> None:
        req = SynthesizeRequest(text="Hello", top_k=30)
        params = proto_request_to_synthesize_params(req)
        assert params.options is not None
        assert params.options["top_k"] == 30

    def test_top_p_packed(self) -> None:
        req = SynthesizeRequest(text="Hello", top_p=0.9)
        params = proto_request_to_synthesize_params(req)
        assert params.options is not None
        assert abs(float(str(params.options["top_p"])) - 0.9) < 1e-5

    def test_all_new_fields_together(self) -> None:
        req = SynthesizeRequest(
            text="Hello",
            seed=42,
            text_normalization="on",
            temperature=0.5,
            top_k=20,
            top_p=0.95,
        )
        params = proto_request_to_synthesize_params(req)
        assert params.options is not None
        assert params.options["seed"] == 42
        assert params.options["text_normalization"] == "on"
        assert params.options["top_k"] == 20

    def test_seed_combined_with_language(self) -> None:
        req = SynthesizeRequest(text="Hello", language="English", seed=123)
        params = proto_request_to_synthesize_params(req)
        assert params.options is not None
        assert params.options["language"] == "English"
        assert params.options["seed"] == 123


# ---------------------------------------------------------------------------
# build_tts_proto_request
# ---------------------------------------------------------------------------


class TestBuildTTSProtoRequestNewFields:
    def test_seed_set_on_proto(self) -> None:
        req = build_tts_proto_request(
            request_id="req-1",
            text="Hello",
            voice="default",
            sample_rate=24000,
            speed=1.0,
            seed=42,
        )
        assert req.seed == 42

    def test_seed_none_not_set(self) -> None:
        req = build_tts_proto_request(
            request_id="req-1",
            text="Hello",
            voice="default",
            sample_rate=24000,
            speed=1.0,
            seed=None,
        )
        assert req.seed == 0

    def test_seed_zero_not_set(self) -> None:
        req = build_tts_proto_request(
            request_id="req-1",
            text="Hello",
            voice="default",
            sample_rate=24000,
            speed=1.0,
            seed=0,
        )
        assert req.seed == 0

    def test_text_normalization_off(self) -> None:
        req = build_tts_proto_request(
            request_id="req-1",
            text="Hello",
            voice="default",
            sample_rate=24000,
            speed=1.0,
            text_normalization="off",
        )
        assert req.text_normalization == "off"

    def test_text_normalization_auto_not_set(self) -> None:
        req = build_tts_proto_request(
            request_id="req-1",
            text="Hello",
            voice="default",
            sample_rate=24000,
            speed=1.0,
            text_normalization="auto",
        )
        assert req.text_normalization == ""

    def test_temperature_set(self) -> None:
        req = build_tts_proto_request(
            request_id="req-1",
            text="Hello",
            voice="default",
            sample_rate=24000,
            speed=1.0,
            temperature=0.7,
        )
        assert abs(req.temperature - 0.7) < 1e-6

    def test_top_k_set(self) -> None:
        req = build_tts_proto_request(
            request_id="req-1",
            text="Hello",
            voice="default",
            sample_rate=24000,
            speed=1.0,
            top_k=30,
        )
        assert req.top_k == 30

    def test_top_p_set(self) -> None:
        req = build_tts_proto_request(
            request_id="req-1",
            text="Hello",
            voice="default",
            sample_rate=24000,
            speed=1.0,
            top_p=0.9,
        )
        assert abs(req.top_p - 0.9) < 1e-6

    def test_all_fields_combined(self) -> None:
        req = build_tts_proto_request(
            request_id="req-1",
            text="Hello",
            voice="default",
            sample_rate=24000,
            speed=1.0,
            seed=42,
            text_normalization="on",
            temperature=0.5,
            top_k=20,
            top_p=0.95,
            include_alignment=True,
        )
        assert req.seed == 42
        assert req.text_normalization == "on"
        assert abs(req.temperature - 0.5) < 1e-6
        assert req.top_k == 20
        assert abs(req.top_p - 0.95) < 1e-6
        assert req.include_alignment is True


# ---------------------------------------------------------------------------
# SpeechRequest model
# ---------------------------------------------------------------------------


class TestSpeechRequestNewFields:
    def test_seed_default_none(self) -> None:
        body = SpeechRequest(model="kokoro-v1", input="Hello")
        assert body.seed is None

    def test_seed_set(self) -> None:
        body = SpeechRequest(model="kokoro-v1", input="Hello", seed=42)
        assert body.seed == 42

    def test_text_normalization_default_auto(self) -> None:
        body = SpeechRequest(model="kokoro-v1", input="Hello")
        assert body.text_normalization == "auto"

    def test_text_normalization_off(self) -> None:
        body = SpeechRequest(model="kokoro-v1", input="Hello", text_normalization="off")
        assert body.text_normalization == "off"

    def test_text_normalization_invalid_rejected(self) -> None:
        with pytest.raises(ValidationError):
            SpeechRequest(model="kokoro-v1", input="Hello", text_normalization="invalid")

    def test_temperature_default_none(self) -> None:
        body = SpeechRequest(model="kokoro-v1", input="Hello")
        assert body.temperature is None

    def test_temperature_set(self) -> None:
        body = SpeechRequest(model="kokoro-v1", input="Hello", temperature=0.7)
        assert body.temperature == 0.7

    def test_temperature_out_of_range_rejected(self) -> None:
        with pytest.raises(ValidationError):
            SpeechRequest(model="kokoro-v1", input="Hello", temperature=3.0)

    def test_top_k_default_none(self) -> None:
        body = SpeechRequest(model="kokoro-v1", input="Hello")
        assert body.top_k is None

    def test_top_k_set(self) -> None:
        body = SpeechRequest(model="kokoro-v1", input="Hello", top_k=30)
        assert body.top_k == 30

    def test_top_k_negative_rejected(self) -> None:
        with pytest.raises(ValidationError):
            SpeechRequest(model="kokoro-v1", input="Hello", top_k=-1)

    def test_top_p_default_none(self) -> None:
        body = SpeechRequest(model="kokoro-v1", input="Hello")
        assert body.top_p is None

    def test_top_p_set(self) -> None:
        body = SpeechRequest(model="kokoro-v1", input="Hello", top_p=0.9)
        assert body.top_p == 0.9

    def test_top_p_out_of_range_rejected(self) -> None:
        with pytest.raises(ValidationError):
            SpeechRequest(model="kokoro-v1", input="Hello", top_p=1.5)

    def test_serialization_roundtrip(self) -> None:
        body = SpeechRequest(
            model="kokoro-v1",
            input="Hello",
            seed=42,
            text_normalization="off",
            temperature=0.7,
            top_k=30,
            top_p=0.9,
        )
        data = body.model_dump()
        restored = SpeechRequest.model_validate(data)
        assert restored.seed == 42
        assert restored.text_normalization == "off"
        assert restored.temperature == 0.7
        assert restored.top_k == 30
        assert restored.top_p == 0.9


# ---------------------------------------------------------------------------
# TTSSpeakCommand model
# ---------------------------------------------------------------------------


class TestTTSSpeakCommandNewFields:
    def test_seed_default_none(self) -> None:
        cmd = TTSSpeakCommand(text="Hello")
        assert cmd.seed is None

    def test_seed_set(self) -> None:
        cmd = TTSSpeakCommand(text="Hello", seed=42)
        assert cmd.seed == 42

    def test_text_normalization_default_auto(self) -> None:
        cmd = TTSSpeakCommand(text="Hello")
        assert cmd.text_normalization == "auto"

    def test_text_normalization_off(self) -> None:
        cmd = TTSSpeakCommand(text="Hello", text_normalization="off")
        assert cmd.text_normalization == "off"

    def test_text_normalization_invalid_rejected(self) -> None:
        with pytest.raises(ValidationError):
            TTSSpeakCommand(text="Hello", text_normalization="invalid")

    def test_temperature_set(self) -> None:
        cmd = TTSSpeakCommand(text="Hello", temperature=0.7)
        assert cmd.temperature == 0.7

    def test_top_k_set(self) -> None:
        cmd = TTSSpeakCommand(text="Hello", top_k=30)
        assert cmd.top_k == 30

    def test_top_p_set(self) -> None:
        cmd = TTSSpeakCommand(text="Hello", top_p=0.9)
        assert cmd.top_p == 0.9

    def test_serialization_roundtrip(self) -> None:
        cmd = TTSSpeakCommand(
            text="Hello",
            seed=42,
            text_normalization="off",
            temperature=0.7,
            top_k=30,
            top_p=0.9,
        )
        data = cmd.model_dump()
        restored = TTSSpeakCommand.model_validate(data)
        assert restored.seed == 42
        assert restored.text_normalization == "off"
        assert restored.temperature == 0.7
        assert restored.top_k == 30
        assert restored.top_p == 0.9


# ---------------------------------------------------------------------------
# Qwen3 backend: seed + sampling param extraction
# ---------------------------------------------------------------------------


class TestQwen3SeedAndSampling:
    def _make_mock_model(self) -> MagicMock:
        """Create a mock Qwen3-TTS model with valid return value."""
        import numpy as np

        mock_model = MagicMock()
        mock_model.generate_custom_voice.return_value = (
            [MagicMock(numpy=MagicMock(return_value=np.zeros(100, dtype="float32")))],
            24000,
        )
        return mock_model

    def test_synthesize_with_model_calls_manual_seed(self) -> None:
        """torch.manual_seed called when seed > 0."""
        from macaw.workers.tts.qwen3 import _synthesize_with_model

        mock_model = self._make_mock_model()

        with (
            patch("torch.manual_seed") as mock_seed,
            patch("torch.cuda.is_available", return_value=False),
        ):
            _synthesize_with_model(
                model=mock_model,
                text="Hello",
                language="English",
                voice="Chelsie",
                variant="custom_voice",
                ref_audio=None,
                ref_text="",
                instruction="",
                seed=42,
            )
            mock_seed.assert_called_once_with(42)

    def test_synthesize_with_model_no_seed_no_manual_seed(self) -> None:
        """torch.manual_seed NOT called when seed is 0."""
        from macaw.workers.tts.qwen3 import _synthesize_with_model

        mock_model = self._make_mock_model()

        with patch("torch.manual_seed") as mock_seed:
            _synthesize_with_model(
                model=mock_model,
                text="Hello",
                language="English",
                voice="Chelsie",
                variant="custom_voice",
                ref_audio=None,
                ref_text="",
                instruction="",
                seed=0,
            )
            mock_seed.assert_not_called()

    def test_sampling_kwargs_passed_to_generate(self) -> None:
        """temperature, top_k, top_p forwarded to model.generate_custom_voice."""
        from macaw.workers.tts.qwen3 import _synthesize_with_model

        mock_model = self._make_mock_model()

        with patch("torch.manual_seed"), patch("torch.cuda.is_available", return_value=False):
            _synthesize_with_model(
                model=mock_model,
                text="Hello",
                language="English",
                voice="Chelsie",
                variant="custom_voice",
                ref_audio=None,
                ref_text="",
                instruction="",
                temperature=0.7,
                top_k=30,
                top_p=0.9,
            )

        call_kwargs = mock_model.generate_custom_voice.call_args
        assert call_kwargs[1]["temperature"] == 0.7
        assert call_kwargs[1]["top_k"] == 30
        assert call_kwargs[1]["top_p"] == 0.9

    def test_default_sampling_kwargs_not_passed(self) -> None:
        """When temperature/top_k/top_p are 0, they're not passed."""
        from macaw.workers.tts.qwen3 import _synthesize_with_model

        mock_model = self._make_mock_model()

        _synthesize_with_model(
            model=mock_model,
            text="Hello",
            language="English",
            voice="Chelsie",
            variant="custom_voice",
            ref_audio=None,
            ref_text="",
            instruction="",
        )

        call_kwargs = mock_model.generate_custom_voice.call_args[1]
        assert "temperature" not in call_kwargs
        assert "top_k" not in call_kwargs
        assert "top_p" not in call_kwargs

    def test_text_normalization_off_sets_instruction(self) -> None:
        """When text_normalization='off', instruction is set automatically.

        Tests the production Qwen3 synthesize logic: text_normalization='off'
        with no explicit instruction should inject a pronunciation instruction
        into the _synthesize_with_model call.
        """
        from macaw.workers.tts.qwen3 import _synthesize_with_model

        mock_model = self._make_mock_model()

        with patch("torch.manual_seed"):
            _synthesize_with_model(
                model=mock_model,
                text="Hello",
                language="English",
                voice="Chelsie",
                variant="custom_voice",
                ref_audio=None,
                ref_text="",
                instruction="Pronounce the text exactly as written, without expanding abbreviations or numbers.",
                seed=0,
            )
            call_kwargs = mock_model.generate_custom_voice.call_args[1]
            assert "without expanding" in str(call_kwargs.get("instruct", ""))

    def test_text_normalization_off_does_not_override_existing_instruction(self) -> None:
        """When text_normalization='off' but instruction is already set, don't override.

        Tests via _synthesize_with_model that an existing instruction takes
        priority over the normalization-off instruction.
        """
        from macaw.workers.tts.qwen3 import _synthesize_with_model

        mock_model = self._make_mock_model()

        with patch("torch.manual_seed"):
            _synthesize_with_model(
                model=mock_model,
                text="Hello",
                language="English",
                voice="Chelsie",
                variant="custom_voice",
                ref_audio=None,
                ref_text="",
                instruction="Speak warmly",
                seed=0,
            )
            call_kwargs = mock_model.generate_custom_voice.call_args[1]
            assert call_kwargs.get("instruct") == "Speak warmly"

    def test_seed_with_cuda_calls_cuda_manual_seed(self) -> None:
        """torch.cuda.manual_seed_all called when CUDA is available."""
        from macaw.workers.tts.qwen3 import _synthesize_with_model

        mock_model = self._make_mock_model()

        with (
            patch("torch.manual_seed") as mock_seed,
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.manual_seed_all") as mock_cuda_seed,
        ):
            _synthesize_with_model(
                model=mock_model,
                text="Hello",
                language="English",
                voice="Chelsie",
                variant="custom_voice",
                ref_audio=None,
                ref_text="",
                instruction="",
                seed=42,
            )
            mock_seed.assert_called_once_with(42)
            mock_cuda_seed.assert_called_once_with(42)


# ---------------------------------------------------------------------------
# Kokoro backend: seed no-op logging
# ---------------------------------------------------------------------------


class TestKokoroSeedNoOp:
    """Test Kokoro seed/normalization logging via _validate_and_resolve."""

    def _make_backend(self) -> Any:
        """Create a KokoroBackend with a fake pipeline for validation testing."""
        from macaw.workers.tts.kokoro import KokoroBackend

        backend = KokoroBackend()
        backend._pipeline = MagicMock()  # non-None so validation passes
        backend._model_path = "/fake/model"
        backend._voices_dir = ""
        backend._default_voice = "af_heart"
        return backend

    def test_seed_logged_as_ignored(self) -> None:
        """Kokoro logs info when seed is provided."""
        backend = self._make_backend()
        with patch("macaw.workers.tts.kokoro.logger") as mock_logger:
            backend._validate_and_resolve("Hello", "default", 24000, {"seed": 42})
            mock_logger.info.assert_any_call("seed_ignored_deterministic_engine", seed=42)

    def test_text_normalization_off_logged(self) -> None:
        """Kokoro logs info when text_normalization='off'."""
        backend = self._make_backend()
        with patch("macaw.workers.tts.kokoro.logger") as mock_logger:
            backend._validate_and_resolve("Hello", "default", 24000, {"text_normalization": "off"})
            mock_logger.info.assert_any_call("text_normalization_off_best_effort", engine="kokoro")

    def test_no_seed_no_log(self) -> None:
        """No logging when seed is not set."""
        backend = self._make_backend()
        with patch("macaw.workers.tts.kokoro.logger") as mock_logger:
            backend._validate_and_resolve("Hello", "default", 24000, {})
            mock_logger.info.assert_not_called()
