"""Testes do ITNStage (Inverse Text Normalization).

nemo_text_processing NAO esta instalado no ambiente de teste.
Todos os testes usam mocks para simular a biblioteca.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from macaw.postprocessing.itn import ITNStage


class TestITNProcessWithMockNemo:
    def test_itn_process_calls_normalizer_and_returns_result(self) -> None:
        """Mock nemo_text_processing, verifica que process chama o mock e retorna resultado."""
        mock_normalizer = MagicMock()
        mock_normalizer.inverse_normalize.return_value = "2000"

        mock_module = MagicMock()
        mock_module.InverseNormalize.return_value = mock_normalizer

        with patch.dict(
            "sys.modules",
            {
                "nemo_text_processing": MagicMock(),
                "nemo_text_processing.inverse_text_normalization": mock_module,
            },
        ):
            stage = ITNStage(language="pt")
            result = stage.process("dois mil")

        assert result == "2000"
        mock_normalizer.inverse_normalize.assert_called_once_with("dois mil", verbose=False)


class TestITNFallbackWhenNotInstalled:
    def test_returns_text_unchanged_when_import_fails(self) -> None:
        """Quando nemo_text_processing nao esta instalado, process retorna texto original."""
        stage = ITNStage(language="pt")
        # Reset state para forcar re-check
        stage._available = None
        stage._normalizer = None

        # nemo_text_processing nao esta instalado no ambiente de teste
        result = stage.process("dois mil")

        assert result == "dois mil"
        assert stage._available is False


class TestITNFallbackOnInitError:
    def test_returns_text_unchanged_when_normalizer_init_raises(self) -> None:
        """Quando InverseNormalize() levanta excecao, process retorna texto original."""
        mock_module = MagicMock()
        mock_module.InverseNormalize.side_effect = RuntimeError("init failed")

        with patch.dict(
            "sys.modules",
            {
                "nemo_text_processing": MagicMock(),
                "nemo_text_processing.inverse_text_normalization": mock_module,
            },
        ):
            stage = ITNStage(language="pt")
            result = stage.process("dois mil")

        assert result == "dois mil"
        assert stage._available is False
        assert stage._normalizer is None


class TestITNFallbackOnProcessError:
    def test_returns_text_unchanged_when_inverse_normalize_raises(self) -> None:
        """Quando inverse_normalize() levanta excecao, retorna texto original."""
        mock_normalizer = MagicMock()
        mock_normalizer.inverse_normalize.side_effect = RuntimeError("process failed")

        mock_module = MagicMock()
        mock_module.InverseNormalize.return_value = mock_normalizer

        with patch.dict(
            "sys.modules",
            {
                "nemo_text_processing": MagicMock(),
                "nemo_text_processing.inverse_text_normalization": mock_module,
            },
        ):
            stage = ITNStage(language="pt")
            result = stage.process("dois mil")

        assert result == "dois mil"
        assert stage._available is True


class TestITNEmptyAndWhitespace:
    def test_empty_text_returned_without_calling_nemo(self) -> None:
        """String vazia retornada sem chamar NeMo."""
        mock_normalizer = MagicMock()
        mock_module = MagicMock()
        mock_module.InverseNormalize.return_value = mock_normalizer

        with patch.dict(
            "sys.modules",
            {
                "nemo_text_processing": MagicMock(),
                "nemo_text_processing.inverse_text_normalization": mock_module,
            },
        ):
            stage = ITNStage(language="pt")
            result = stage.process("")

        assert result == ""
        mock_module.InverseNormalize.assert_not_called()
        mock_normalizer.inverse_normalize.assert_not_called()

    def test_whitespace_text_returned_without_calling_nemo(self) -> None:
        """String com apenas espacos retornada sem chamar NeMo."""
        mock_normalizer = MagicMock()
        mock_module = MagicMock()
        mock_module.InverseNormalize.return_value = mock_normalizer

        with patch.dict(
            "sys.modules",
            {
                "nemo_text_processing": MagicMock(),
                "nemo_text_processing.inverse_text_normalization": mock_module,
            },
        ):
            stage = ITNStage(language="pt")
            result = stage.process("   ")

        assert result == "   "
        mock_module.InverseNormalize.assert_not_called()
        mock_normalizer.inverse_normalize.assert_not_called()


class TestITNLazyLoading:
    def test_first_call_triggers_import_second_call_uses_cache(self) -> None:
        """Primeira chamada dispara import, segunda usa cache (nao re-importa)."""
        mock_normalizer = MagicMock()
        mock_normalizer.inverse_normalize.return_value = "2000"

        mock_module = MagicMock()
        mock_module.InverseNormalize.return_value = mock_normalizer

        with patch.dict(
            "sys.modules",
            {
                "nemo_text_processing": MagicMock(),
                "nemo_text_processing.inverse_text_normalization": mock_module,
            },
        ):
            stage = ITNStage(language="pt")

            # Primeira chamada: dispara _ensure_loaded e cria normalizer
            stage.process("dois mil")
            assert mock_module.InverseNormalize.call_count == 1

            # Segunda chamada: usa cache, NAO cria normalizer novamente
            stage.process("tres mil")
            assert mock_module.InverseNormalize.call_count == 1

            # Confirma que ambas as chamadas usaram o mesmo normalizer
            assert mock_normalizer.inverse_normalize.call_count == 2


class TestITNLanguageParam:
    def test_language_passed_to_normalizer_constructor(self) -> None:
        """Parametro language e repassado para InverseNormalize(lang=...)."""
        mock_normalizer = MagicMock()
        mock_normalizer.inverse_normalize.return_value = "result"

        mock_module = MagicMock()
        mock_module.InverseNormalize.return_value = mock_normalizer

        with patch.dict(
            "sys.modules",
            {
                "nemo_text_processing": MagicMock(),
                "nemo_text_processing.inverse_text_normalization": mock_module,
            },
        ):
            stage = ITNStage(language="en")
            stage.process("two thousand")

        mock_module.InverseNormalize.assert_called_once_with(lang="en")

    def test_default_language_is_pt(self) -> None:
        """Language padrao e 'pt'."""
        mock_normalizer = MagicMock()
        mock_normalizer.inverse_normalize.return_value = "result"

        mock_module = MagicMock()
        mock_module.InverseNormalize.return_value = mock_normalizer

        with patch.dict(
            "sys.modules",
            {
                "nemo_text_processing": MagicMock(),
                "nemo_text_processing.inverse_text_normalization": mock_module,
            },
        ):
            stage = ITNStage()
            stage.process("dois mil")

        mock_module.InverseNormalize.assert_called_once_with(lang="pt")
