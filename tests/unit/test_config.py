"""Testes das configuracoes de pipeline."""

from macaw.config.postprocessing import (
    ITNConfig,
    PostProcessingConfig,
)
from macaw.config.preprocessing import PreprocessingConfig


class TestPreprocessingConfig:
    def test_defaults(self) -> None:
        config = PreprocessingConfig()
        assert config.resample is True
        assert config.target_sample_rate == 16000
        assert config.dc_remove is True
        assert config.dc_remove_cutoff_hz == 20
        assert config.gain_normalize is True
        assert config.target_dbfs == -3.0

    def test_custom_values(self) -> None:
        config = PreprocessingConfig(
            target_dbfs=-6.0,
        )
        assert config.target_dbfs == -6.0


class TestPostProcessingConfig:
    def test_defaults(self) -> None:
        config = PostProcessingConfig()
        assert config.itn.enabled is True
        assert config.itn.default_language == "pt"

    def test_itn_config(self) -> None:
        config = ITNConfig(enabled=False, default_language="en")
        assert config.enabled is False
        assert config.default_language == "en"

    def test_nested_config(self) -> None:
        config = PostProcessingConfig(
            itn=ITNConfig(enabled=True, default_language="en"),
        )
        assert config.itn.default_language == "en"
