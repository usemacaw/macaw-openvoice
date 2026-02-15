"""Testes das configuracoes de pipeline."""

from macaw.config.postprocessing import (
    EntityFormattingConfig,
    HotWordCorrectionConfig,
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
        assert config.itn.language == "pt"
        assert config.entity_formatting.enabled is False
        assert config.hot_word_correction.enabled is False

    def test_itn_config(self) -> None:
        config = ITNConfig(enabled=False, language="en")
        assert config.enabled is False
        assert config.language == "en"

    def test_entity_formatting_config(self) -> None:
        config = EntityFormattingConfig(enabled=True, domain="banking")
        assert config.enabled is True
        assert config.domain == "banking"

    def test_hot_word_correction_config(self) -> None:
        config = HotWordCorrectionConfig(enabled=True, max_edit_distance=3)
        assert config.enabled is True
        assert config.max_edit_distance == 3

    def test_nested_config(self) -> None:
        config = PostProcessingConfig(
            itn=ITNConfig(enabled=True, language="en"),
            entity_formatting=EntityFormattingConfig(enabled=True, domain="medical"),
        )
        assert config.itn.language == "en"
        assert config.entity_formatting.domain == "medical"
