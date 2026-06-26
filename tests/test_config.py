import os
import tempfile
import yaml
import pytest

from src.config import PipelineConfig, InferenceConfig, ModelConfig, FusionConfig


class TestPipelineConfig:
    def test_default_values(self):
        config = PipelineConfig()
        assert config.inference.num_samples == 10
        assert config.inference.activation == "sigmoid"
        assert config.model.in_channels == 3
        assert config.mc_dropout.p == 0.01
        assert config.noisy.noise_std == 0.01

    def test_from_yaml(self):
        data = {
            "inference": {"num_samples": 20, "activation": "softmax"},
            "model": {"in_channels": 1, "out_channels": 3},
            "noisy": {"noise_std": 0.05},
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(data, f)
            fname = f.name
        config = PipelineConfig.from_yaml(fname)
        assert config.inference.num_samples == 20
        assert config.inference.activation == "softmax"
        assert config.model.in_channels == 1
        assert config.model.out_channels == 3
        assert config.noisy.noise_std == 0.05
        os.unlink(fname)
