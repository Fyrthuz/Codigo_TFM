import torch
import pytest

from src.uncertainty.tta import RandomImageTransformer


class TestRandomImageTransformer:
    def test_transform_shape(self):
        transformer = RandomImageTransformer()
        x = torch.randn(3, 64, 64)
        out = transformer(x)
        assert out.shape == (3, 64, 64)

    def test_transform_preserves_range(self):
        transformer = RandomImageTransformer()
        x = torch.ones(3, 32, 32)
        out = transformer(x)
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_different_transforms(self):
        transformer = RandomImageTransformer(padding_mode="zeros")
        x = torch.randn(3, 32, 32)
        out1 = transformer(x)
        out2 = transformer(x)
        assert out1.shape == out2.shape
