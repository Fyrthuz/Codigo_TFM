import numpy as np
import pytest
import torch
import torch.nn.functional as F

from src.uncertainty.tta import _RESIZE_SAFE_COMBOS, RandomImageTransformer, tta_inference


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


class _FixedSizeModel(torch.nn.Module):
    """Mimics UniVerSeg: always resizes inputs to a fixed internal size."""

    def __init__(self, size: int = 64):
        super().__init__()
        self.size = size

    def forward(self, x):
        x = F.interpolate(x, size=(self.size, self.size), mode="bilinear", align_corners=False)
        return x.mean(dim=1, keepdim=True)


class TestResizeSafeTTA:
    def test_resize_safe_runs_with_internal_resize(self):
        model = _FixedSizeModel(64)
        image = torch.rand(1, 3, 256, 256)
        imgs, masks, mean, entropy = tta_inference(
            model, image, device="cpu", resize_safe=True
        )
        assert len(_RESIZE_SAFE_COMBOS) == 9
        assert len(imgs) == 9
        assert mean.shape[-2:] == (64, 64)
        assert entropy.shape[-2:] == (64, 64)
        assert np.isfinite(mean).all()

    def test_resize_safe_preserves_size_and_orientation(self):
        # Size preservation (ttach Scale breaks stack) and orientation
        # preservation (query-support matching breaks if the query is
        # flipped while the fixed support set is not).
        x = torch.zeros(1, 3, 32, 32)
        x[:, :, :, 16:] = 1.0  # right half bright
        for name, transform in _RESIZE_SAFE_COMBOS:
            y = torch.clamp(transform(x), 0.0, 1.0)
            assert y.shape == x.shape, name
            assert y[..., :8].mean() < y[..., -8:].mean(), f"{name} altered orientation"

    def test_ttach_path_fails_with_internal_resize(self):
        # Regression test for the original issue: ttach's Scale transforms
        # deaugment masks assuming output size == augmented input size, so
        # stacked predictions have mixed sizes (256/128/64) and torch.stack
        # raises RuntimeError. resize_safe=True is the fix.
        model = _FixedSizeModel(64)
        image = torch.rand(1, 3, 256, 256)
        with pytest.raises(RuntimeError):
            tta_inference(model, image, device="cpu", resize_safe=False)
