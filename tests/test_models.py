import torch
import pytest

from src.models.unet import UNet
from src.models.dense_nn import DenseNN


class TestUNet2D:
    def test_forward_shape(self):
        model = UNet(in_channels=3, out_channels=1, init_features=32)
        x = torch.randn(2, 3, 256, 256)
        out = model(x)
        assert out.shape == (2, 1, 256, 256)

    def test_forward_multiclass(self):
        model = UNet(in_channels=1, out_channels=3, init_features=16)
        x = torch.randn(1, 1, 128, 128)
        out = model(x)
        assert out.shape == (1, 3, 128, 128)

    def test_bilinear_flag(self):
        model = UNet(in_channels=3, out_channels=1, init_features=16, bilinear=True)
        x = torch.randn(1, 3, 64, 64)
        out = model(x)
        assert out.shape == (1, 1, 64, 64)

    def test_small_input(self):
        model = UNet(in_channels=1, out_channels=2, init_features=8)
        x = torch.randn(1, 1, 32, 32)
        out = model(x)
        assert out.shape == (1, 2, 32, 32)
 
class TestDenseNN:
    def test_forward_shape(self):
        model = DenseNN()
        x = torch.randn(4, 1, 28, 28)
        out = model(x)
        assert out.shape == (4, 10)

    def test_output_probability(self):
        model = DenseNN()
        x = torch.randn(2, 1, 28, 28)
        out = model(x)
        assert torch.allclose(out.sum(dim=1), torch.ones(2), atol=1e-5)
