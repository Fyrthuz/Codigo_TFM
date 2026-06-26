import torch
import torch.nn as nn
import pytest

from src.uncertainty.mc_dropout import MCDropout, mc_dropout_inference


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 1, 3, padding=1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.sigmoid(x)
        return x


class TestMCDropout:
    def test_init(self):
        model = SimpleModel()
        mc = MCDropout(model, p=0.1)
        assert mc.p == 0.1
        assert not mc.enabled
        assert len(mc.hooks) == 0

    def test_enable_remove(self):
        model = SimpleModel()
        mc = MCDropout(model, p=0.1)
        mc.enable(ignore_type_layers=[nn.ReLU, nn.Sigmoid])
        assert mc.enabled
        assert len(mc.hooks) > 0
        mc.remove()
        assert not mc.enabled
        assert len(mc.hooks) == 0

    def test_mask_applied_when_enabled(self):
        model = SimpleModel()
        mc = MCDropout(model, p=0.5)
        mc.enable(ignore_type_layers=[nn.ReLU, nn.Sigmoid])

        x = torch.randn(1, 1, 8, 8)
        out1 = model(x)
        mc.remove()

        mc.enable(ignore_type_layers=[nn.ReLU, nn.Sigmoid])
        out2 = model(x)
        mc.remove()

        assert not torch.allclose(out1, out2)

    def test_no_mask_when_disabled(self):
        model = SimpleModel()
        mc = MCDropout(model, p=0.5)
        x = torch.ones(1, 1, 8, 8)
        out1 = model(x)
        out2 = model(x)
        assert torch.allclose(out1, out2)

    def test_double_enable_noop(self):
        model = SimpleModel()
        mc = MCDropout(model, p=0.1)
        mc.enable()
        hooks_before = len(mc.hooks)
        mc.enable()
        assert len(mc.hooks) == hooks_before


class TestMCDropoutInference:
    def test_binary_output_shapes(self):
        model = SimpleModel().eval()
        x = torch.randn(1, 1, 32, 32)
        images, masks, mean_prob, entropy = mc_dropout_inference(
            model, x, num_samples=5, activation="sigmoid"
        )
        assert len(images) == 5
        assert len(masks) == 5
        assert mean_prob.shape == (32, 32)
        assert entropy.shape == (32, 32)

    def test_sigmoid_prob_range(self):
        model = SimpleModel().eval()
        x = torch.randn(1, 1, 16, 16)
        _, _, mean_prob, _ = mc_dropout_inference(
            model, x, num_samples=3, activation="sigmoid"
        )
        assert mean_prob.min() >= 0.0
        assert mean_prob.max() <= 1.0
