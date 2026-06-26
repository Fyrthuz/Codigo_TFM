import torch
import pytest

from src.uncertainty.noise_inference import NoisyInference, noisy_inference


class SimpleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 1, 3, padding=1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.conv(x))


class TestNoisyInference:
    def test_noisy_samples_shape(self):
        x = torch.randn(1, 1, 32, 32).cpu()
        ni = NoisyInference(x, N_SAMPLES=5, noise_std=0.1)
        samples = ni.generate_noisy_samples()
        assert len(samples) == 5
        for s in samples:
            assert s.shape == (1, 1, 32, 32)
            assert s.device.type == x.device.type

    def test_noise_added(self):
        x = torch.zeros(1, 1, 32, 32).cpu()
        ni = NoisyInference(x, N_SAMPLES=1, noise_std=0.5)
        samples = ni.generate_noisy_samples()
        assert not torch.allclose(samples[0], x)

    def test_noisy_inference_output(self):
        model = SimpleModel().cpu().eval()
        x = torch.randn(1, 1, 16, 16).cpu()
        ni = NoisyInference(x, N_SAMPLES=3, noise_std=0.01)
        images, masks, mean_prob, entropy = noisy_inference(
            ni, model, activation="sigmoid"
        )
        assert len(images) == 3
        assert len(masks) == 3
        assert mean_prob.shape == (16, 16)
        assert entropy.shape == (16, 16)
