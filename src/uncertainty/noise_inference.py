import torch
import torch.nn.functional as F
import numpy as np


class NoisyInference:
    def __init__(self, image, N_SAMPLES=10, noise_std=0.1):
        self.N_SAMPLES = N_SAMPLES
        self.noise_std = noise_std

        if isinstance(image, str):
            from PIL import Image
            from torchvision.transforms.functional import to_tensor
            self.image_tensor = to_tensor(Image.open(image).convert("RGB")).unsqueeze(0)
        elif isinstance(image, torch.Tensor):
            self.image_tensor = image.clone()
        else:
            raise ValueError("Input image must be a file path or torch.Tensor.")

    def add_noise(self, image_tensor):
        noise = torch.randn_like(image_tensor) * self.noise_std
        noisy_image = image_tensor + noise
        noisy_image = torch.clamp(noisy_image, 0, 1)
        return noisy_image

    def generate_noisy_samples(self):
        noisy_samples = []
        for _ in range(self.N_SAMPLES):
            noisy_samples.append(self.add_noise(self.image_tensor))
        return noisy_samples


def noisy_inference(noisy_model, model, activation="sigmoid"):
    noisy_samples = noisy_model.generate_noisy_samples()
    all_probs = []
    noisy_images = []
    masks_list = []

    with torch.no_grad():
        for sample in noisy_samples:
            noisy_images.append(sample.cpu().numpy())
            output = model(sample)
            all_probs.append(output)

    all_probs = torch.stack(all_probs)

    if activation == "softmax":
        mean_probs = all_probs.mean(dim=0).cpu().numpy()
        entropy_map = -np.sum(mean_probs * np.log(mean_probs + 1e-8), axis=0)
        masks_list = np.argmax(all_probs.cpu().numpy(), axis=1)
    elif activation == "sigmoid":
        all_probs_sig = torch.sigmoid(all_probs)
        mean_probs = np.squeeze(all_probs_sig.mean(dim=0).cpu().numpy())
        entropy_map = -(
            mean_probs * np.log(mean_probs + 1e-8)
            + (1 - mean_probs) * np.log(1 - mean_probs + 1e-8)
        )
        masks_list = (all_probs_sig.cpu().numpy() > 0.5).astype(np.uint8)
    else:
        raise ValueError("activation must be 'softmax' or 'sigmoid'")

    return noisy_images, masks_list, mean_probs, entropy_map
