import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.functional import to_tensor
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

class NoisyInference:
    def __init__(self, image, N_SAMPLES=10, noise_std=0.1):
        """
        Initialize the NoisyInference class.

        Args:
            image (torch.Tensor or PIL.Image or str): Input image (tensor, PIL Image, or file path).
            N_SAMPLES (int): Number of noisy samples to generate.
            noise_std (float): Standard deviation of the Gaussian noise to add.
        """
        self.N_SAMPLES = N_SAMPLES
        self.noise_std = noise_std
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load the image if a file path or PIL Image is provided
        if isinstance(image, str):
            self.image = Image.open(image).convert('RGB')
            self.image_tensor = to_tensor(self.image).unsqueeze(0).to(self.device)  # Add batch dimension and move to device
        elif isinstance(image, Image.Image):
            self.image_tensor = to_tensor(image).unsqueeze(0).to(self.device)  # Add batch dimension and move to device
        elif isinstance(image, torch.Tensor):
            self.image_tensor = image.to(self.device)  # Ensure tensor is on the correct device
        else:
            raise ValueError("Input image must be a file path, PIL Image, or torch.Tensor.")

    def add_noise(self, image_tensor):
        """
        Add Gaussian noise to the image tensor.

        Args:
            image_tensor (torch.Tensor): Input image tensor.

        Returns:
            torch.Tensor: Noisy image tensor.
        """
        noise = torch.randn_like(image_tensor) * self.noise_std
        noisy_image = image_tensor + noise
        noisy_image = torch.clamp(noisy_image, 0, 1)  # Clamp values to [0, 1]
        return noisy_image

    def generate_noisy_samples(self):
        """
        Generate N_SAMPLES noisy versions of the input image.

        Returns:
            list: List of noisy image tensors.
        """
        noisy_samples = []
        for _ in range(self.N_SAMPLES):
            noisy_image = self.add_noise(self.image_tensor)
            noisy_samples.append(noisy_image)
        return noisy_samples



if __name__ == '__main__':
    import torchvision.transforms as T
    from models.unet.unet import UNet

    NUM_CLASSES = 2
    INPUT_CHANNELS = 3

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load an image and convert to a tensor.
    image_pil = Image.open("car.png").convert("RGB")
    image = T.ToTensor()(image_pil).unsqueeze(0).to(device)  # Shape: (1, C, H, W)

    plt.imshow(image_pil)
    plt.title("Input Image")
    plt.axis('off')
    plt.show()

    plt.imshow(image[0].permute(1, 2, 0).cpu().numpy())
    plt.title("Input Image (Tensor)")
    plt.axis('off')
    plt.show()


    # Load a pre-trained model.
    model = UNet(in_channels=INPUT_CHANNELS, num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load("models/unet/last.pt", map_location=device))
    model.to(device)

    # Perform inference on the input image
    output = model(image)
    output = F.softmax(output, dim=1)  # Apply softmax to get class probabilities
    output = np.argmax(output.squeeze().detach().cpu().numpy(), axis=0)

    plt.imshow(output, cmap='gray')
    plt.title("Normal Prediction")
    plt.axis('off')
    plt.show()



    # Perform noisy inference
    noisy_inference = NoisyInference(image, N_SAMPLES=10, noise_std=0.1)

    # Generate noisy samples and plot them
    noisy = noisy_inference.generate_noisy_samples()

    # Visualize the noisy samples
    plt.figure(figsize=(18, 6))
    plt.suptitle(f"Noisy Samples with {noisy_inference.N_SAMPLES} samples", fontsize=16)

    for i, sample in enumerate(noisy):
        plt.subplot(2, 5, i + 1)
        plt.imshow(sample.squeeze().permute(1, 2, 0).cpu().numpy())
        plt.title(f"Noisy Sample {i + 1}")
        plt.axis('off')

    plt.tight_layout()
    plt.show()

    # Predict with noisy inference and collect probabilities
    all_probs = []
    for sample in noisy:
        output = model(sample)
        output_probs = F.softmax(output, dim=1)  # (1, C, H, W)
        all_probs.append(output_probs)
    
    # Stack the probabilities
    all_probs = torch.stack(all_probs)  # Shape: (N_SAMPLES, 1, C, H, W)

    # Average probabilities across samples
    mean_probs = all_probs.mean(dim=0)  # Shape: (1, C, H, W)

    # Compute entropy from the averaged probabilities
    entropy = -torch.sum(mean_probs * torch.log(mean_probs + 1e-10), dim=1)  # (1, H, W)
    entropy = entropy.squeeze().detach().cpu().numpy()  # (H, W)

    # Get mean prediction (argmax of averaged probabilities)
    mean_prediction = torch.argmax(mean_probs, dim=1).squeeze().cpu().numpy()  # (H, W)

    # Visualize the noisy predictions
    plt.figure(figsize=(18, 6))
    plt.suptitle(f"Noisy Predictions with {noisy_inference.N_SAMPLES} samples", fontsize=16)

    for i, prediction in enumerate(all_probs):
        plt.subplot(2, 5, i + 1)
        plt.imshow(prediction.detach().squeeze().argmax(dim=0).cpu().numpy(), cmap='gray')
        plt.title(f"Noisy Prediction {i + 1}")
        plt.axis('off')

    plt.tight_layout()

    plt.show()

    # Compute entropy from the averaged probabilities
    entropy = -torch.sum(mean_probs * torch.log(mean_probs + 1e-10), dim=1)  # (1, H, W)
    entropy = entropy.squeeze().detach().cpu().numpy()  # (H, W)

    # Get mean prediction (argmax of averaged probabilities)
    mean_prediction = torch.argmax(mean_probs, dim=1).squeeze().cpu().numpy()  # (H, W)

    # Visualize the corrected entropy and mean prediction
    plt.figure(figsize=(18, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(mean_prediction, cmap='gray')
    plt.title("Mean Prediction")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(entropy, cmap='viridis', vmin=0, vmax=np.log(NUM_CLASSES))
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.title("Uncertainty (Entropy)")
    plt.axis('off')

    plt.show()

