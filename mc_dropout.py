import torch
import torch.nn as nn
import time

class MCDropout:
    def __init__(self, model: nn.Module, p: float = 0.2):
        """
        Initialize Monte Carlo Dropout wrapper.

        Args:
            model: PyTorch model to apply MC Dropout
            p: Probability of masking a neuron (0.0-1.0)
        """
        self.model = model
        self.p = p
        self.hooks = []
        self.enabled = False  # Flag to track if dropout is enabled

    def _apply_mask(self, module, input, output):
        """Internal method to apply dropout mask"""
        if self.enabled and isinstance(output, torch.Tensor):  # Only apply if enabled
            mask = (torch.rand_like(output) > self.p).float()
            return output * mask
        return output

    def enable(self, 
               ignore_specific_layers: list = None,
               ignore_type_layers: list = None,
               layer_types: list = None):
        """
        Enable MC Dropout with fine-grained control

        Args:
            ignore_specific_layers: List of specific layer instances to exclude
            ignore_type_layers: List of layer types to exclude (e.g., [nn.ReLU])
            layer_types: Tuple of layer types to apply to (default: Linear/Conv2d)
        """
        if self.enabled:  # Don't re-apply if already enabled
            return

        # Set default arguments (no change here)
        ignore_specific_layers = ignore_specific_layers or []
        ignore_type_layers = ignore_type_layers or []
        layer_types = layer_types or []

        # Register hooks (no significant change, but simplified logic)
        for name, module in self.model.named_modules():
            if name == "":  # Skip parent container
                continue

            apply_condition = (
                (isinstance(module, tuple(layer_types)) or not layer_types) and
                module not in ignore_specific_layers and
                not isinstance(module, tuple(ignore_type_layers))
            )

            if apply_condition:
                self.hooks.append(module.register_forward_hook(self._apply_mask))

        self.enabled = True  # Set the flag


    def remove(self):
        """Remove all active dropout masks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.enabled = False # Clear the flag

    def __enter__(self):
        """Context manager support"""
        self.enable()  # Enable when entering context
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Cleanup hooks on context exit"""
        self.remove() # Ensure removal even if exception


    def forward(self, *args, **kwargs): # Add a forward method
        """
        Forward pass through the model with MC Dropout enabled.
        """
        if not self.enabled:
            raise RuntimeError("MC Dropout is not enabled. Call enable() or use the context manager.")
        return self.model(*args, **kwargs)

if __name__ == "__main__":
    import torch
    import torchvision
    import matplotlib.pyplot as plt
    import PIL
    import numpy as np
    from models.unet.unet import UNet

    NUM_SAMPLES = 10
    PROBABILITY = 0.2
    NUM_CLASSES = 2
    INPUT_CHANNELS = 3


    # Corrección: Mover modelo y datos al dispositivo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UNet(in_channels=INPUT_CHANNELS, num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load("models/unet/last.pt", map_location=device))

    # Load a test image
    image = PIL.Image.open("data/car.png").convert("RGB")
    image = torchvision.transforms.ToTensor()(image).unsqueeze(0).to(device)

    # Normal Inference
    output = model(image)
    output = np.argmax(output.squeeze().detach().numpy(), axis=0)


    # MC Dropout
    mc = MCDropout(model, p=PROBABILITY)
    mc.enable(ignore_type_layers=[torch.nn.ReLU, torch.nn.Softmax])
    
    
    tiempo_inicial = time.time()
    # Repetir la imagen para el batch
    batch = image.repeat(NUM_SAMPLES, 1, 1, 1)

    with torch.no_grad():
        outputs = model(batch) # (NUM_SAMPLES, C, H, W)
    
    # Calcular estadísticas
    mean = outputs.mean(dim=0).squeeze(0).cpu().numpy()  # (C, H, W)
    variance = outputs.var(dim=0).squeeze(0).cpu().numpy()

    # Calcular entropía
    probabilities = torch.nn.functional.softmax(outputs, dim=1)  # Aplicar softmax porque no se aplica en el modelo
    mean_probs = probabilities.mean(dim=0).squeeze(0).cpu().numpy()  # (C, H, W)

    # Calcular entropía (asegurar que las probabilidades sumen 1)
    epsilon = 1e-10
    entropy = -np.sum(mean_probs * np.log(mean_probs + epsilon), axis=0)  # (H, W)

    tiempo_final = time.time()
    print(f"Tiempo de inferencia: {tiempo_final - tiempo_inicial} segundos")

    # Visualización
    plt.figure(figsize=(18, 6))

    # Figuer title
    plt.suptitle(f"MC Dropout using {NUM_SAMPLES} samples with p={PROBABILITY}", fontsize=16)

    # Imagen de entrada
    plt.subplot(1, 4, 1)
    input_image = image.squeeze().cpu().permute(1, 2, 0).numpy()
    plt.imshow(np.clip(input_image, 0, 1))
    plt.title("Input Image")
    plt.axis('off')

    # Predicción normal
    plt.subplot(1, 4, 2)
    plt.imshow(output, cmap="jet")
    plt.title("Normal Prediction")
    plt.axis('off')

    # Predicción media
    plt.subplot(1, 4, 3)
    plt.imshow(np.argmax(mean_probs, axis=0), cmap="jet")
    plt.title("Mean Prediction")
    plt.axis('off')

    # Incertidumbre (entropía con escala ajustada)
    plt.subplot(1, 4, 4)
    entropy_plot = plt.imshow(entropy, cmap="viridis", vmin=0, vmax=np.log(NUM_CLASSES))  # Máx entropía para NUM_CLASSES clases: ln(NUM_CLASSES)
    plt.colorbar(entropy_plot, fraction=0.046, pad=0.04)
    plt.title("Uncertainty (Entropy)")
    plt.axis('off')

    plt.show()

    tiempo_inicial = time.time()
    # Generar muestras MC
    with torch.no_grad():
        outputs = [model(image) for _ in range(NUM_SAMPLES)]
    outputs = torch.stack(outputs)  # (NUM_SAMPLES, 1, C, H, W)

    print(outputs.shape)
    outputs = outputs.squeeze(1)  # (NUM_SAMPLES, C, H, W)
    # Calcular estadísticas
    mean = outputs.mean(dim=0).squeeze(0).cpu().numpy()  # (C, H, W)
    variance = outputs.var(dim=0).squeeze(0).cpu().numpy()

    # Calcular entropía
    probabilities = torch.nn.functional.softmax(outputs, dim=1)  # Aplicar softmax porque no se aplica en el modelo
    mean_probs = probabilities.mean(dim=0).squeeze(0).cpu().numpy()  # (C, H, W)

    # Calcular entropía (asegurar que las probabilidades sumen 1)
    epsilon = 1e-10
    entropy = -np.sum(mean_probs * np.log(mean_probs + epsilon), axis=0)  # (H, W)

    tiempo_final = time.time()
    print(f"Tiempo de inferencia: {tiempo_final - tiempo_inicial} segundos")
    # Visualización
    plt.figure(figsize=(18, 6))

    # Figuer title
    plt.suptitle(f"MC Dropout using {NUM_SAMPLES} samples with p={PROBABILITY}", fontsize=16)

    # Imagen de entrada
    plt.subplot(1, 4, 1)
    input_image = image.squeeze().cpu().permute(1, 2, 0).numpy()
    plt.imshow(np.clip(input_image, 0, 1))
    plt.title("Input Image")
    plt.axis('off')

    # Predicción normal
    plt.subplot(1, 4, 2)
    plt.imshow(output, cmap="jet")
    plt.title("Normal Prediction")
    plt.axis('off')

    # Predicción media
    plt.subplot(1, 4, 3)
    plt.imshow(np.argmax(mean_probs, axis=0), cmap="jet")
    plt.title("Mean Prediction")
    plt.axis('off')

    # Incertidumbre (entropía con escala ajustada)
    plt.subplot(1, 4, 4)
    entropy_plot = plt.imshow(entropy, cmap="viridis", vmin=0, vmax=np.log(NUM_CLASSES))  # Máx entropía para NUM_CLASSES clases: ln(NUM_CLASSES)
    plt.colorbar(entropy_plot, fraction=0.046, pad=0.04)
    plt.title("Uncertainty (Entropy)")
    plt.axis('off')

    plt.show()


    # Visualizacion de todas las muestras
    plt.figure(figsize=(18, 6))
    plt.suptitle(f"MC Dropout using {NUM_SAMPLES} samples with p={PROBABILITY}", fontsize=16)
    for i in range(NUM_SAMPLES):
        plt.subplot(1, NUM_SAMPLES, i + 1)
        plt.imshow(np.argmax(probabilities[i].squeeze().cpu().numpy(), axis=0), cmap="jet")
        plt.title(f"Sample {i + 1}")
        plt.axis('off')

    plt.show()