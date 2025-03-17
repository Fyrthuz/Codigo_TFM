import torch
import torchvision
import matplotlib.pyplot as plt
import PIL
import numpy as np
from models.unet.unet import UNet
from mc_dropout import MCDropout

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



# Generar muestras MC
with torch.no_grad():
    outputs = [model(image) for _ in range(NUM_SAMPLES)]
outputs = torch.stack(outputs)  # (NUM_SAMPLES, 1, C, H, W)

# Calcular estadísticas
mean = outputs.mean(dim=0).squeeze(0).cpu().numpy()  # (C, H, W)
variance = outputs.var(dim=0).squeeze(0).cpu().numpy()

# Calcular entropía
probabilities = torch.nn.functional.softmax(outputs, dim=2)  # Aplicar softmax porque no se aplica en el modelo
mean_probs = probabilities.mean(dim=0).squeeze(0).cpu().numpy()  # (C, H, W)

# Calcular entropía (asegurar que las probabilidades sumen 1)
epsilon = 1e-10
entropy = -np.sum(mean_probs * np.log(mean_probs + epsilon), axis=0)  # (H, W)

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