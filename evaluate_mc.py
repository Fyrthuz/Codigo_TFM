import torch
import torch.nn as nn
from models.dense_nn import load_model
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from mc_dropout import MCDropout

def apply_mc_mask(p: float):
    """Hook que aplica una máscara binaria con probabilidad p de anular valores."""
    def hook(module, input, output):
        if isinstance(output, torch.Tensor):
            # Generar máscara aleatoria con distribución Bernoulli(1-p)
            mask = (torch.rand_like(output) > p).float()
            # Anular valores (sin escalar para mantener la varianza)
            return output * mask
        return output
    return hook

# You define the layer to not apply MC Dropout
def enable_explicit_mc_dropout(
    model: nn.Module,
    p: float,
    ignore_specific_model_layers: list = [],  # Lista de módulos específicos a excluir
    ignore_type_layers: list = [],  # Lista de tipos de capas a excluir (ej. [nn.Linear])
):
    """Activa MC Dropout, excluyendo capas específicas o por tipo."""
    for name, module in model.named_modules():
        # Excluir el módulo padre (DenseNN) y otros contenedores
        if name == "":  # El módulo padre (DenseNN) no tiene nombre en named_modules
            continue
        # Aplica hooks solo a capas con parámetros (Linear, Conv2d, etc.)
        if (
            (module not in ignore_specific_model_layers)
            and (not isinstance(module, tuple(ignore_type_layers)))
            # and isinstance(module, (nn.Linear, nn.Conv2d))  # Aplica solo a estos tipos
        ):
            module.register_forward_hook(apply_mc_mask(p))

def enable_mc_dropout(model: nn.Module, p: float):
    """Activa MC Dropout en todas las capas del modelo."""
    for name, module in model.named_modules():
        # Excluir el módulo padre (DenseNN) y otros contenedores
        if name == "":  # El módulo padre (DenseNN) no tiene nombre en named_modules
            continue
        # Aplicar solo a capas con parámetros, excepto BatchNorm, Dropout y activ
        module.register_forward_hook(apply_mc_mask(p))

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Inicializar y configurar MC Dropout
model = load_model("mnist_model.pth", device)

enable_mc_dropout(model, p=0.2)  # Probabilidad de anulación: 20%

# Define transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load test dataset
test_dataset = torchvision.datasets.MNIST(root="./data", train=False, transform=transform, download=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

# Inferencia con múltiples pasadas estocásticas
model.eval()

print("Módulos con hooks:")
for module in model.modules():
    has_hook = "✅" if len(module._forward_hooks) > 0 else "❌"
    print(f"- {module}: {has_hook}")

with torch.no_grad():
    x, _ = next(iter(test_loader))
    x = x.to(device)
    outputs = [model(x) for _ in range(100)]  # 100 muestras MC

# Calcular media y varianza para incertidumbre
mean = torch.stack(outputs).mean(dim=0)
variance = torch.stack(outputs).var(dim=0)

print("Mean shape:", mean.shape)
print("Variance shape:", variance.shape)

# Check the mean and variance values for the first sample
print("Sample mean:", mean[0].cpu().numpy())
print("Sample variance:", variance[0].cpu().numpy())
print("Sample prediction:", mean[0].argmax().item())

# Try with explicit MC Dropout
print("\n===============================Explicit MC Dropout===============================")
model = load_model("mnist_model.pth", device)
# enable_explicit_mc_dropout(model, p=0.2, ignore_type_layers=[nn.ReLU, nn.Softmax], ignore_specific_model_layers=[model.fc1])
enable_explicit_mc_dropout(model, p=0.2, ignore_type_layers=[nn.ReLU, nn.Softmax], ignore_specific_model_layers=[])


print("Módulos con hooks:")
for module in model.modules():
    has_hook = "✅" if len(module._forward_hooks) > 0 else "❌"
    print(f"- {module}: {has_hook}")


model.eval()
with torch.no_grad():
    x, _ = next(iter(test_loader))
    x = x.to(device)
    outputs = [model(x) for _ in range(100)]  # 100 mue
# Calcular media y varianza para incertidumbre
mean = torch.stack(outputs).mean(dim=0)
variance = torch.stack(outputs).var(dim=0)

print("Mean shape:", mean.shape)
print("Variance shape:", variance.shape)

# Check the mean and variance values for the first sample
print("Sample mean:", mean[0].cpu().numpy())
print("Sample variance:", variance[0].cpu().numpy())
print("Sample prediction:", mean[0].argmax().item())


# Try with explicit MC Dropout in the class
print("\n===============================Explicit MC Dropout (with class)===============================")
model = load_model("mnist_model.pth", device)
# enable_explicit_mc_dropout(model, p=0.2, ignore_type_layers=[nn.ReLU, nn.Softmax], ignore_specific_model_layers=[model.fc1])
MCDropout(model, p=0.2).enable(ignore_type_layers=[nn.ReLU, nn.Softmax])
# MCDropout(model, p=0.2).enable(ignore_type_layers=[nn.ReLU, nn.Softmax], layer_types=[nn.Linear, nn.Conv2d])

print("Módulos con hooks:")
for module in model.modules():
    has_hook = "✅" if len(module._forward_hooks) > 0 else "❌"
    print(f"- {module}: {has_hook}")


model.eval()
with torch.no_grad():
    x, _ = next(iter(test_loader))
    x = x.to(device)
    outputs = [model(x) for _ in range(100)]  # 100 mue
# Calcular media y varianza para incertidumbre
mean = torch.stack(outputs).mean(dim=0)
variance = torch.stack(outputs).var(dim=0)

print("Mean shape:", mean.shape)
print("Variance shape:", variance.shape)

# Check the mean and variance values for the first sample
print("Sample mean:", mean[0].cpu().numpy())
print("Sample variance:", variance[0].cpu().numpy())
print("Sample prediction:", mean[0].argmax().item())