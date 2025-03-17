import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import DataLoader
from models.dense_nn import load_model

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load test dataset
test_dataset = torchvision.datasets.MNIST(root="./data", train=False, transform=transform, download=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

# Load trained model
model = load_model("mnist_model.pth", device)

# Evaluate the model
correct = 0
total = 0
all_predictions = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)  # Softmax applied here

        # Convert to one-hot encoding
        predictions = torch.argmax(outputs, dim=1)  # Get class indices
        one_hot_predictions = torch.zeros(outputs.shape).scatter_(1, predictions.unsqueeze(1), 1)

        all_predictions.append(one_hot_predictions.cpu().numpy())

        total += labels.size(0)
        correct += (predictions == labels).sum().item()

print(f"Test Accuracy: {100 * correct / total:.2f}%")

# Convert all predictions to a numpy array
all_predictions = np.vstack(all_predictions)
print("Sample one-hot encoded predictions:")
print(all_predictions[:5])  # Show first 5 predictions
