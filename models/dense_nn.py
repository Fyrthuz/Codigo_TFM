import torch
import torch.nn as nn

class DenseNN(nn.Module):
    def __init__(self, input_size=28*28, hidden_sizes=[128, 64], output_size=10):
        super(DenseNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fc3 = nn.Linear(hidden_sizes[1], output_size)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)  # Apply Softmax for one-hot output

    def forward(self, x):
        x = x.view(-1, 28*28)  # Flatten the input
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return self.softmax(x)  # Return probabilities as one-hot encoded output

def load_model(model_path, device):
    model = DenseNN().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model
