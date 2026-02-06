import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# Configuration
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Dataset Preparation: MNIST
print("Downloading and preparing MNIST dataset...")
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

# 2. Implementation of Core Logic: Robust Custom Layer Normalization
class CustomLayerNorm(nn.Module):
    """
    Implements Layer Normalization from scratch.
    Paper Reference: Ba, Kiros, Hinton (2016)
    Formula: y = (x - mean) / sqrt(var + eps) * gamma + beta
    """
    def __init__(self, normalized_shape, eps=1e-5):
        super(CustomLayerNorm, self).__init__()
        # Handle int or list/tuple for normalized_shape
        if isinstance(normalized_shape, int):
            self.normalized_shape = (normalized_shape,)
        else:
            self.normalized_shape = tuple(normalized_shape)
        
        self.eps = eps
        
        # Learnable parameters: Gamma (gain) and Beta (bias)
        self.gamma = nn.Parameter(torch.ones(self.normalized_shape))
        self.beta = nn.Parameter(torch.zeros(self.normalized_shape))

    def forward(self, x):
        # Determine which dimensions to reduce over.
        # LayerNorm reduces over the last len(normalized_shape) dimensions.
        num_dims = len(self.normalized_shape)
        dims = tuple(range(x.dim() - num_dims, x.dim()))
        
        # 1. Calculate Mean
        mean = x.mean(dim=dims, keepdim=True)
        
        # 2. Calculate Variance
        # unbiased=False matches the population variance definition used in the paper
        var = x.var(dim=dims, keepdim=True, unbiased=False)
        
        # 3. Normalize
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        
        # 4. Scale and Shift
        output = x_norm * self.gamma + self.beta
        
        return output

# 3. Model Architecture with Switchable Normalization
class DeepFeedForward(nn.Module):
    def __init__(self, norm_type="none"):
        super(DeepFeedForward, self).__init__()
        
        # Architecture: Flatten -> 784 -> 512 -> 512 -> 10
        layers = []
        
        # FIX: Explicitly add Flatten to the sequential block to handle image input [B, 1, 28, 28]
        layers.append(nn.Flatten())
        
        # --- Layer 1 ---
        layers.append(nn.Linear(28*28, 512))
        if norm_type == "batch":
            layers.append(nn.BatchNorm1d(512))
        elif norm_type == "layer":
            layers.append(CustomLayerNorm(512))
        layers.append(nn.ReLU())
        
        # --- Layer 2 ---
        layers.append(nn.Linear(512, 512))
        if norm_type == "batch":
            layers.append(nn.BatchNorm1d(512))
        elif norm_type == "layer":
            layers.append(CustomLayerNorm(512))
        layers.append(nn.ReLU())
        
        # --- Output Layer ---
        layers.append(nn.Linear(512, 10))
        
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# 4. Training Helper
def train_model(norm_type, epochs=EPOCHS):
    print(f"\nTraining Model with Normalization: {norm_type.upper()}")
    model = DeepFeedForward(norm_type=norm_type).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    loss_history = []
    acc_history = []
    
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if i % 100 == 0:
                loss_history.append(loss.item())
        
        epoch_acc = 100 * correct / total
        acc_history.append(epoch_acc)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}, Accuracy: {epoch_acc:.2f}%")
        
    return loss_history, acc_history

if __name__ == "__main__":
    # Run experiments
    # 1. No Normalization
    loss_none, acc_none = train_model("none")
    
    # 2. Batch Normalization
    loss_batch, acc_batch = train_model("batch")
    
    # 3. Layer Normalization (Ours)
    loss_layer, acc_layer = train_model("layer")

    # Visualization
    plt.figure(figsize=(14, 6))

    # Smoothing function for cleaner plots
    def smooth(scalars, weight=0.9):
        last = scalars[0]
        smoothed = []
        for point in scalars:
            smoothed_val = last * weight + (1 - weight) * point
            smoothed.append(smoothed_val)
            last = smoothed_val
        return smoothed

    plt.subplot(1, 2, 1)
    plt.plot(smooth(loss_none), label='No Norm', alpha=0.7)
    plt.plot(smooth(loss_batch), label='Batch Norm', alpha=0.7)
    plt.plot(smooth(loss_layer), label='Layer Norm', alpha=0.9, linewidth=2)
    plt.xlabel('Training Steps (x100)')
    plt.ylabel('Loss')
    plt.title('Training Loss Convergence')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(acc_none, marker='o', label='No Norm')
    plt.plot(acc_batch, marker='s', label='Batch Norm')
    plt.plot(acc_layer, marker='^', label='Layer Norm')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.title('Validation Accuracy vs Epochs')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()