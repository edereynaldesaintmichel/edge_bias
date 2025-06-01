#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import copy

# Set seeds for reproducibility
# torch.manual_seed(42)
np.random.seed(42)
batch_size = 128
exponent = 1
decay_rate = -4
N = 500
# N = 50
# x = np.random.uniform(-1, 1, (N, 1))
x = np.linspace(-1, 1, N).reshape(-1, 1)
x = np.sign(x)*(abs(x)**(1/exponent))

x.sort(axis=0)

# Target function: sine-based function of x
# y = np.random.normal(0, 1, (N, 1))
y = np.cos(np.pi * abs(x)**exponent * 15)
data = x # Data is just x now

X = torch.from_numpy(data).float()
Y = torch.from_numpy(y).float()
activation = nn.GELU

class Net(nn.Module):
    def __init__(self, input_size, hidden_layer_width, num_hidden_layers, output_size):
        super(Net, self).__init__()
        layers = []
        # Input layer
        layers.append(nn.Linear(input_size, hidden_layer_width))
        layers.append(activation())
        # Hidden layers
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(hidden_layer_width, hidden_layer_width))
            layers.append(activation())
        # Output layer
        layers.append(nn.Linear(hidden_layer_width, output_size))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


n_training_run = 4
max_layers = 6
best_models = [None for _ in range(max_layers)]
best_losses = [float('inf') for _ in range(max_layers)]
all_losses = [[float('inf') for _ in range(n_training_run)] for k in range(max_layers)]
hidden_layer_width = 16  # width of each hidden layer
criterion = nn.MSELoss()

for i in range(max_layers):
    print(f"Training model with {i+1} hidden layer(s)")

    for k in range(n_training_run):
        print(f"  Training trial {k+1}")
        num_hidden_layers = i

        model = Net(input_size=1, hidden_layer_width=hidden_layer_width, 
                    num_hidden_layers=num_hidden_layers, output_size=1)

        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)

        num_epochs = 10000

        for epoch in range(num_epochs):
            random_indices = torch.randperm(N)[:batch_size]
            optimizer.zero_grad()
            outputs = model(X[random_indices])
            loss = criterion(outputs, Y[random_indices])
            loss.backward()
            optimizer.step()

            if epoch % 100 == 0:
                outputs = model(X)
                loss = criterion(outputs, Y)
                if loss.item() < best_losses[i]:
                    best_losses[i] = loss.item()
                    best_models[i] = copy.deepcopy(model)
                    torch.save(model.state_dict(), f"best_model_{i}.pt")
                
                if loss.item() < all_losses[i][k]:
                    all_losses[i][k] = loss.item()
    print(f"Finished training for {i+1} hidden layer(s). Best loss: {best_losses[i]:.6f}")


x_grid = x
grid_tensor = torch.from_numpy(x_grid).float()
average_losses = torch.Tensor(all_losses).mean(dim=-1)

fig, axes = plt.subplots(2, 3, figsize=(15, 10), sharex=True, sharey=True)

axes = axes.ravel()

for i, model in enumerate(best_models):
    if model is None:
        model = Net(input_size=1, hidden_layer_width=hidden_layer_width, num_hidden_layers=i, output_size=1)
        model.load_state_dict(torch.load(f"best_model_{i}.pt"))
        # continue

    param_count = sum(p.numel() for p in model.parameters())
    ax = axes[i]
    model.eval()
    with torch.no_grad():
        Z_pred = model(grid_tensor).numpy()

    ax.plot(x_grid, Z_pred, color='blue', label='Model Prediction')
    ax.scatter(x, y, color='red', s=1, alpha=0.5, label='Training Data')

    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax.set_title(f'{i+1} Hidden Layer(s)\n{param_count} params\nLoss: {best_losses[i]:.6f}')
    ax.grid(True, linestyle='--', alpha=0.6)
    # ax.set_ylim(-1.5, 1.5) # Set consistent y-limits

# Add legend to the first subplot (or create a global one)
axes[0].legend(loc='upper right')

# Remove empty subplots if any (not needed for 3x3)
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
# fig.suptitle('Neural Network Predictions for 1D Sine Function', fontsize=16) # Optional overall title
plt.show()