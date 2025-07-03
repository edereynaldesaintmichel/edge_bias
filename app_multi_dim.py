#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import copy
from collections import defaultdict
import math

def ReLU(x):
    return x * (x > 0)

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Hyperparameters
k = 3 # Input dimension (can be any positive integer now!)
exponent = 1.0
decay_rate = -4
N_initial = 1e5  # Initial number of samples before filtering
volume_ratio = (math.pi / 4)**(k / 2) / math.gamma(k / 2 + 1)
# N_initial = int(N_initial / volume_ratio)
# Generate k-dimensional input data
# Generate random samples in hypercube
# x = np.random.uniform(-1, 1, (N_initial, k))
base_linspace = np.linspace(-1, 1, int(round(N_initial**(1/k))))
base_linspace = np.sign(base_linspace) * (abs(base_linspace)**(1/exponent))

linspaces = [copy.deepcopy(base_linspace) for _ in range(k)]
grids = np.meshgrid(*linspaces, indexing='ij')
x = np.stack([grid.ravel() for grid in grids], axis=-1)

# Filter to keep only points inside unit sphere
norms = np.linalg.norm(x, ord=1, axis=1)
inside_sphere = norms <= 1.8
x = x[inside_sphere]
N = len(x)

print(f"Generated {N} samples inside unit sphere (from {N_initial} initial samples)")
print(f"Retention rate: {N/N_initial:.2%}")

# Target function: generalized to k dimensions
# Compute L2 norm for multi-dimensional case
x_norm = np.linalg.norm(x, axis=1, keepdims=True)
# y = np.cos(np.pi * x_norm**exponent * 20)# + 2.71828**(decay_rate * x_norm) * 2
# y = np.cos(np.pi * x[0]**exponent * 10)# + 2.71828**(decay_rate * x_norm) * 2
y = np.cos(np.pi * np.sign(x)*np.abs(x)**exponent * 7)


# Convert to PyTorch tensors
X = torch.from_numpy(x).float()
Y = torch.from_numpy(y).float()

Y = Y.sum(dim=1)
activation = nn.LeakyReLU


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
all_losses = [[float('inf') for _ in range(n_training_run)] for _ in range(max_layers)]
hidden_layer_width = 32  # Increased for higher dimensions
criterion = nn.MSELoss()

# for i in range(max_layers):
#     print(f"Training model with {i+1} hidden layer(s)")

#     for j in range(n_training_run):
#         print(f"  Training trial {j+1}")
#         num_hidden_layers = i

#         model = Net(input_size=k, hidden_layer_width=hidden_layer_width, 
#                     num_hidden_layers=num_hidden_layers, output_size=1)

#         optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)

#         num_epochs = 10000

#         for epoch in range(num_epochs):
#             optimizer.zero_grad()
#             outputs = model(X)
#             loss = criterion(outputs[:,0], Y)
#             loss.backward()
#             optimizer.step()

#             if loss.item() < best_losses[i]:
#                 best_losses[i] = loss.item()
#                 best_models[i] = copy.deepcopy(model)
#                 torch.save(model.state_dict(), f"best_model_{i}_dim{k}.pt")
            
#             if loss.item() < all_losses[i][j]:
#                 all_losses[i][j] = loss.item()
    
#     print(f"Finished training for {i+1} hidden layer(s). Best loss: {best_losses[i]:.6f}")

# Edge Bias Visualization
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.ravel()

# Compute norms for all samples (already filtered to unit sphere)
norms = np.linalg.norm(x, ord=1, axis=1)
# max_norm = 1.8  # We know max norm is 1 since we filtered

for i, model in enumerate(best_models):
    if model is None:
        model = Net(input_size=k, hidden_layer_width=hidden_layer_width, 
                   num_hidden_layers=i, output_size=1)
        model.load_state_dict(torch.load(f"best_model_{i}_dim{k}.pt", map_location=torch.device('cpu')))
        print(sum(p.numel() for p in model.parameters()))
    model.eval()
    ax = axes[i]
    
    # Get predictions and compute squared errors
    with torch.no_grad():
        predictions = model(X)
    squared_errors = ((predictions[:,0] - Y) ** 2).numpy()
    
    # Create dictionaries for cumulative values
    cumulative_samples = defaultdict(int)
    cumulative_loss = defaultdict(float)
    norm_samples = defaultdict(int)
    norm_loss = defaultdict(float)
    
    # Sort by norm for efficient cumulative computation
    sorted_indices = np.argsort(norms)
    
    # Build cumulative dictionaries
    running_count = 0
    running_loss = 0.0

    
    for idx in sorted_indices:
        norm_val = norms[idx]
        norm_rounded = round(norm_val, 2)
        norm_very_rounded = math.ceil(norm_val * 100) / 100

        sample_loss = squared_errors[idx]
        
        running_count += 1
        running_loss += sample_loss
        
        if norm_very_rounded not in norm_samples:
            norm_samples[norm_very_rounded] = 0
            norm_loss[norm_very_rounded] = 0

        norm_samples[norm_very_rounded] += 1
        norm_loss[norm_very_rounded] += sample_loss
        cumulative_samples[norm_rounded] = running_count
        cumulative_loss[norm_rounded] = running_loss
    
    # Extract sorted norm values and corresponding cumulative values
    norm_values = sorted(cumulative_samples.keys())
    norm_values_very_rounded = sorted(norm_samples.keys())
    mean_loss_values = [norm_loss[n]/norm_samples[n] for n in norm_values_very_rounded]
    norm_samples_values = [norm_samples[n] for n in norm_values_very_rounded]

    # Plot loss on primary y-axis
    ax.plot(norm_values_very_rounded, mean_loss_values, 'b-', linewidth=2, 
            label='Mean Loss', alpha=0.8)

    # Create secondary y-axis for number of samples
    ax2 = ax.twinx()
    ax2.plot(norm_values_very_rounded, norm_samples_values, 'r-', linewidth=2, 
            label='Nb of Samples', alpha=0.8)

    # Styling for primary axis
    ax.set_xlabel('Radius (r)')
    ax.set_ylabel('MSE', color='b')
    ax.tick_params(axis='y', labelcolor='b')
    ax.set_title(f'{i+1} Hidden Layer(s)\nMSE: {sum(norm_loss.values())/sum(norm_samples.values()):.4f}')
    ax.grid(True, alpha=0.3)

    # Styling for secondary axis
    ax2.set_ylabel('Nb of Samples', color='r')
    ax2.tick_params(axis='y', labelcolor='r')

    # Combine legends from both axes
    if i == 0:
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='lower right', fontsize=8)

# Remove empty subplots if any
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.suptitle(f'Edge Bias Visualization: {k}D Input → 1D Output\n',
             fontsize=14)
plt.tight_layout()
plt.show()
