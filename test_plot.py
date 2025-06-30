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
k = 2  # Input dimension (can be any positive integer now!)
exponent = 2
decay_rate = -4
N_initial = 1e4  # Initial number of samples before filtering
volume_ratio = (math.pi / 4)**(k / 2) / math.gamma(k / 2 + 1)

# Generate k-dimensional input data
if k == 1:
    x = np.linspace(-1, 1, N_initial).reshape(-1, 1)
    x = np.sign(x) * (abs(x)**(1/exponent))
else:
    # Generate random samples in hypercube
    base_linspace = np.linspace(-1, 1, int(round(N_initial**(1/k))))
    base_linspace = np.sign(base_linspace) * (abs(base_linspace)**(1/exponent))

    linspaces = [copy.deepcopy(base_linspace) for _ in range(k)]
    grids = np.meshgrid(*linspaces, indexing='ij')
    x = np.stack([grid.ravel() for grid in grids], axis=-1)

# Filter to keep only points inside unit sphere
norms = np.linalg.norm(x, ord=1, axis=1)
inside_sphere = norms <= 2
x = x[inside_sphere]
N = len(x)

print(f"Generated {N} samples inside unit sphere (from {N_initial} initial samples)")
print(f"Retention rate: {N/N_initial:.2%}")

# Target function: generalized to k dimensions
# Compute L2 norm for multi-dimensional case
x_norm = np.linalg.norm(x, axis=1, keepdims=True)
y = np.cos(np.pi * np.sign(x)*np.abs(x)**exponent * 7)

# Convert to PyTorch tensors
X = torch.from_numpy(x).float()
Y = torch.from_numpy(y).float()

Y = Y.sum(dim=1)

# Convert back to numpy for plotting
Y_numpy = Y.numpy()
x1 = x[:, 0]
x2 = x[:, 1]

# Create figure with multiple subplots
fig = plt.figure(figsize=(15, 5))

# 1. 3D surface plot
ax1 = fig.add_subplot(131, projection='3d')
scatter = ax1.scatter(x1, x2, Y_numpy, c=Y_numpy, cmap='viridis', s=1, alpha=0.6)
ax1.set_xlabel('x1')
ax1.set_ylabel('x2')
ax1.set_zlabel('Y')
ax1.set_title('3D Surface Plot: Y vs (x1, x2)')
plt.colorbar(scatter, ax=ax1, shrink=0.5)

# 2. 2D heatmap/contour plot
ax2 = fig.add_subplot(132)
# Create a grid for interpolation
grid_size = int(np.sqrt(N))
xi = np.linspace(x1.min(), x1.max(), grid_size)
yi = np.linspace(x2.min(), x2.max(), grid_size)
Xi, Yi = np.meshgrid(xi, yi)

# Interpolate Y values on grid
from scipy.interpolate import griddata
Zi = griddata((x1, x2), Y_numpy, (Xi, Yi), method='linear')

# Create contour plot
contour = ax2.contourf(Xi, Yi, Zi, levels=20, cmap='viridis')
ax2.set_xlabel('x1')
ax2.set_ylabel('x2')
ax2.set_title('2D Contour Plot: Y vs (x1, x2)')
plt.colorbar(contour, ax=ax2)

# Add circle to show unit sphere boundary
circle = plt.Circle((0, 0), 1, fill=False, color='red', linewidth=2)
ax2.add_patch(circle)
ax2.set_aspect('equal')

# 3. Scatter plot colored by Y value
ax3 = fig.add_subplot(133)
scatter2 = ax3.scatter(x1, x2, c=Y_numpy, cmap='viridis', s=1, alpha=0.8)
ax3.set_xlabel('x1')
ax3.set_ylabel('x2')
ax3.set_title('2D Scatter Plot: Points colored by Y value')
plt.colorbar(scatter2, ax=ax3)

# Add circle to show unit sphere boundary
circle2 = plt.Circle((0, 0), 1, fill=False, color='red', linewidth=2)
ax3.add_patch(circle2)
ax3.set_aspect('equal')

plt.tight_layout()
plt.show()

# Additional plot: Y vs radial distance
fig2, ax = plt.subplots(figsize=(8, 6))
radial_dist = np.linalg.norm(x, axis=1)
ax.scatter(radial_dist, Y_numpy, s=1, alpha=0.5)
ax.set_xlabel('Radial distance from origin')
ax.set_ylabel('Y')
ax.set_title('Y vs Radial Distance')
ax.grid(True, alpha=0.3)
plt.show()

# Print some statistics
print(f"\nY statistics:")
print(f"Min: {Y_numpy.min():.4f}")
print(f"Max: {Y_numpy.max():.4f}")
print(f"Mean: {Y_numpy.mean():.4f}")
print(f"Std: {Y_numpy.std():.4f}")