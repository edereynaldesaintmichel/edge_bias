import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import copy

N = 500
x = np.linspace(-1., 1., N).reshape(-1, 1)
y = np.cos(np.pi * abs(x)**1.5 * 15)  # Fixed the exponent syntax
activation = nn.GELU

class Net(nn.Module):
    def __init__(self, input_size, hidden_layer_width, num_hidden_layers, output_size):
        super(Net, self).__init__()
        layers = []
        # Input layer
        layers.append(nn.Linear(input_size, hidden_layer_width))
        layers.append(activation())
        # Hidden layers
        for _ in range(num_hidden_layers):  # Fixed variable name
            layers.append(nn.Linear(hidden_layer_width, hidden_layer_width))
            layers.append(activation())
        # Output layer
        layers.append(nn.Linear(hidden_layer_width, output_size))
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)
    
    def forward_with_intermediates(self, x):
        """Forward pass that returns intermediate activations after each hidden layer"""
        intermediates = []
        current = x
        
        # Process layers in pairs (Linear + Activation)
        for i in range(0, len(self.model) - 1, 2):  # Skip the final output layer
            if i + 1 >= len(self.model):
                break
            linear = self.model[i]
            activation_fn = self.model[i + 1]
            current = activation_fn(linear(current))
            intermediates.append(current)
        
        # Final output layer (no activation)
        final_linear = self.model[-1]
        final_output = final_linear(current)
        
        return final_output, intermediates

n_training_run = 4
num_hidden_layers = 2
hidden_layer_width = 32  # width of each hidden layer
model = Net(input_size=1, hidden_layer_width=hidden_layer_width,
           num_hidden_layers=num_hidden_layers, output_size=1)

# Load the trained model
model.load_state_dict(torch.load(f"best_model_{num_hidden_layers}_dim1.pt"))

grid_tensor = torch.from_numpy(x).float()
model.eval()

with torch.no_grad():
    final_output, intermediates = model.forward_with_intermediates(grid_tensor)
    
    # Compute norms for each hidden layer
    layer_norms = []
    rates_of_change = []
    for i, layer_output in enumerate(intermediates):
        norms = torch.norm(layer_output, dim=1).numpy()
        layer_norms.append(norms)
        rate_of_change = torch.norm(layer_output[:-1] - layer_output[1:], dim=1)
        rates_of_change.append(rate_of_change)

# Plot the norms
plt.figure(figsize=(12, 8))

# Plot each hidden layer's norm
colors = ['blue', 'green', 'orange', 'purple', 'brown', 'pink']


for i, norms in enumerate(layer_norms):
    color = colors[i % len(colors)]
    plt.plot(x.flatten(), norms, color=color, label=f'Hidden Layer {i+1} Norm', linewidth=2)

plt.xlabel('Input')
plt.ylabel('Layer Output Norm')
plt.title('Norm of Hidden Layer Outputs vs Input')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()


for i, roc in enumerate(rates_of_change):
    color = colors[i % len(colors)]
    plt.plot(x[1:].flatten(), roc, color=color, label=f'Hidden Layer {i+1} ROC', linewidth=2)

plt.xlabel('Input')
plt.ylabel('Rate of change')
plt.title('Norm of Derivative of Layer Outputs vs Input')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

# Optional: Also plot the original prediction for comparison
plt.figure(figsize=(12, 8))
Z_pred = final_output.numpy()
plt.plot(x, Z_pred, color='red', label='Model Prediction', linewidth=2)
plt.scatter(x, y, color='black', s=1, alpha=0.5, label='Training Data')
plt.xlabel('Input')
plt.ylabel('Output')
plt.title('Model Prediction vs Training Data')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()