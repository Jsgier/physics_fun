import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the energy density function (example)
def energy_density(x, y, z):
    # Example energy density function (you can define your own)
    return np.sin(x) * np.cos(y) * np.sin(z)

# Define the spatial grid
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
z = np.linspace(-5, 5, 100)
X, Y, Z = np.meshgrid(x, y, z)

# Calculate energy density at each spatial point
energy_density_values = energy_density(X, Y, Z)

# Plot the energy density distribution in 3D
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X, Y, Z, c=energy_density_values, cmap='viridis')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Energy Density Distribution in 3D Space')
plt.show()
