import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Schwarzschild radius
def schwarzschild_radius(M):
    G = 1  # Use natural units where G = 1
    return 2 * G * M

# Example parameters
M = 1
r_s = schwarzschild_radius(M)

# Define a grid of points in space (r, theta)
r = np.linspace(r_s * 1.1, 10, 500)
theta = np.linspace(0, 2 * np.pi, 500)
R, Theta = np.meshgrid(r, theta)

# Compute components of the metric tensor at each point
g_rr = (1 - r_s / R)**(-1)
g_tt = (R * np.sin(Theta))**2

# Plot the g_rr component
plt.figure(figsize=(10, 8))
plt.contourf(R, Theta, g_rr, levels=50, cmap='viridis')
plt.colorbar(label='g_rr component')
plt.title('Schwarzschild Metric Tensor Field (g_rr)')
plt.xlabel('r')
plt.ylabel('theta')
plt.show()
