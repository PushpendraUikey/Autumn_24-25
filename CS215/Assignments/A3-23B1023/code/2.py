"""
Pushpendra 23b1023
Nischal 23b1024
Nithin
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class EpanechnikovKDE:
    def __init__(self, bandwidth=1.0):
        """Initialize the KDE with bandwidth."""
        self.bandwidth = bandwidth
        self.data = None

    def fit(self, data):
        """Fit the KDE model with the given 2D data."""
        self.data = np.array(data)

    # def epanechnikov_kernel(self, u):
    #     """Epanechnikov kernel function."""
    #     norm_u = np.linalg.norm(u, axis=1)
    #     kernel_values = (2 / np.pi) * (1 - norm_u**2)
    #     kernel_values[norm_u > 1] = 0  # Apply the condition ||u||2 <= 1
    #     return kernel_values

    def epanechnikov_kernel(self, u):
        """Epanechnikov kernel function that works for both single points and arrays."""
        # Handle the case for a single point
        if u.ndim == 1:
            norm_u = np.linalg.norm(u)
            if norm_u > 1:
                return 0  # Return 0 if ||u||2 > 1 for a single point
            else:
                return (2 / np.pi) * (1 - norm_u**2)
        else:
            norm_u = np.linalg.norm(u, axis=1)
            kernel_values = (2 / np.pi) * (1 - norm_u**2)
            kernel_values[norm_u > 1] = 0 # the condition ||u||2 <= 1 for arrays
            return kernel_values


    def evaluate(self, x):
        """Evaluate the KDE at a single point x."""
        distances = (x - self.data) / self.bandwidth  # (x - xi) / h
        kernel_values = self.epanechnikov_kernel(distances)
        density = np.sum(kernel_values) / (len(self.data) * self.bandwidth**2)
        return density

# Load the 2D data from the NPZ file
data_file = np.load('transaction_data.npz')
data = data_file['data']  # Assuming 'data' is a (n, 2) array representing 2D points

# Initialize the EpanechnikovKDE class with a bandwidth
epan_kde = EpanechnikovKDE(bandwidth=1)

# Fit the data
epan_kde.fit(data)

# Creating a grid of points to evaluate the KDE
x_min, x_max = min(data[:, 0]) - 1, max(data[:, 0]) + 1
y_min, y_max = min(data[:, 1]) - 1, max(data[:, 1]) + 1
x_vals, y_vals = np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100)
X, Y = np.meshgrid(x_vals, y_vals)

# Evaluating the KDE on the grid using vectorized computation
Z = np.zeros(X.shape)
grid_points = np.c_[X.ravel(), Y.ravel()]   # Create (100x100, 2) grid points

for i, gp in enumerate(grid_points):
    Z.ravel()[i] = epan_kde.evaluate(gp)    # Vectorized evaluation

# reshape Z back to the grid shape
Z = Z.reshape(X.shape)

# Plot the 3D KDE surface
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis')

ax.set_title('Epanechnikov KDE - Transaction Distribution')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Density')

# Save the plot
plt.savefig('transaction_distribution.png')
plt.show()


