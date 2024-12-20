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

    def epanechnikov_kernel(self, u):
        """Epanechnikov kernel function for a single point u."""
        norm_u = np.linalg.norm(u)  # norm of a single 2D point
        
        if norm_u <= 1:
            return (2 / np.pi) * (1 - norm_u**2)  # Kernel value if ||u||2 <= 1
        else:
            return 0  # Return 0 if the norm exceeds 1

    def evaluate(self, x):
        """Evaluate the KDE at a single point x."""
        kernel_values = np.zeros(len(self.data))  # Store kernel values
        
        for i, xi in enumerate(self.data):
            # Compute (x - xi) / bandwidth
            u = (x - xi) / self.bandwidth
            kernel_values[i] = self.epanechnikov_kernel(u)  # Apply kernel
        
        # Compute KDE estimate: average the kernel values
        density = np.sum(kernel_values) / (len(self.data) * self.bandwidth**2)
        return density



####################### part 2.2 

# Load the 2D data from the NPZ file
data_file = np.load('transaction_data.npz')
data = data_file['data']  # 'data' is a (n, 2) array representing 2D points

# Initialize the EpanechnikovKDE class with a bandwidth
epan_kde = EpanechnikovKDE(bandwidth=1)

# Fit the data
epan_kde.fit(data)

# Create a grid of points to evaluate the KDE
x_min, x_max = min(data[:, 0]) - 1, max(data[:, 0]) + 1
y_min, y_max = min(data[:, 1]) - 1, max(data[:, 1]) + 1
x_vals, y_vals = np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100)
X, Y = np.meshgrid(x_vals, y_vals)

# Evaluate the KDE on the grid
Z = np.zeros(X.shape)
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        Z[i, j] = epan_kde.evaluate(np.array([X[i, j], Y[i, j]]))

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



# import numpy as np
# import matplotlib.pyplot as plt

# # Custom Epanechnikov KDE class
# class EpanechnikovKDE:
#     def __init__(self, bandwidth=1.0):
#         self.bandwidth = bandwidth
#         self.data = None

#     def fit(self, data):
#         """Fit the KDE model with the given data."""
#         self.data = data

#     def epanechnikov_kernel(self, x, xi):
#         """Epanechnikov kernel function."""
#         u = (x - xi) / self.bandwidth
#         if abs(u) <= 1:
#             return 2 * (1 - u**2) * 1/np.pi
#         else:
#             return 0.0

#     def evaluate(self, x):
#         """Evaluate the KDE at point x."""
#         result = 0
#         for d in self.data:
#             result += self.epanechnikov_kernel(x, d) / self.bandwidth

#         result = result / len(self.data)
#         return result


# # Load the data from the NPZ file
# data_file = np.load('transaction_data.npz')
# data = data_file['data']

# print(len(data))
# # TODO: Initialize the EpanechnikovKDE class
# epan_kde = EpanechnikovKDE(0.8)

# # TODO: Fit the data
# epan_kde.fit(data)

# # TODO: Plot the estimated density in a 3D plot
# x_vals = np.linspace(min(data) - 1, max(data) + 1, 1000)
# kde_vals = np.array([epan_kde.evaluate(x) for x in x_vals])

# plt.plot(x_vals, kde_vals, label='Undersmoothed', color='r')
# plt.title('Undersmoothed')
# plt.xlim(min(data) - 0.5, max(data) + 0.5)  # Adjust X-axis limits for better visibility
# plt.ylim(0, max(kde_vals) + 1)  # Adjust Y-axis limits for better visualization

# # TODO: Save the plot 
# plt.show()