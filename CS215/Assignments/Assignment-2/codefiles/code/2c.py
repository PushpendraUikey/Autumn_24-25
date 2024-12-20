import numpy as np
import matplotlib.pyplot as plt

# Function to generate samples from a normal distribution
def generate_samples(mean, variance, num_samples):
    std_dev = np.sqrt(variance)  # Calculate standard deviation
    samples = np.random.normal(mean, std_dev, num_samples)  # Generate samples
    return samples

# Function to plot the histogram and Gaussian curve
def plot_normal_distribution(mean, variance, color, label, num_samples):
    samples = generate_samples(mean, variance, num_samples)  # Generate samples
    plt.hist(samples, bins=100, density=True, alpha=0.6, color=color, label=label)  # Plot histogram

    # Calculate and plot the Gaussian curve
    std_dev = np.sqrt(variance)
    x_values = np.linspace(min(samples), max(samples), 1000)
    gaussian_curve = (1 / (std_dev * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_values - mean) / std_dev) ** 2)
    plt.plot(x_values, gaussian_curve, color=color, linewidth=2)

# Main part of the script
num_samples = 100000  # Number of samples for each distribution

# Define parameters for the distributions (mean, variance, color, label)
distributions = [
    (0, 0.2, 'blue', 'μ = 0, σ² = 0.2'),
    (0, 1.0, 'red', 'μ = 0, σ² = 1.0'),
    (0, 5.0, 'yellow', 'μ = 0, σ² = 5.0'),
    (-2, 0.5, 'green', 'μ = -2, σ² = 0.5')
]

plt.figure(figsize=(10, 6))  # Create a new figure

# Loop through each distribution and plot
for mean, variance, color, label in distributions:
    plot_normal_distribution(mean, variance, color, label, num_samples)

# Customize the plot
plt.title('Overlay of Normal Distributions')
plt.xlabel('x')
plt.ylabel('Probability Density')
plt.legend()
plt.show()
