import numpy as np
import matplotlib.pyplot as plt

# Number of balls to simulate
num_balls = 100000  

# Different depths for the Galton board simulation
depths = [10, 50, 100]

# Function to simulate the Galton board for a given depth
def simulate_galton_board(depth, num_balls):
    # Initialize an array to store the final position of each ball
    positions = np.zeros(num_balls)

    # Loop over each depth level
    for _ in range(depth):
        # Randomly decide if the ball moves left (-1) or right (+1) for each ball
        moves = np.random.choice([-1, 1], size=num_balls)
        # Update the positions based on the moves
        positions += moves

    return positions

# Function to plot the histogram of final positions
def plot_results(positions, depth, filename):
    # Plot the histogram of the ball positions
    plt.hist(positions, bins=np.arange(positions.min(), positions.max() + 1), density=True, alpha=0.7)
    plt.title(f'Galton Board Simulation (Depth = {depth})')
    plt.xlabel('Final Position')
    plt.ylabel('Probability Density')
    plt.grid(True)
    plt.savefig(filename)  # Save the plot as a file
    plt.show()

# Run the simulation for each depth value and plot the results
for depth in depths:
    positions = simulate_galton_board(depth, num_balls)
    plot_results(positions, depth, f"galton_depth_{depth}.png")
