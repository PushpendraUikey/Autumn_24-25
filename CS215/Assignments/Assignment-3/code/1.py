"""
Pushpendra 23b1023
Nischal 23b1024
Nithin 23b0993
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Part (a) Data Preprocessing and 10-bin Histogram
def load_and_filter_data(file_path):
    # Load the dataset
    data = pd.read_csv(file_path)
    data = data.iloc[:1500]  # Limit to first 1500 rows
    
    filtered_data = data[data['D (Mpc)'] < 4]
    return filtered_data['D (Mpc)']



def plot_histogram(data, bins, file_name):
    hist, edges = np.histogram(data, bins=bins, density=True)
    
    # Calculate estimated probabilities
    bin_width = edges[1] - edges[0]
    estimated_probabilities = hist * bin_width
    print(f"Estimated probabilities for {bins} bins: {estimated_probabilities}")
    
    # Plot histogram
    plt.hist(data, bins=bins, density=True, edgecolor='black')
    plt.title(f'Histogram with {bins} bins')
    plt.xlabel('Distance (Mpc)')
    plt.ylabel('Probability')
    plt.grid(True)
    plt.savefig(file_name)
    plt.close()
    
    return estimated_probabilities


# Part (c) Cross-validation score for bin widths from 1 to 1000
def calculate_cross_validation_score(data, max_bins, file_name):
    n = len(data)
    scores = []
    bins_range = np.arange(1, max_bins + 1)
    
    for bins in bins_range:
        hist, edges = np.histogram(data, bins=bins, density=True)
        bin_width = edges[1] - edges[0]
        
        # Ensure to use the histogram values as probabilities
        p_j = hist * bin_width  # Use hist to calculate p_j directly
        
        term1 = 2 / ((n - 1) * bin_width)
        term2 = ((n + 1) / ((n - 1) * bin_width)) * np.sum(p_j ** 2)
        
        # Calculate score
        score = term1 - term2
        scores.append(score)
    
    # Plot the cross-validation score
    plt.plot(bins_range, scores, label="Cross-validation score")
    plt.xlabel("Number of bins")
    plt.ylabel('Cross-Validation Score as a function of bin width')
    plt.grid(True)
    plt.legend()
    plt.title("Cross-validation score vs Number of bins")
    plt.savefig(file_name)
    plt.close()
    
    return scores


# Part (d) Find the optimal bin width
def find_optimal_bin_width(scores):
    optimal_bins = np.argmin(scores) + 1 
    return optimal_bins

# Part (e) Plot histogram with the optimal bin width and compare with the 10-bin histogram
def plot_optimal_histogram(data, optimal_bins, file_name):
    plot_histogram(data, bins=optimal_bins, file_name=file_name)
    
    
    

# Load and filter data
data = load_and_filter_data('data.csv')  # Update the path to the dataset

# Part (a): Plot 10-bin histogram
estimated_probabilities = plot_histogram(data, bins=10, file_name='10binhistogram.png')

# Part (c): Calculate cross-validation score for bin widths from 1 to 1000
scores = calculate_cross_validation_score(data, max_bins=1000, file_name='crossvalidation.png')

# Part (d): Find optimal bin width
optimal_bins = find_optimal_bin_width(scores)
print(f"Optimal number of bins: {optimal_bins}")

# Part (e): Plot histogram with optimal bin width
plot_optimal_histogram(data, optimal_bins, file_name='optimalhistogram.png')
