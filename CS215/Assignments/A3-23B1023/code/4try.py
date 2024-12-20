import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import LeavePOut

class EpanechnikovKDE:
    def __init__(self, bandwidth=1.0):
        self.bandwidth = bandwidth
        self.data = None

    def fit(self, data):
        """Fit the KDE model with the given data."""
        self.data = data

    def epanechnikov_kernel(self, x, xi):
        """Epanechnikov kernel function."""
        arg = (x - xi) / self.bandwidth
        if abs(arg) <= 1:  
            return 0.75 * (1 - arg**2)
        else:
            return 0
        
    def evaluate(self,x):
        estimate = 0
        for xi in self.data:
            estimate += self.epanechnikov_kernel(x,xi)
        estimate /= len(self.data)*self.bandwidth
        return estimate

class GaussianKDE:
    def __init__(self, bandwidth=1.0):  
        self.bandwidth = bandwidth
        self.data = None

    def fit(self, data):
        self.data = data

    def gaussian_kernel(self, x, xi):
        arg = (x - xi) / self.bandwidth
        return math.exp(-arg**2 / 2) / math.sqrt(2 * math.pi) 
    
    def evaluate(self,x):
        estimate = 0
        for xi in self.data:
            estimate += self.gaussian_kernel(x,xi)
        estimate /= len(self.data)*self.bandwidth
        return estimate
    
def cross_validate_Ep(data,bandwidths, p=1):
    lpo = LeavePOut(p)
    errors = []

    for bandwidth in bandwidths:
        kde = EpanechnikovKDE(bandwidth=bandwidth)
        fold_errors = []
        
        for train_idx, test_idx in lpo.split(data):
            train_data = data[train_idx]
            test_data = data[test_idx]
            
            kde.fit(train_data)
            density_estimates = kde.evaluate(test_data)
            
            # Calculate error (for example, using Mean Squared Error)
            error = np.mean((density_estimates - np.ones_like(test_data))**2)  # Here we assume true density is uniform
            fold_errors.append(error)
        
        # Average error across folds
        errors.append(np.mean(fold_errors))
    
    return errors

def cross_validate_G(data,bandwidths, p=1):
    lpo = LeavePOut(p)
    errors = []

    for bandwidth in bandwidths:
        kde = GaussianKDE(bandwidth=bandwidth)
        fold_errors = []
        
        for train_idx, test_idx in lpo.split(data):
            train_data = data[train_idx]
            test_data = data[test_idx]
            
            kde.fit(train_data)
            density_estimates = kde.evaluate(test_data)
            
            # Calculate error (for example, using Mean Squared Error)
            error = np.mean((density_estimates - np.ones_like(test_data))**2)  # Here we assume true density is uniform
            fold_errors.append(error)
        
        # Average error across folds
        errors.append(np.mean(fold_errors))
    
    return errors

# Load data
data = pd.read_csv('glass.dat.txt', delim_whitespace=True)
x = data['Al'].to_numpy()
y = data['RI'].to_numpy()

x_validate = np.random.sample(100)
bandwidths_train = np.linspace(0.1, 3.0, 30)
errors = cross_validate_Ep(x_validate, bandwidths_train)
optimal_bandwidth = bandwidths_train[np.argmin(errors)]

# Epanechnikov estimate
fig, axs = plt.subplots(2, 2,figsize=(12,7))

# Create KDE plots
bandwidths = [0.8,0.05, optimal_bandwidth]
for idx, bw in enumerate(bandwidths):
    kde = EpanechnikovKDE(bandwidth=bw)
    kde.fit(x)
    x_range = np.linspace(0, 3.5, 350)
    f = np.empty_like(x_range)

    for i in range(len(x_range)):
        num = 0
        den = 0
        for j in range(len(x)):
            num += kde.epanechnikov_kernel(x_range[i], x[j]) * y[j]
            den += kde.epanechnikov_kernel(x_range[i], x[j])
        f[i] = num / den if den != 0 else 0

    axs[idx // 2, idx % 2].plot(x_range, f)
    axs[idx // 2, idx % 2].scatter(x, y,s=10)
    if bw == 0.8:
        axs[idx // 2, idx % 2].set_title('Oversmooth h=0.8')
    elif bw == 0.1:
        axs[idx // 2, idx % 2].set_title('Undersmooth h=0.1')
    else:
        axs[idx // 2, idx % 2].set_title('Optimal h=0.5')
    
axs[1,1].plot(bandwidths_train,errors)
axs[1,1].set_title('Cross validaton')

plt.tight_layout()
plt.show()

x_validate = np.random.sample(100)
bandwidths_train = np.linspace(0.1, 3.0, 30)
errors = cross_validate_G(x_validate, bandwidths_train)
optimal_bandwidth = bandwidths_train[np.argmin(errors)]

# Gaussian estimate
fig2, axs2 = plt.subplots(2, 2,figsize=(12,7))

# Create Gaussian KDE plots
bandwidths = [0.8,0.05, optimal_bandwidth]
for idx, bw in enumerate(bandwidths):
    kde = GaussianKDE(bandwidth=bw)
    kde.fit(x)
    x_range = np.linspace(0, 3.5, 350)
    f = np.empty_like(x_range)

    for i in range(len(x_range)):
        num = 0
        den = 0
        for j in range(len(x)):
            num += kde.gaussian_kernel(x_range[i], x[j]) * y[j]
            den += kde.gaussian_kernel(x_range[i], x[j])
        f[i] = num / den if den != 0 else 0

    axs2[idx // 2, idx % 2].plot(x_range, f)
    axs2[idx // 2, idx % 2].scatter(x, y,s=10)
    if bw == 0.8:
        axs[idx // 2, idx % 2].set_title('Oversmooth h=0.8')
    elif bw == 0.1:
        axs[idx // 2, idx % 2].set_title('Undersmooth h=0.1')
    else:
        axs[idx // 2, idx % 2].set_title('Optimal h=0.5')

axs2[1,1].plot(bandwidths_train,errors)
axs2[1,1].set_title('Cross validaton')

plt.tight_layout()
plt.show()