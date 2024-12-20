# Checking for any runtime errors and inspecting outputs to validate the code
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Placeholder for dataset paths since the original dataset is not available
train_path = 'train.csv'
test_path = 'test.csv'

# Simulating the dataset with random data for testing purposes
np.random.seed(42)
train = pd.DataFrame(np.random.rand(100, 5), columns=['feature1', 'feature2', 'feature3', 'feature4', 'yield'])
test = pd.DataFrame(np.random.rand(20, 4), columns=['feature1', 'feature2', 'feature3', 'feature4'])

# Preprocessing: Separating features and target variable
X = train.drop(columns=['yield']).values
Y = train['yield'].values

# Scaling the features and target variable
scaler_X = StandardScaler()
scaler_Y = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
Y_scaled = scaler_Y.fit_transform(Y.reshape(-1, 1)).ravel()

# Splitting the data into train and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y_scaled, test_size=0.2, random_state=42)

# Define OLS regression using the normal equation
def ols_regression(X, Y):
    X = np.column_stack([np.ones(X.shape[0]), X])
    return np.linalg.inv(X.T @ X) @ X.T @ Y

# Running OLS regression
beta = ols_regression(X_train, Y_train)

# Gaussian and Epanechnikov Kernel Functions
# def gaussian_kernel(x, xi, h):
#     diff = xi - x.reshape(1, -1)  # Broadcasting happens here
#     u = diff / h
#     return (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * np.sum(u ** 2, axis=1))

# def epanechnikov_kernel(x, xi, h):
#     u = (x - xi) / h
#     mask = np.abs(u) <= 1
#     return np.where(mask, 0.75 * (1 - u ** 2), 0)

# Kernel regression implementation
def kernel_regression(X_train, Y_train, X_test, h, kernel_func):
    predictions = []
    for x in X_test:
        weights = kernel_func(x, X_train, h)
        numerator = np.sum(weights * Y_train)
        denominator = np.sum(weights)
        predictions.append(numerator / denominator if denominator != 0 else 0)
    return np.array(predictions)

def gaussian_kernel(x, xi, h):
    diff = (x - xi) / h  # Broadcasting over all features
    return (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * np.sum(diff ** 2, axis=-1))

def epanechnikov_kernel(x, xi, h):
    diff = (x - xi) / h  # Broadcasting over features
    condition = np.abs(diff) <= 1  # Check if within the kernel boundary for each feature
    return np.where(np.all(condition, axis=-1), 0.75 * (1 - np.sum(diff**2, axis=-1)), 0)

def triangular_kernel(x, xi, h):
    diff = (x - xi) / h
    condition = np.abs(diff) <= 1
    return np.where(np.all(condition, axis=-1), 1 - np.abs(np.sum(diff, axis=-1)), 0)

def nadaraya_watson(x_train, y_train, x_test, kernel, h, epsilon=1e-8):
    # Broadcasting to compute the weights for each test point with each training point
    weights = kernel(x_test[:, None, :], x_train[None, :, :], h)  # Vectorized for multivariate case
    
    weights_sum = np.sum(weights, axis=1) + epsilon  # Sum of weights for each test point
    y_pred = np.sum(weights * y_train, axis=1) / weights_sum  # Weighted sum of predictions
    return y_pred



# Perform kernel regression using Gaussian kernel
h = 1.0
# Y_pred = kernel_regression(X_train, Y_train, X_test, h, gaussian_kernel)
Y_pred = nadaraya_watson(X_train, Y_train, X_test, gaussian_kernel, h)


# Output the results to check correctness
beta, Y_pred[:5]  # Returning first 5 predictions and beta coefficients for inspection
