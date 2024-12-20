'''
Pushpendra Uikey 23b1023
Nischal 23b1024
Nithin
'''
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import LeavePOut
from sklearn.model_selection import KFold

file_path = 'glass.dat.txt' 
# data = pd.read_csv(file_path, sep=r'\s+')
data = pd.read_table(file_path, sep=r'\s+')

# Extract the 'Al' and 'RI' columns, this will be the data for assinment 
X_train = data['Al'].values.reshape(-1, 1).ravel()  # Aluminum content as input and reshap is to convert it into 1-D array of shape(n,1)
Y_train = data['RI'].values.reshape(-1, 1).ravel()   # Refractive index as target output

def gaussian_kernel(x, xi, h):
    u = (x-xi)/h
    return  ((1 / np.sqrt(2 * np.pi))) * np.exp(-0.5 * u ** 2)

def epanechnikov_kernel(x, xi, h):
    u = (x - xi) / h
    encodn = np.abs(u) <= 1  # Element-wise condition
    return np.where(encodn, 0.75 * (1 - u**2), 0)  # Apply kernel formula where condition is met

# def epanechnikov_kernel(x, xi, h):
#     u = (x-xi)/h
#     if abs(u) <= 1:
#         return 0.75 * (1 - u**2)
#     else:
#         return 0

def triangular_kernel(x, xi, h):
    u = (x - xi) / h
    econdn = np.abs(u) <= 1                       # Element-wise condition
    return np.where(econdn, (1 - np.abs(u)), 0)   # Apply kernel formula where condition is met(in the )

# def triangular_kernel(x, xi, h):
#     u = (x-xi)/h

#     if abs(u)<=1:
#         return ( 1-abs(u) )
#     else:
#         return 0

###############################################################################################################

# using vectorized approach to optimize
def nadaraya_watson(x_train, y_train, x_test, kernel, h, epsilon=1e-8):
    x_train_flat = x_train.flatten()
    # Broadcasting x_test and x_train for vectorized kernel evaluation
    weights = kernel(x_test[:, None], x_train_flat[None, :], h)     # Vectorized computation, broadcasting will handle comutation weights for all x_test[i] at one go
    weights_sum = np.sum(weights, axis=1) + epsilon                 # Summing along the row (test samples)
    y_pred = np.sum(weights * y_train, axis=1) / weights_sum        # Computing predictions for all x_test
    return y_pred

# Below implementation is slow as it loops over to predict each Y as well as and also weights

# def nadaraya_watson(x_train, y_train, x_test, kernel, h, epsilon=1e-8):
#     y_pred = np.zeros(len(x_test)).ravel() 
#     for i, x in enumerate(x_test):
#         weights = np.array([kernel(x, xi, h) for xi in x_train.flatten()])
#         weights_sum = np.sum(weights) + epsilon  # Add epsilon to avoid division by zero
#         y_pred[i] = np.sum(weights * y_train) / weights_sum
#     return y_pred

################################################################################################


def k_fold_cross_validation(x_train, y_train, kernel_function, bandwidths, k=50):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)  # Shuffling and setting a random state for reproducibility
    errors = []         # errors on diff bandwidths

    for h in bandwidths:
        fold_errors = []  # To accumulate errors across folds
        for train_idx, test_idx in kf.split(x_train):
            X_train_fold, X_test_fold = x_train[train_idx], x_train[test_idx]   # train_idx has indices of train sample and test_idx has indices of testing sample
            Y_train_fold, Y_test_fold = y_train[train_idx], y_train[test_idx]

            fold_error = 0
            
            # for i in range(len(X_test_fold)):
                # y_pred = nadaraya_watson(X_train_fold, Y_train_fold, X_test_fold[i].flatten(), kernel_function, h)
                # fold_error += (Y_test_fold[i] - y_pred) ** 2  # MSE for the fold
            

            #the above loop causing too much time, so compute all y_pred in the nadaraya method itself using vector method
            y_pred = nadaraya_watson(X_train_fold, Y_train_fold, X_test_fold.flatten(), kernel_function, h)
            fold_error = np.mean((Y_test_fold - y_pred) ** 2) 


            fold_errors.append(fold_error / len(X_test_fold))  # Average MSE for this fold

        # Average error over all k folds for the given bandwidth
        errors.append(np.mean(fold_errors))

    # best bandwidth (which minimizes error)
    best_bandwidth = bandwidths[np.argmin(errors)]  # argmin returns the index of min val
    return best_bandwidth, errors


####################################################################


def non_paramRegreesion(x_train, y_train, x_values, kernel_func, kernelName):
    # Bandwidth parameter
    std_dev = np.std(X_train)  # Standard deviation of the data
    bandwidths = np.linspace(0.1 * std_dev, 2 * std_dev, 100).ravel()   # Bandwidths from 0.1 to 2 times the std dev

    bandwidth = 1.0
    plt.figure(figsize=(12, 8))


    # Undersmoothed
    bandwidth = bandwidths.min() - 0.01
    Y_pred_undersmoothed=nadaraya_watson(x_train, y_train, x_values.flatten(), kernel_func, bandwidth)
    plt.subplot(2, 2, 1)
    plt.scatter(X_train, Y_train, color='blue', s=5)
    plt.plot(x_values, Y_pred_undersmoothed, label='Undersmoothed', color='r')
    plt.title('Undersmoothed')
    # plt.xlim(min(x_values) - 2, max(x_values) + 2)  # Adjust X-axis limits for better visibility
    # plt.ylim(min(Y_pred_undersmoothed) - 2, max(Y_pred_undersmoothed) + 2)  # Adjust Y-axis limits for better visualization



    # Oversmoothed
    bandwidth = bandwidths.max()
    Y_pred_oversmoothed = nadaraya_watson(x_train, y_train, x_values.flatten(), kernel_func, bandwidth)
    plt.subplot(2, 2, 2)
    plt.scatter(X_train, Y_train, color='blue', s=5)
    plt.plot(x_values, Y_pred_oversmoothed, label='Oversmoothed')
    plt.title('Oversmoothed')
    # plt.xlim(min(x_values) - 0.5, max(x_values) + 0.5)  # Adjust X-axis limits for better visibility
    # plt.ylim(min(Y_pred_oversmoothed) - 1, max(Y_pred_oversmoothed) + 1)  # Adjust Y-axis limits for better visualization

    

    
    # best_h, errors = cross_validation(X_train, y_train, gaussian_kernel, bandwidths)
    best_h, errors = k_fold_cross_validation(x_train=x_train, y_train=y_train, kernel_function=kernel_func,bandwidths=bandwidths)

    # Just-right
    Y_pred_just_right = nadaraya_watson(x_train, y_train, x_values.flatten(), kernel_func, best_h)
    plt.subplot(2, 2, 3)
    plt.scatter(X_train, Y_train, color='blue', s=5)
    plt.plot(x_values, Y_pred_just_right, label='Just-right Fit', color='g')
    plt.title('Just-right fit')
    # plt.xlim(min(x_values) - 0.5, max(x_values) + 0.5)  # Adjust X-axis limits for better visibility
    # plt.ylim(min(Y_pred_just_right) - 1, max(Y_pred_just_right) + 1)  # Adjust Y-axis limits for better visualization


    # Cross-validation curve
    plt.subplot(2, 2, 4)
    plt.plot(bandwidths, errors, label='Cross Validation', color='m')
    plt.xlabel('bandwidth')
    plt.ylabel('estimated risk')
    # plt.xlim(min(x_values) - 0.5, max(x_values) + 0.5)  # Adjust X-axis limits for better visibility
    # plt.ylim(min(errors) - 1, max(errors) + 1)  # Adjust Y-axis limits for better visualization


    plt.tight_layout()
    plt.savefig(f"{kernelName}_kernel_regression.png")
    plt.show()

    print(f"bandwidth corresponding to minimum estimated risk is {best_h}")


# smaple x values to estimate the function(PDF) and fit into Y = f(x) + E
X_values = np.linspace(min(X_train ), max(X_train), 100).reshape(-1,1)

print("Guassian Kernel")
non_paramRegreesion(X_train, Y_train, X_values, gaussian_kernel, "gaussian")
print(f"\nTriangular Kernel")
non_paramRegreesion(X_train, Y_train, X_values, triangular_kernel, "triangular")
print(f"\nEpanechnikov kernel")
non_paramRegreesion(X_train, Y_train, X_values, epanechnikov_kernel, "epanechnikov")









####################################################################################################
################## functions and code used during assignment completion ###########################

# print(X_train[0:3])
# print("----------")
# print(Y_train[0:3])

# Generate a range of x values for prediction
# X_pred = np.linspace(min(X_train), max(X_train), 100)
# y_pred_gaussian = [nadaraya_watson(X_train, Y_train, x, gaussian_kernel, 1) for x in X_pred]

# # Plot
# plt.scatter(X_train, Y_train, color='blue', label='Data points')
# plt.plot(X_pred, y_pred_gaussian, color='red', label='Gaussian Kernel Regression')
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.legend()
# plt.show()



# def Y_prediction(x_train, y_train, x_values, kernel_func, bandwidth):
#     n = len(x_train)
#     y_pred = np.zeros(n)

#     for i in range(n):
#         y_pred[i] = nadaraya_watson(x_train, y_train, x_values[i], kernel_func, bandwidth)
    
#     return y_pred



# def nadaraya_watson(x_train, y_train, x_req, kernel_function, bandwidth):
#     n = len(x_train)
#     weights = np.zeros(n)       # to store weights of each point for given x
    
#     # weights for each point based on the kernel function
#     i=0
#     for x_ in x_train:
#         weights[i] = kernel_function((x_req - x_) / bandwidth)    #  \K_h(x-X_i)
#         i = i+1
    
#     # the weighted sum for the estimator
#     numerator = np.sum(weights * y_train)       # \sigma( K(x-X_i) * Y_i )
#     denominator = np.sum(weights)               # \sigma( K(x-X_i) )
    
#     # Return the estimated value at x_req
#     if denominator == 0:  # Avoid division by zero
#         return 0
#     return numerator / denominator

# print(f"Best bandwidth from k-fold cross-validation is: {best_h}")


#####################################################################################

# def lpocv_cross_validation(x_train, y_train, kernel_function, bandwidths, p=2):
#     n = len(x_train)
#     lpo = LeavePOut(p)
#     errors = []

#     for h in bandwidths:
#         fold_errors = []  # To accumulate errors across p-out splits
#         for train_idx, test_idx in lpo.split(x_train):
#             X_train_fold, X_test_fold = x_train[train_idx], x_train[test_idx]
#             Y_train_fold, Y_test_fold = y_train[train_idx], y_train[test_idx]

#             fold_error = 0
#             for i in range(len(X_test_fold)):
#                 y_pred = nadaraya_watson(X_train_fold, Y_train_fold, X_test_fold[i], kernel_function, h)
#                 fold_error += (Y_test_fold[i] - y_pred) ** 2  # MSE for the fold

#             fold_errors.append(fold_error / len(X_test_fold))  # Average MSE for this fold

#         # Average error over all p-out splits for the given bandwidth
#         errors.append(np.mean(fold_errors))

#     # Return the best bandwidth (minimizes error)
#     best_bandwidth = bandwidths[np.argmin(errors)]
#     return best_bandwidth, errors




#############################################################################

# def cross_validation(x_train, y_train, kernel_function, bandwidths):
#     errors = []
    
#     for h in bandwidths:
#         error = 0
#         for i in range(len(x_train)):
#             # Use all points except the ith one by using indexing
#             X_loo = np.concatenate([x_train[:i], x_train[i+1:]])
#             y_loo = np.concatenate([y_train[:i], y_train[i+1:]])
            
#             # Predict the ith value using the rest of the data
#             y_pred = nadaraya_watson(X_loo, y_loo, np.array([x_train[i]]), kernel_function, h)
            
#             # Accumulate the squared error
#             error += (y_train[i] - y_pred[0]) ** 2
        
#         # Store the average error for this bandwidth
#         errors.append(error / len(x_train))  # Mean Squared Error (MSE)

#     # Return the best bandwidth (minimizes error)
#     best_bandwidth = bandwidths[np.argmin(errors)]
#     return best_bandwidth, errors
