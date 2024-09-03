'''
author: 23b1023
Pushpendra

'''

import numpy as np
import matplotlib.pyplot as plt
from skimage import io

image_url = "Mona_Lisa.jpg"
image = io.imread(image_url, as_gray=True)  # read as grayscale

# display the Original Image
plt.imshow(image, cmap='gray')
plt.title("Original Image")
plt.show()

# Shift the Image and Calculate Correlation Coefficient Calculation
shift_values = range(-10, 11)  # tx values from -10 to +10
correlation_coefficients = []

# function to compute correlation coefficient
def correlation_coefficient(img1, img2):
    img1_mean = np.mean(img1)
    img2_mean = np.mean(img2)
    numerator = np.sum((img1 - img1_mean) * (img2 - img2_mean))
    denominator = np.sqrt(np.sum((img1 - img1_mean)**2) * np.sum((img2 - img2_mean)**2))
    return numerator / denominator

# Shifting the image
for tx in shift_values:
    shifted_image = np.zeros_like(image)  # Create an empty array for shifted image, it also assign 0 to unoccupied pixels
    if tx > 0:
        shifted_image[:, tx:] = image[:, :-tx]  # Shift right
    elif tx < 0:
        shifted_image[:, :tx] = image[:, -tx:]  # Shift left
    else:
        shifted_image = image  # No shift
    
    # Compute correlation coefficient with the original image
    corr = correlation_coefficient(image, shifted_image)
    correlation_coefficients.append(corr)

    # display shifted images
    plt.imshow(shifted_image, cmap='gray')
    plt.title(f"Shifted Image by {tx} pixels")
    plt.savefig(f"Monalisa-{tx}.png", bbox_inches='tight')
    plt.close()

# Plot Correlation Coefficients
plt.plot(shift_values, correlation_coefficients, marker='o')
plt.xlabel("Shift Value (tx)")
plt.ylabel("Correlation Coefficient")
plt.title("Correlation Coefficient vs Shift")
plt.grid(True)
plt.show()

# Generate Normalized Histogram for Original Image
def compute_histogram(image):
    hist = np.zeros(256)  # Assuming pixel values range from 0 to 255
    for pixel_value in image.flatten():
        hist[int(pixel_value)] += 1
    hist /= image.size  # Normalize histogram
    return hist
    

# Compute histogram
histogram = compute_histogram(image)

# Plot normalized histogram
plt.bar(range(256), histogram, color='gray')
plt.xlabel("Pixel Value")
plt.ylabel("Normalized Frequency")
plt.title("Normalized Histogram of Original Image")
plt.show()
