import cv2
import numpy as np

# Load the image in grayscale
img = cv2.imread("Mod7-Full.png", cv2.IMREAD_GRAYSCALE)

# Ensure image is loaded correctly
if img is None:
    raise FileNotFoundError("Could not load input_image.jpg")

# --- Global thresholding ---
# Apply a fixed global threshold (e.g., 127)
_, global_thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

# --- Adaptive Mean Thresholding ---
# The threshold value is the mean of the neighborhood area minus a constant C
adaptive_mean = cv2.adaptiveThreshold(
    img, 
    255, 
    cv2.ADAPTIVE_THRESH_MEAN_C, 
    cv2.THRESH_BINARY, 
    11, 2
)

# --- Adaptive Gaussian Thresholding ---
# Similar to mean but the threshold value is a weighted sum of neighborhood values
adaptive_gauss = cv2.adaptiveThreshold(
    img, 
    255, 
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
    cv2.THRESH_BINARY, 
    11, 2
)

# Display all results side by side for comparison
combined = np.hstack((img, global_thresh, adaptive_mean, adaptive_gauss))

cv2.imshow("Original | Global | Adaptive Mean | Adaptive Gaussian", combined)
cv2.waitKey(0)
cv2.destroyAllWindows()
