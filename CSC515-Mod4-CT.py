# CSC515 Foundations of Computer Vision
# Module 4: Laplacian Filters for Different Kernel Windows

# === Load necessary libraries ===
import cv2
import numpy as np

# === Load the image in grayscale ===
img = cv2.imread("Mod4CT2.jpg", cv2.IMREAD_GRAYSCALE)

# Verify that the image was successfully loaded
if img is None:
    raise FileNotFoundError("Could not load input_image.jpg")

# === Define filter and text parameters ===
sigma = 1.3                            # Standard deviation for Gaussian blur
kernel_sizes = [3, 5, 7]               # List of kernel sizes to test
font = cv2.FONT_HERSHEY_COMPLEX_SMALL  # Font type for labeling
font_scale = 0.7                       # Font size (scaled down for smaller text)
color = (0, 0, 255)                    # Red color for text labels (BGR format)
thickness = 1                          # Text thickness
margin = 25                            # Vertical offset for text placement

# === Prepare a list to store each row of the image grid ===
rows = []

# Loop through each kernel size to apply all filters
for k in kernel_sizes:
    # --- Apply Gaussian filter ---
    gaussian = cv2.GaussianBlur(img, (k, k), sigma)

    # --- Apply Laplacian filter ---
    laplacian = cv2.Laplacian(img, cv2.CV_64F, ksize=k)
    laplacian = cv2.convertScaleAbs(laplacian)  # Convert for display (8-bit)

    # --- Apply Gaussian followed by Laplacian ---
    gauss_then_lap = cv2.Laplacian(gaussian, cv2.CV_64F, ksize=k)
    gauss_then_lap = cv2.convertScaleAbs(gauss_then_lap)

    # --- Define function to label each filtered image ---
    def label_image(image, text):
        # Convert grayscale to BGR so colored text can be applied
        labeled = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        # Add label text in the upper-left corner
        cv2.putText(labeled, text, (10, margin), font, font_scale, color, thickness, cv2.LINE_AA)
        return labeled

    # Label each image with its filter type and kernel size
    g_label = label_image(gaussian, f"Gaussian {k}x{k}")
    l_label = label_image(laplacian, f"Laplacian {k}x{k}")
    gl_label = label_image(gauss_then_lap, f"Gaussian+Laplacian {k}x{k}")

    # --- Combine each filter result horizontally into one row ---
    row = np.hstack((g_label, l_label, gl_label))
    rows.append(row)

# === Stack all rows vertically to form a 3x3 comparison grid ===
grid = np.vstack(rows)

# === Display the final composite grid ===
cv2.imshow("Filter Comparison Grid", grid)
cv2.waitKey(0)          # Wait for a key press to close
cv2.destroyAllWindows() # Clean up display windows
