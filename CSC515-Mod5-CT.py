# CSC515 Foundations of Computer Vision
# Module 5: Morphology Operations for Fingerprint Enhancement

import cv2
import numpy as np

# === Step 1: Load Fingerprint Image ===
img = cv2.imread("fingerprint_smudge.png", cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError("Could not load fingerprint_smudge.png")

# === Step 2: Preprocessing ===
# Apply a very light Gaussian blur to suppress small noise 
# without degrading ridge details.
blurred = cv2.GaussianBlur(img, (1, 1), 0)

# === Step 3: Adaptive Thresholding ===
# Convert image to binary using adaptive Gaussian thresholding.
binary = cv2.adaptiveThreshold(
    blurred, 255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY, 13, 3
)

# Invert so fingerprint ridges appear white — suitable for morphology.
binary = cv2.bitwise_not(binary)

# === Step 4: Morphological Operations ===
# Define an elliptical structuring element (elongated horizontally).
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 1))

# Apply a variety of morphological transformations.
eroded   = cv2.erode(binary, kernel, iterations=1)
dilated  = cv2.dilate(binary, kernel, iterations=1)
opened   = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
closed   = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

# Compute morphological gradient to emphasize ridge edges.
gradient = cv2.morphologyEx(binary, cv2.MORPH_GRADIENT, kernel)
gradient = cv2.normalize(gradient, None, 0, 255, cv2.NORM_MINMAX)
gradient = cv2.convertScaleAbs(gradient)

# Combine opened + gradient images for enhanced contrast.
enhanced = cv2.addWeighted(opened, 0.8, gradient, 0.2, 3)
enhanced = cv2.normalize(enhanced, None, 0, 255, cv2.NORM_MINMAX)
enhanced = cv2.convertScaleAbs(enhanced)

# === Step 5: Labeling Helper Function ===
def add_label(image, text):
    """Add a red label in the upper-right corner of an image."""
    labeled = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
    x = image.shape[1] - text_size[0] - 10  # right align text
    y = 20
    cv2.putText(
        labeled, text, (x, y),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA
    )
    return labeled

# === Step 6: Add Labels to Each Image ===
original_labeled = add_label(img, "Original")
eroded_labeled   = add_label(eroded, "Erosion")
dilated_labeled  = add_label(dilated, "Dilation")
opened_labeled   = add_label(opened, "Opening")
closed_labeled   = add_label(closed, "Closing")
enhanced_labeled = add_label(enhanced, "Enhanced")

# Save enhanced version for reference
cv2.imwrite('enhanced.jpg', enhanced)

# === Step 7: Create 2×3 Grid of Morphology Results ===
top_row    = np.hstack((original_labeled, eroded_labeled, dilated_labeled))
bottom_row = np.hstack((opened_labeled, enhanced_labeled, closed_labeled))
grid_display = np.vstack((top_row, bottom_row))

# === Step 8: SIFT Feature Detection ===
# Reload color version for visualization.
img_color = cv2.imread('fingerprint_smudge.png')
gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

# Create SIFT detector and detect keypoints.
sift = cv2.SIFT_create()
keypoints = sift.detect(gray, None)

# Draw keypoints on the image.
sift_img = cv2.drawKeypoints(gray, keypoints, img_color)

# Save SIFT keypoints visualization.
cv2.imwrite('sift_keypoints.jpg', sift_img)

# === Step 9: Display Results ===
cv2.imshow("Fingerprint Morphology Grid", grid_display)
cv2.imshow("SIFT Keypoints", sift_img)
cv2.waitKey(0)
cv2.destroyAllWindows()


