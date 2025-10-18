import cv2
import numpy as np

# === Load fingerprint image ===
img = cv2.imread("fingerprint2.png", cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError("Could not load fingerprint2.png")

# === Step 1: Preprocessing ===
# Light blur to reduce noise without losing ridge detail
blurred = cv2.GaussianBlur(img, (1, 1), 0)

# === Step 2: Adaptive thresholding (Gaussian) ===
binary = cv2.adaptiveThreshold(
    blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY, 13, 3
)

# Invert so ridges are white for morphology operations
binary = cv2.bitwise_not(binary)

# === Step 3: Structuring element (elliptical, slightly elongated) ===
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 1))

# === Step 4: Morphological operations ===
eroded   = cv2.erode(binary, kernel, iterations=1)
dilated  = cv2.dilate(binary, kernel, iterations=1)
opened   = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
closed   = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

# Morphological gradient (edge emphasis)
gradient = cv2.morphologyEx(binary, cv2.MORPH_GRADIENT, kernel)
gradient = cv2.normalize(gradient, None, 0, 255, cv2.NORM_MINMAX)
gradient = cv2.convertScaleAbs(gradient)

# Combine opened + gradient for enhancement
enhanced = cv2.addWeighted(opened, 0.8, gradient, 0.2, 3)
enhanced = cv2.normalize(enhanced, None, 0, 255, cv2.NORM_MINMAX)
enhanced = cv2.convertScaleAbs(enhanced)

# === Step 5: Labeling helper ===
def add_label(image, text):
    labeled = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
    x = image.shape[1] - text_size[0] - 10   # upper-right corner
    y = 20
    cv2.putText(
        labeled, text, (x, y),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA
    )
    return labeled

# === Step 6: Add labels ===
original_labeled = add_label(img, "Original")
eroded_labeled   = add_label(eroded, "Erosion")
dilated_labeled  = add_label(dilated, "Dilation")
opened_labeled   = add_label(opened, "Opening")
closed_labeled   = add_label(closed, "Closing")
enhanced_labeled = add_label(enhanced, "Enhanced")

# === Step 7: Create 2×3 grid ===
top_row    = np.hstack((original_labeled, eroded_labeled, dilated_labeled))
bottom_row = np.hstack((opened_labeled, enhanced_labeled, closed_labeled))
grid_display = np.vstack((top_row, bottom_row))

# === Step 8: Display final grid ===
cv2.imshow("Fingerprint Morphology Grid", grid_display)
cv2.waitKey(0)
cv2.destroyAllWindows()


