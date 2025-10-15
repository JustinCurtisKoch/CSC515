import cv2
import numpy as np

# === Load fingerprint image ===
img = cv2.imread("fingerprint.png", cv2.IMREAD_GRAYSCALE)

if img is None:
    raise FileNotFoundError("Could not load fingerprint.jpg")

# === Step 1: Define a basic 3x3 rectangular kernel ===
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

# === Step 2: Apply Morphological Operations ===
eroded = cv2.erode(img, kernel, iterations=1)
dilated = cv2.dilate(img, kernel, iterations=1)
opened = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
closed = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

# === Step 3: Combine results for side-by-side comparison ===
# Stack images horizontally and vertically for easier viewing
top_row = np.hstack((eroded, dilated))
bottom_row = np.hstack((opened, closed))  # blank slot for alignment
comparison = np.vstack((top_row, bottom_row))

# Add labels for reference
# (optional, helps identify operations if displaying in reports)
def add_label(image, text, position):
    labeled = image.copy()
    cv2.putText(labeled, text, position, cv2.FONT_HERSHEY_SIMPLEX, 
                0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return labeled

# === Step 4: Display Results ===
cv2.imshow("Erosion - Dilation (Top) | Opening - Closing (Bottom)", comparison)
cv2.waitKey(0)
cv2.destroyAllWindows()
