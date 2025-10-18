import cv2
import numpy as np

# === Step 1: Load image ===
img = cv2.imread("Mod6_DF.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# === Step 2: Apply individual filters independently ===

# Laplacian filter (edge enhancement)
laplacian = cv2.Laplacian(gray, cv2.CV_64F)
laplacian = cv2.convertScaleAbs(laplacian)

# Thresholding
_, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# Morphological filtering (opening + closing)
kernel = np.ones((3, 3), np.uint8)
morph = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel, iterations=1)
morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, kernel, iterations=1)

# Region-based filtering
# We'll threshold first, then filter small contours
_, region_mask = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
contours, _ = cv2.findContours(region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
min_area = 500
filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
region_filtered = img.copy()
cv2.drawContours(region_filtered, filtered_contours, -1, (0, 255, 0), 2)

# === Step 3: Convert grayscale images to BGR for consistent stacking ===
lap_bgr = cv2.cvtColor(laplacian, cv2.COLOR_GRAY2BGR)
thresh_bgr = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
morph_bgr = cv2.cvtColor(morph, cv2.COLOR_GRAY2BGR)

# === Step 4: Stack all images horizontally ===
combined = np.hstack((lap_bgr, thresh_bgr, morph_bgr, region_filtered))

# === Step 5: Display results ===
cv2.imshow("Laplacian | Threshold | Morphology | Region Filtering", combined)
cv2.waitKey(0)
cv2.destroyAllWindows()


