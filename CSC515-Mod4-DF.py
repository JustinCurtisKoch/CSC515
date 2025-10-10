import cv2
import numpy as np

# === Load the image ===
image = cv2.imread("Mod4_DF.png")
if image is None:
    raise ValueError("Image not found. Check your path.")

# === Apply Filters ===
# 1. Gaussian Blur
gaussian_blur = cv2.GaussianBlur(image, (9, 9), 0)

# 2. Laplacian Filter
laplacian = cv2.Laplacian(image, cv2.CV_64F)
laplacian = cv2.convertScaleAbs(laplacian)

# 3. Sharpening Filter
sharpen_kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])
sharpened = cv2.filter2D(image, -1, sharpen_kernel)

# === Resize function ===
def resize(img, scale=0.7):
    return cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)))

# Resize all images
resized_original = resize(image)
resized_gaussian = resize(gaussian_blur)
#resized_highpass = resize(high_pass)
resized_laplacian = resize(laplacian)
resized_sharpened = resize(sharpened)

# === Stack resized images side by side ===
combined = np.hstack((resized_original, resized_gaussian, resized_laplacian, resized_sharpened))

# === Display ===
cv2.imshow("Original | Gaussian Blur | Laplacian | Sharpened", combined)
cv2.waitKey(0)
cv2.destroyAllWindows()

