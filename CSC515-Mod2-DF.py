import cv2
import numpy as np
from matplotlib import pyplot as plt

image = cv2.imread("Mod2_CT.jpg")


# Show image dimensions
print("Image shape:", image.shape)


# Define points on the original image
pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
# Define points movement
pts2 = np.float32([[60, 60], [210, 60], [60, 210]])

# Get the affine transform matrix
M_affine = cv2.getAffineTransform(pts1, pts2)

# Apply affine transform
rows, cols, ch = image.shape
affine_transformed = cv2.warpAffine(image, M_affine, (cols, rows))

# Convert and display
affine_rgb = cv2.cvtColor(affine_transformed, cv2.COLOR_BGR2RGB)
plt.imshow(affine_rgb)
plt.title("Affine Transformed Image")
plt.axis("off")
plt.show()
