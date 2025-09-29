import cv2
import numpy as np

# === Step 1: Load the image ===
image = cv2.imread("Mod3_CT_Dog.jpg")
cv2.imshow("Original Image", image)
print(image.shape)  # Print image dimensions (height, width, channels)
height, width, channels = image.shape

# === Step 2: Split channels ===
# OpenCV loads images in BGR format
blue_channel, green_channel, red_channel = cv2.split(image)

# Display each channel (grayscale representation)
spilt_image = np.empty([height, width * 3, 3], "uint8")
spilt_image[:, 0:width] = cv2.merge([blue_channel, blue_channel, blue_channel])
spilt_image[:, width:width * 2] = cv2.merge([green_channel, green_channel, green_channel])
spilt_image[:, width * 2:width * 3] = cv2.merge([red_channel, red_channel, red_channel])
cv2.imshow("Split Channels", spilt_image)
cv2.moveWindow("Split Channels", 0, height)

# === Step 3: Merge channels back into original image ===
merged_image = cv2.merge((blue_channel, green_channel, red_channel))
cv2.imshow("Merged Original Image", merged_image)

# === Step 4: Swap red and green channels (GRB order) ===
swapped_image = cv2.merge((blue_channel, red_channel, green_channel))
cv2.imshow("Swapped Red-Green Image (GRB)", swapped_image)

# === Step 5: Display comparison of merged and swapped images ===
comparison = np.hstack((merged_image, swapped_image))
cv2.imshow("Merged vs Swapped", comparison)

# Wait for a key press and close all windows
cv2.waitKey(0)
cv2.destroyAllWindows()
