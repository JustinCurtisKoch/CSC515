# CSC515 Portfolio Milestone Module 1
# Introduction to OpenCV (Open-Source Computer Vision Library)

# Import the OpenCV library
import cv2 as cv

# Load an image from file
image = cv.imread('Mod1_Numbers.jpg')

# Display the image in a window
cv.namedWindow('Image', cv.WINDOW_NORMAL)
cv.imshow('Image', image)

# Wait for a key press and close the window
cv.waitKey(0)
cv.destroyAllWindows()

# Save the image to a new file
cv.imwrite('Numbers_Mod1.png', image)
