# CSC515 Portfolio Milestone Module 1
# Introduction to OpenCV (Open-Source Computer Vision Library)

import numpy as np
import cv2 as cv

# Load an image from file
image = cv.imread('Mod1_Numbers.jpg')
cv.namedWindow('Image', cv.WINDOW_NORMAL)
cv.imshow('Image', image)
cv.waitKey(0)
cv.destroyAllWindows()
cv.imwrite('Numbers_mod1.png', image)

#"C:\Users\justi\CSC515\Mod1_Numbers.jpg.jpg"
