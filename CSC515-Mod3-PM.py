# CSC515 Foundations of Computer Vision
# Module 3: Drawing Functions in OpenCV
 
import cv2

# === Load image ===
img = cv2.imread('Glasses.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# === Load cascades ===
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml")

# === Detect faces ===
faces = face_cascade.detectMultiScale(gray, 1.1, 4)
print('Number of detected faces:', len(faces))

for (x, y, w, h) in faces:
    # Draw a green oval around the face
    center = (x + w // 2, y + h // 2)
    axes = (w // 2, int(h * 0.6))  # taller oval
    cv2.ellipse(img, center, axes, 0, 0, 360, (0, 255, 0), 2)

    # Define region of interest for eyes
    roi_gray = gray[y:y + h, x:x + w]
    roi_color = img[y:y + h, x:x + w]

    # Detect eyes
    eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 5)
    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 0, 255), 2)

# === Add text (upper-right corner) ===
text = "Chuck Close"
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.8
color = (255, 0, 255)  # purple (BGR)
thickness = 2

# Get image width and height
(h_img, w_img) = img.shape[:2]
(text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)

# Position text 10 px from top-right corner
pos_x = w_img - text_w - 10
pos_y = text_h + 10
cv2.putText(img, text, (pos_x, pos_y), font, font_scale, color, thickness)

# === Show image ===
cv2.imshow('Face and Eye Detection', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

