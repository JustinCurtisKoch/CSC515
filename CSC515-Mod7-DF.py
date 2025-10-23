import cv2
import numpy as np

# === Step 1: Generate synthetic image ===
height, width = 400, 400
background_intensity = 100  # light gray background
img = np.full((height, width, 3), background_intensity, dtype=np.uint8)

# Filled shapes
cv2.rectangle(img, (100, 100), (200, 200), color=(200, 5, 5), thickness=-1)
cv2.circle(img, (280, 280), 50, color=(0, 0, 255), thickness=-1)

# === Step 2: Add Gaussian noise ===
mean = 0
std_dev = 15  # adjust for noise intensity
noise = np.random.normal(mean, std_dev, img.shape).astype(np.int16)

noisy_img = img.astype(np.int16) + noise
noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)

# === Step 3: Convert to grayscale ===
gray = cv2.cvtColor(noisy_img, cv2.COLOR_BGR2GRAY)

# === Step 4: Edge detectors ===
canny_edges = cv2.Canny(gray, 100, 200)

sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
sobel_edges = np.uint8(np.clip(cv2.magnitude(sobelx, sobely), 0, 255))
sobel_edges = (sobel_edges > 50).astype(np.uint8) * 255

laplacian = cv2.Laplacian(gray, cv2.CV_64F)
laplacian_edges = np.uint8(np.absolute(laplacian))
laplacian_edges = (laplacian_edges > 50).astype(np.uint8) * 255

# === Step 5: Ground truth edges ===
gt_edges = np.zeros((height, width), dtype=np.uint8)
cv2.rectangle(gt_edges, (100, 100), (200, 200), color=255, thickness=1)
cv2.circle(gt_edges, (280, 280), 50, color=255, thickness=1)
gt_binary = (gt_edges > 0).astype(np.uint8)

# === Step 6: F1 scoring function ===
def edge_f1_score(detected, ground_truth):
    detected_bin = (detected > 0).astype(np.uint8)
    gt_bin = (ground_truth > 0).astype(np.uint8)
    TP = np.sum((detected_bin == 1) & (gt_bin == 1))
    FP = np.sum((detected_bin == 1) & (gt_bin == 0))
    FN = np.sum((detected_bin == 0) & (gt_bin == 1))
    precision = TP / (TP + FP + 1e-6)
    recall = TP / (TP + FN + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    return precision, recall, f1

# === Step 7: Compute F1 for each ===
p_canny, r_canny, f1_canny = edge_f1_score(canny_edges, gt_binary)
p_sobel, r_sobel, f1_sobel = edge_f1_score(sobel_edges, gt_binary)
p_lap, r_lap, f1_lap = edge_f1_score(laplacian_edges, gt_binary)

print("=== Edge Detection Performance (F1 Score) with Noise ===")
print(f"Canny     -> Precision: {p_canny:.3f}, Recall: {r_canny:.3f}, F1: {f1_canny:.3f}")
print(f"Sobel     -> Precision: {p_sobel:.3f}, Recall: {r_sobel:.3f}, F1: {f1_sobel:.3f}")
print(f"Laplacian -> Precision: {p_lap:.3f}, Recall: {r_lap:.3f}, F1: {f1_lap:.3f}")

# === Step 8: Create color overlays ===
def make_overlay(base_img, detected, ground_truth):
    overlay = base_img.copy()
    detected_mask = (detected > 0)
    gt_mask = (ground_truth > 0)
    # Red: true edges
    overlay[gt_mask] = (0, 0, 255)
    # Green: detected edges
    overlay[detected_mask] = (0, 255, 0)
    # Yellow: overlap (correct detections)
    overlap = detected_mask & gt_mask
    overlay[overlap] = (0, 255, 255)
    return overlay

overlay_canny = make_overlay(noisy_img, canny_edges, gt_edges)
overlay_sobel = make_overlay(noisy_img, sobel_edges, gt_edges)
overlay_laplacian = make_overlay(noisy_img, laplacian_edges, gt_edges)

# === Step 9: Display results ===
cv2.imshow("Noisy Image", noisy_img)
cv2.imshow("Canny Overlay", overlay_canny)
cv2.imshow("Sobel Overlay", overlay_sobel)
cv2.imshow("Laplacian Overlay", overlay_laplacian)
cv2.imshow("Ground Truth", gt_edges)

cv2.waitKey(0)
cv2.destroyAllWindows()

