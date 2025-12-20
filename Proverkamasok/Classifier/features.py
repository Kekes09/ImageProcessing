# Classifier/features.py
import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern

IMG_SIZE = (128, 128)
LBP_P = 8
LBP_R = 1

def read_gray_bgr(path: str):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        return None
    img = cv2.resize(img, IMG_SIZE, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray

def feat_hog(gray: np.ndarray) -> np.ndarray:
    return hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True,
    ).astype(np.float32)

def feat_lbp(gray: np.ndarray) -> np.ndarray:
    lbp = local_binary_pattern(gray, P=LBP_P, R=LBP_R, method="uniform")
    n_bins = LBP_P + 2  # фиксированная длина! (иначе vstack падает)
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
    return hist.astype(np.float32)
