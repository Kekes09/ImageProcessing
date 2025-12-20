# Classifier/train_classical.py
import os
from glob import glob
import numpy as np
from tqdm import tqdm
import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

from Classifier.features import read_gray_bgr, feat_hog, feat_lbp

DATA_DIRS = {
    "train": "data/Train",
    "val": "data/Validation",
    "test": "data/Test",
}
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

def list_images(root_dir):
    classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
    paths, labels = [], []
    for idx, c in enumerate(classes):
        c_paths = []
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"):
            c_paths += glob(os.path.join(root_dir, c, ext))
        paths += c_paths
        labels += [idx] * len(c_paths)
    return paths, np.array(labels, dtype=np.int64), classes

def build_dataset(paths, labels, feature_fn):
    X, y = [], []
    bad = 0
    for p, lab in tqdm(list(zip(paths, labels)), total=len(paths), desc="Extract features"):
        g = read_gray_bgr(p)
        if g is None:
            bad += 1
            continue
        X.append(feature_fn(g))
        y.append(lab)
    if bad:
        print("WARNING: skipped unreadable images:", bad)
    X = np.vstack([x[None, :] for x in X])
    y = np.array(y, dtype=np.int64)
    return X, y

def main():
    tr_paths, tr_y, classes = list_images(DATA_DIRS["train"])
    va_paths, va_y, _ = list_images(DATA_DIRS["val"])
    te_paths, te_y, _ = list_images(DATA_DIRS["test"])

    print("Classes:", classes)
    print("Train:", len(tr_paths), "Val:", len(va_paths), "Test:", len(te_paths))

    # --- HOG + SVM (калибруем, чтобы были вероятности) ---
    Xtr_h, ytr = build_dataset(tr_paths, tr_y, feat_hog)
    Xva_h, yva = build_dataset(va_paths, va_y, feat_hog)
    Xte_h, yte = build_dataset(te_paths, te_y, feat_hog)

    hog_svm = Pipeline([
        ("scaler", StandardScaler()),
        ("svc", CalibratedClassifierCV(LinearSVC(max_iter=20000), cv=3)),
    ])
    hog_svm.fit(Xtr_h, ytr)
    print("\n=== HOG+SVM (val) ===")
    print(classification_report(yva, hog_svm.predict(Xva_h), target_names=classes))
    print("\n=== HOG+SVM (test) ===")
    print(classification_report(yte, hog_svm.predict(Xte_h), target_names=classes))

    # --- LBP + LogisticRegression ---
    Xtr_l, ytr2 = build_dataset(tr_paths, tr_y, feat_lbp)
    Xva_l, yva2 = build_dataset(va_paths, va_y, feat_lbp)
    Xte_l, yte2 = build_dataset(te_paths, te_y, feat_lbp)

    lbp_lr = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=2000)),
    ])
    lbp_lr.fit(Xtr_l, ytr2)
    print("\n=== LBP+LR (val) ===")
    print(classification_report(yva2, lbp_lr.predict(Xva_l), target_names=classes))
    print("\n=== LBP+LR (test) ===")
    print(classification_report(yte2, lbp_lr.predict(Xte_l), target_names=classes))

    joblib.dump({"model": hog_svm, "classes": classes}, os.path.join(MODELS_DIR, "hog_svm.joblib"))
    joblib.dump({"model": lbp_lr, "classes": classes}, os.path.join(MODELS_DIR, "lbp_lr.joblib"))
    print("\nSaved classical models to models/")

if __name__ == "__main__":
    main()
