# Classifier/predict.py
import json
import numpy as np
import cv2
import joblib
import tensorflow as tf

from Classifier.features import feat_hog, feat_lbp

HOG = "models/hog_svm.joblib"
LBP = "models/lbp_lr.joblib"
CNN = "models/cnn_mask.keras"
LBL = "models/labels.json"

def _read_rgb(path, size):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Can't read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    return img

def predict_all(image_path: str):
    with open(LBL, "r", encoding="utf-8") as f:
        classes = json.load(f)["class_names"]

    # классика
    pack_h = joblib.load(HOG)
    pack_l = joblib.load(LBP)
    hog_model = pack_h["model"]
    lbp_model = pack_l["model"]

    img128 = _read_rgb(image_path, (128, 128))
    gray = cv2.cvtColor(img128, cv2.COLOR_RGB2GRAY)

    p_hog = hog_model.predict_proba(feat_hog(gray)[None, :])[0]
    p_lbp = lbp_model.predict_proba(feat_lbp(gray)[None, :])[0]

    # CNN
    cnn = tf.keras.models.load_model(CNN)
    img224 = _read_rgb(image_path, (224, 224)).astype(np.float32)
    prob = float(cnn.predict(img224[None, ...], verbose=0)[0][0])  # sigmoid
    p_cnn = np.array([1.0 - prob, prob], dtype=np.float32)

    def pack(p):
        i = int(np.argmax(p))
        return classes[i], float(p[i])

    return {
        "HOG+SVM": pack(p_hog),
        "LBP+LR":  pack(p_lbp),
        "CNN":     pack(p_cnn),
    }
