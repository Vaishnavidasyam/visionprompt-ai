# src/load_data.py

import os
import cv2
import numpy as np

DATA_DIR = "../data"
IMG_SIZE = (224, 224)

CLASSES = [
    "airplane", "animals", "bird",
    "car", "cat", "dog", "flower"
]

X = []
y = []

print("Loading and resizing images...")
for label in CLASSES:
    class_path = os.path.join(DATA_DIR, label)
    print(f"  Loading {label}...")

    if not os.path.exists(class_path):
        print(f"    {class_path} does not exist, skipping.")
        continue

    for img_name in os.listdir(class_path):
        img_path = os.path.join(class_path, img_name)
        img = cv2.imread(img_path)

        if img is not None:
            img = cv2.resize(img, IMG_SIZE)
            X.append(img)
            y.append(label)

X = np.array(X, dtype=np.float32) / 255.0
y = np.array(y)

print("Data shapes:", X.shape, y.shape)

os.makedirs("../model", exist_ok=True)
np.save("../model/X.npy", X)
np.save("../model/y.npy", y)

print("Data saved to ../model/X.npy, ../model/y.npy")