import os
import cv2
import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

IMG_SIZE = 224
DATASET_PATH = "../dataset"

categories = ["with_mask", "without_mask"]

X = []
y = []

for label, category in enumerate(categories):
    path = os.path.join(DATASET_PATH, category)
    for img_name in os.listdir(path):
        img_path = os.path.join(path, img_name)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            X.append(img)
            y.append(label)

X = np.array(X) / 255.0
y = to_categorical(np.array(y), num_classes=2)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

np.savez("processed_data.npz", X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)