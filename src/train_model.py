# src/train_model.py

import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# === Paths & config ===
X = np.load("../model/X.npy")
y = np.load("../model/y.npy")

print("Data shapes:", X.shape, y.shape)

enc = LabelEncoder()
y_encoded = enc.fit_transform(y)
y_cat = to_categorical(y_encoded)

print("Classes:", enc.classes_)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_cat, test_size=0.2, random_state=42, stratify=y_encoded
)

print("Train shape:", X_train.shape, y_train.shape)
print("Test shape:", X_test.shape, y_test.shape)

# === Model ===
model = Sequential([
    Conv2D(32, (3,3), activation="relu", input_shape=(224,224,3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation="relu"),
    MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation="relu"),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.5),
    Dense(len(enc.classes_), activation="softmax")
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=10,
    batch_size=16,
    verbose=1
)

# === Save ===
model.save("../model/classifier.h5")

import pickle
with open("../model/label_encoder.pkl", "wb") as f:
    pickle.dump(enc, f)

print("Model saved to ../model/classifier.h5")
print("Label encoder saved to ../model/label_encoder.pkl")