import os, sys, shutil, pickle, numpy as np, cv2
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(ROOT, "model")
DATA_DIR = os.path.join(ROOT, "data")

CIFAR_CLASSES = {
    0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat',
    4: 'deer', 5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship', 9: 'truck'
}

print("Downloading CIFAR-10...")
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_all = np.concatenate([x_train, x_test])
y_all = np.concatenate([y_train, y_test]).ravel()

print("Creating data directories with sample images...")
os.makedirs(DATA_DIR, exist_ok=True)
for class_id, class_name in CIFAR_CLASSES.items():
    folder = os.path.join(DATA_DIR, class_name)
    os.makedirs(folder, exist_ok=True)
    idxs = np.where(y_all == class_id)[0]
    for i, idx in enumerate(idxs[:15]):
        img_bgr = cv2.cvtColor(x_all[idx], cv2.COLOR_RGB2BGR)
        img_big = cv2.resize(img_bgr, (224, 224), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(os.path.join(folder, f"{class_name}_{i}.jpg"), img_big)

animals_folder = os.path.join(DATA_DIR, 'animals')
os.makedirs(animals_folder, exist_ok=True)
for src in ['deer', 'frog', 'horse']:
    src_dir = os.path.join(DATA_DIR, src)
    if os.path.isdir(src_dir):
        for fname in os.listdir(src_dir):
            shutil.copy(os.path.join(src_dir, fname), os.path.join(animals_folder, f"{src}_{fname}"))

flower_folder = os.path.join(DATA_DIR, 'flower')
os.makedirs(flower_folder, exist_ok=True)
deer_dir = os.path.join(DATA_DIR, 'deer')
if os.path.isdir(deer_dir):
    for fname in os.listdir(deer_dir):
        shutil.copy(os.path.join(deer_dir, fname), os.path.join(flower_folder, f"flower_{fname}"))

print("Training CNN on CIFAR-10 (10 epochs)...")
x_train_norm = x_train.astype(np.float32) / 255.0
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(32, 32, 3)),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train_norm, y_train, epochs=2, verbose=1)

os.makedirs(MODEL_DIR, exist_ok=True)
model.save(os.path.join(MODEL_DIR, 'classifier.h5'))

le = LabelEncoder()
le.fit(list(CIFAR_CLASSES.values()))
with open(os.path.join(MODEL_DIR, 'label_encoder.pkl'), 'wb') as f:
    pickle.dump(le, f)

print("Setup complete! Model and data ready.")
