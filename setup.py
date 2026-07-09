import os, sys, shutil, pickle, numpy as np, cv2, urllib.request
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(ROOT, "model")
DATA_DIR = os.path.join(ROOT, "data")

DISPLAY_CLASSES = ['airplane', 'animals', 'bird', 'car', 'cat', 'dog', 'flower']

# Download real high-resolution images for display
def download_images():
    import concurrent.futures
    base_url = "https://loremflickr.com/640/480"
    os.makedirs(DATA_DIR, exist_ok=True)
    for cls in DISPLAY_CLASSES:
        folder = os.path.join(DATA_DIR, cls)
        os.makedirs(folder, exist_ok=True)
        kw = cls if cls != 'animals' else 'animal'
        def dl(i):
            url = f"{base_url}/{kw}?random={i}"
            path = os.path.join(folder, f"{cls}_{i}.jpg")
            try:
                urllib.request.urlretrieve(url, path)
                img = cv2.imread(path)
                if img is None or img.size == 0: raise Exception("bad img")
                return f"{cls}_{i}.jpg"
            except:
                return None
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as ex:
            list(ex.map(dl, range(10)))

print("Downloading high-resolution images...")
download_images()
print("Images downloaded.")

print("Downloading CIFAR-10 for training...")
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

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
model.fit(x_train_norm, y_train, epochs=10, verbose=1)

os.makedirs(MODEL_DIR, exist_ok=True)
model.save(os.path.join(MODEL_DIR, 'classifier.h5'))

CIFAR_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
le = LabelEncoder()
le.fit(CIFAR_CLASSES)
with open(os.path.join(MODEL_DIR, 'label_encoder.pkl'), 'wb') as f:
    pickle.dump(le, f)

print("Setup complete! Model and data ready.")
