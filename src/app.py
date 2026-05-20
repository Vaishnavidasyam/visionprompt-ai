# src/app.py

import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import pickle

# === Paths ===
MODEL_PATH   = "../model/classifier.h5"
ENCODER_PATH = "../model/label_encoder.pkl"
DATA_DIR     = "../data"
IMG_SIZE     = (224, 224)

CLASSES = [
    "airplane", "animals", "bird",
    "car", "cat", "dog", "flower"
]

# === Load model and encoder ===
print("Loading model and label encoder...")
model = load_model(MODEL_PATH)

with open(ENCODER_PATH, "rb") as f:
    enc = pickle.load(f)

print("Classes:", enc.classes_)

# === Helper: load random image + tensor for a class ===
def load_random_image_from_class(class_name):
    folder = os.path.join(DATA_DIR, class_name)
    if not os.path.exists(folder):
        return None, None, None

    choices = os.listdir(folder)
    if not choices:
        return None, None, None

    img_name = np.random.choice(choices)
    img_path = os.path.join(folder, img_name)

    img = cv2.imread(img_path)
    if img is None:
        return None, None, None

    img_display = img.copy()
    img = cv2.resize(img, IMG_SIZE)
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)

    return img, img_display, img_path

# === Prompt → class mapping (works for ANY prompt) ===
def classify_from_prompt(prompt):
    prompt = prompt.lower().strip()

    def any_in(keys):
        return any(k in prompt for k in keys)

    # Prioritize specific keywords first
    if any_in(["cat", "feline", "meow", "kitten", "kitty"]):
        class_name = "cat"
    elif any_in(["dog", "pup", "puppy", "poodle", "canine", "bark"]):
        class_name = "dog"
    elif any_in([
        "car", "vehicle", "auto", "sedan", "truck",
        "van", "suv", "motor car", "taxi"
    ]):
        class_name = "car"
    elif any_in(["airplane", "plane", "jet", "aircraft", "flight", "fly"]):
        class_name = "airplane"
    elif any_in(["bird", "pigeon", "sparrow", "parrot", "crow"]):
        class_name = "bird"
    elif any_in(["flower", "bloom", "blossom", "garden", "petal"]):
        class_name = "flower"
    elif any_in(["animal", "pet", "mammal"]):
        class_name = "animals"
    else:
        # Fallback: pick any class by default
        class_name = "animals"

    img_tensor, img_display, img_path = load_random_image_from_class(class_name)
    return class_name, img_tensor, img_display, img_path

# === Demo: prompt → image + prediction ===
def demo_prompt(prompt):
    print(f"\nPrompt: '{prompt}'")

    class_name, img_tensor, img_display, img_path = classify_from_prompt(prompt)
    if img_display is None:
        print(f"❌ No images found for class '{class_name}'.")
        return

    cv2.imshow("Input image (from prompt)", img_display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if img_tensor is None:
        print("❌ Could not load image tensor.")
        return

    pred = model.predict(img_tensor)[0]
    predicted_idx = np.argmax(pred)
    predicted_label = enc.classes_[predicted_idx]
    confidence = pred[predicted_idx]

    print(f"✅ Predicted class: {predicted_label}")
    print(f"   Confidence: {confidence:.2f}")
    print(f"   Image path: {img_path}")

if __name__ == "__main__":
    prompts = [
        "Show me a cat",
        "Find a dog",
        "Display a car",
        "Show me a flower",
        "Find an airplane",
        "Find a bird",
        "animal photo",
        "random text that should go to animals"
    ]

    for p in prompts:
        demo_prompt(p)