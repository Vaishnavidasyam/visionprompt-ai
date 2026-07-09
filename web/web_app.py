import os
import uuid
import numpy as np
import cv2
import pickle
from flask import Flask, render_template, request, jsonify, send_from_directory
from tensorflow.keras.models import load_model

app = Flask(__name__)

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(ROOT_DIR, "model", "classifier.h5")
ENC_PATH = os.path.join(ROOT_DIR, "model", "label_encoder.pkl")
DATA_DIR = os.path.join(ROOT_DIR, "data")
IMG_SIZE = (224, 224)
STATIC_IMG_DIR = os.path.join(ROOT_DIR, "web", "static", "images")
os.makedirs(STATIC_IMG_DIR, exist_ok=True)

print("Loading model and encoder...")
model = load_model(MODEL_PATH)
with open(ENC_PATH, "rb") as f:
    enc = pickle.load(f)
print("Classes:", enc.classes_)


def load_random_image_from_class(class_name, prompt=""):
    folder = os.path.join(DATA_DIR, class_name)
    if not os.path.exists(folder):
        return None, None, None
    choices = os.listdir(folder)
    if not choices:
        return None, None, None
    prompt_words = set(w.strip() for w in prompt.lower().split() if w.strip())
    candidates = [n for n in choices
                  if any(w in os.path.splitext(n)[0].lower() for w in prompt_words)]
    if not candidates:
        candidates = choices
    img_name = np.random.choice(candidates)
    img_path = os.path.join(folder, img_name)
    img = cv2.imread(img_path)
    if img is None:
        return None, None, None
    img_display = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, IMG_SIZE)
    img_tensor = np.expand_dims(img_resized.astype(np.float32) / 255.0, axis=0)
    return img_tensor, img_display, img_path


def classify_from_prompt(prompt):
    prompt = prompt.lower().strip()

    def any_in(keys):
        return any(k in prompt for k in keys)

    if any_in(["cat", "feline", "meow", "kitten", "kitty"]):
        class_name = "cat"
    elif any_in(["dog", "pup", "puppy", "poodle", "canine", "bark"]):
        class_name = "dog"
    elif any_in(["car", "vehicle", "auto", "sedan", "truck", "van", "suv", "taxi"]):
        class_name = "car"
    elif any_in(["airplane", "plane", "jet", "aircraft", "flight", "fly"]):
        class_name = "airplane"
    elif any_in(["bird", "pigeon", "sparrow", "parrot", "crow"]):
        class_name = "bird"
    elif any_in(["flower", "bloom", "blossom", "garden", "petal"]):
        class_name = "flower"
    elif any_in(["animal", "pet", "mammal", "deer", "frog", "horse"]):
        class_name = "animals"
    else:
        class_name = "animals"

    img_tensor, img_display, img_path = load_random_image_from_class(class_name, prompt)
    return class_name, img_tensor, img_display, img_path


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/classify", methods=["POST"])
def classify():
    prompt = request.form.get("prompt", "").strip()
    if not prompt:
        return jsonify({"error": "Prompt is empty."})

    class_name, img_tensor, img_display, img_path = classify_from_prompt(prompt)
    if class_name is None or img_display is None:
        return jsonify({"error": "Could not find any image for this prompt."})

    file_ext = os.path.splitext(img_path)[1]
    web_img_name = f"{uuid.uuid4()}{file_ext}"
    web_img_path = os.path.join(STATIC_IMG_DIR, web_img_name)
    cv2.imwrite(web_img_path, cv2.cvtColor(img_display, cv2.COLOR_RGB2BGR))

    pred = model.predict(img_tensor)[0]
    predicted_idx = np.argmax(pred)
    predicted_label = enc.classes_[predicted_idx]
    confidence = float(pred[predicted_idx])

    web_img_url = f"/static/images/{web_img_name}"

    feedback = f"Showing a {predicted_label} image"
    if "flower" in prompt:
        feedback += " (flower-like)"
    elif "cat" in prompt:
        feedback += " (cat-like)"
    elif "dog" in prompt:
        feedback += " (dog-like)"
    elif "car" in prompt:
        feedback += " (car-like)"

    return jsonify({
        "class": predicted_label,
        "confidence": confidence,
        "image_url": web_img_url,
        "image_path": img_path,
        "feedback": feedback
    })


@app.route("/static/images/<filename>")
def static_img(filename):
    return send_from_directory(STATIC_IMG_DIR, filename)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
