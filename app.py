from flask import Flask, render_template, request
import os
import random
import cv2

app = Flask(__name__)

DATASET_PATH = "static"

@app.route("/", methods=["GET", "POST"])
def home():

    image_paths = []

    if request.method == "POST":

        prompt = request.form["prompt"].lower()

        prompt_folder = os.path.join(
            DATASET_PATH,
            prompt
        )

        if os.path.exists(prompt_folder):

            images = [
                file for file in os.listdir(prompt_folder)
                if file.endswith(
                    ('.jpg', '.png', '.jpeg')
                )
            ]

            selected_images = random.sample(
                images,
                min(4, len(images))
            )

            processed_images = []

            for img_name in selected_images:

                image_path = os.path.join(
                    prompt_folder,
                    img_name
                )

                img = cv2.imread(image_path)

                # Resize image
                img = cv2.resize(img, (300,300))

                # Convert grayscale
                gray = cv2.cvtColor(
                    img,
                    cv2.COLOR_BGR2GRAY
                )

                processed_name = (
                    "processed_" + img_name
                )

                processed_path = os.path.join(
                    prompt_folder,
                    processed_name
                )

                cv2.imwrite(
                    processed_path,
                    gray
                )

                processed_images.append(
                    f"{prompt}/{processed_name}"
                )

            image_paths = processed_images

    return render_template(
        "index.html",
        images=image_paths
    )

if __name__ == "__main__":
    app.run(debug=True)