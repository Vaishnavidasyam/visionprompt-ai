import cv2
import os
import random

# User prompt
prompt = input("Enter prompt: ").lower()

dataset_path = "dataset"

prompt_folder = os.path.join(dataset_path, prompt)

if os.path.exists(prompt_folder):

    images = [
        file for file in os.listdir(prompt_folder)
        if file.endswith(('.jpg', '.png', '.jpeg'))
    ]

    if len(images) > 0:

        # Select 4 random images
        selected_images = random.sample(images, min(4, len(images)))

        for image_name in selected_images:

            image_path = os.path.join(prompt_folder, image_name)

            img = cv2.imread(image_path)

            img = cv2.resize(img, (400, 400))

            cv2.imshow(image_name, img)

        cv2.waitKey(0)

        cv2.destroyAllWindows()

    else:
        print("No images found!")

else:
    print("Prompt folder not found!")