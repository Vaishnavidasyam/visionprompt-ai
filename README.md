
# VisionPromptAI

A prompt-driven image retrieval and classification tool built with Python, Flask, and TensorFlow.

---

## Project Description

VisionPromptAI is a simple app where you type a word or phrase (like "cat", "red car", or "rose flower") and it finds a matching image from its local dataset, then runs that image through a deep learning model to predict what it is. The whole idea is to combine image searching with AI classification in one go.

This project was built as a learning experiment — to understand how image classification works end-to-end, from collecting images to training a model and serving predictions through a web interface. Nothing too fancy, just a practical way to see how CNNs (Convolutional Neural Networks) work with real images.

---

## Features

- **Text-to-image lookup** — Type a prompt and the app fetches a relevant image from its dataset using keyword matching
- **AI image classification** — A trained CNN model predicts the category of the image and shows a confidence score
- **Web interface** — A clean dark-themed web page where you can type prompts and see results instantly
- **Command-line demos** — Two CLI scripts included if you prefer terminal over browser
- **7 image categories** — Supports cat, dog, car, airplane, bird, flower, and animals
- **Model training pipeline** — Scripts included to retrain the model with your own images

---

## Technologies Used

- **Python 3.13** — The main programming language
- **Flask** — Runs the web server and handles requests
- **TensorFlow / Keras** — For building and training the CNN model
- **OpenCV** — To read and process images
- **NumPy** — For handling image data as arrays
- **scikit-learn** — Used for label encoding and splitting data
- **HTML, CSS, JavaScript** — Frontend of the web app

---

## Project Structure

Here's a quick look at the important folders and files:

```
VisionPromptAI/
├── app.py                     # Simple Flask app (older version)
├── main.py                    # CLI script (older version)
├── train_model.py             # Standalone CIFAR-10 trainer
├── requirements.txt           # Currently empty (dependencies listed below)
│
├── data/                      # Image dataset organized by category
│   ├── airplane/
│   ├── animals/
│   ├── bird/
│   ├── car/
│   ├── cat/
│   ├── dog/
│   └── flower/
│
├── model/                     # Trained model and related files
│   ├── classifier.h5          # The trained CNN model
│   ├── label_encoder.pkl      # Maps class names to numbers
│   ├── X.npy                  # Preprocessed image data
│   └── y.npy                  # Image labels
│
├── src/                       # Core source code
│   ├── app.py                 # Runs predictions from the command line
│   ├── load_data.py           # Reads images and converts them to .npy files
│   └── train_model.py         # Builds and trains the CNN
│
└── web/                       # Web application
    ├── web_app.py             # Flask server (main web entry point)
    ├── templates/
    │   └── index.html         # The web page you see in the browser
    └── static/images/         # Result images are stored here temporarily
```

---

## Installation

Follow these steps to run the project on your own machine.

### 1. Clone the repository

```bash
git clone https://github.com/Vaishnavidasyam/visionprompt-ai.git
cd visionprompt-ai
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv .venv
```

Activate it:

- **Windows:** `.venv\Scripts\activate`
- **Mac/Linux:** `source .venv/bin/activate`

### 3. Install the required packages

```bash
pip install flask opencv-python tensorflow numpy scikit-learn scipy h5py
```

### 4. (Optional) Retrain the model

If you want to train the model from scratch:

```bash
python src/load_data.py
python src/train_model.py
```

### 5. Run the web app

```bash
python web/web_app.py
```

### 6. Open in browser

Go to `http://127.0.0.1:5000` in your browser.

---

## How to Use

1. Open the web app in your browser
2. Type a prompt in the text box — for example, "show me a cat" or "red car" or "bird flying"
3. Click the button (or press Enter)
4. The app will:
   - Match your prompt to one of the 7 categories
   - Pick a relevant image from the dataset
   - Run it through the CNN model
   - Show you the image along with the predicted class and confidence score
5. Try different prompts to see how the model performs

You can also run the CLI scripts:

```bash
# Try the batch demo
python src/app.py
```

---

## User Roles

This app doesn't have user roles or login system. Anyone who runs it can use all the features. It's a single-user tool.

---

## Screenshots

*(Add a screenshot of the web interface here)*

*(Add a screenshot showing a prediction result here)*

---

## Future Improvements

- Allow users to upload their own images for classification
- Support more image categories and a larger dataset
- Add a camera capture feature for real-time classification
- Improve the prompt matching logic to understand more complex sentences
- Deploy the app online so it can be accessed without installing anything

---

## Author

Created by **Vaishnavidasyam**  
Email: vaishnavi.41469@gmail.com  
GitHub: [@Vaishnavidasyam](https://github.com/Vaishnavidasyam)
