from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input

# Load dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize
x_train = x_train / 255.0
x_test = x_test / 255.0

# Build model
model = Sequential([

    Input(shape=(32,32,3)),

    Conv2D(32, (3,3), activation='relu'),

    MaxPooling2D((2,2)),

    Flatten(),

    Dense(64, activation='relu'),

    Dense(10, activation='softmax')

])

# Compile model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train model
model.fit(
    x_train,
    y_train,
    epochs=3
)

# Save model
model.save("vision_model.keras")
print("Model saved successfully!")