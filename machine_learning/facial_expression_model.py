from tensorflow import keras
from tensorflow.keras import layers
import os
import tensorflow as tf

# Assuming input shape of (48, 48, 1) for grayscale images, adjust if needed

model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(7, activation='softmax')  # 7 classes for facial expressions
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

save_dir = 'facial_expression'
os.makedirs(save_dir, exist_ok=True)

# Save the model inside the 'facial_expression' folder
model.save(os.path.join(save_dir, 'model.keras'))

