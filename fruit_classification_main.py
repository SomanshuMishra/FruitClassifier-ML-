# fruit_classification_main.py

import fruit_classifier
import data_preprocessing
import matplotlib.pyplot as plt
import numpy as np
import os

# Define the directory containing your dataset
data_dir = "/Users/somanshumishra/Desktop/ML/ImageClassification/fruits/"

# Load and preprocess the dataset
images, labels = data_preprocessing.load_data(data_dir)
print("Number of images:", len(images))
print("Number of labels:", len(labels))
print("Labels:", np.unique(labels))

# Preprocess the data
images_preprocessed, labels = data_preprocessing.preprocess_data(images, labels)

# Convert labels to integers
label_to_int = {label: idx for idx, label in enumerate(np.unique(labels))}
labels_encoded = np.array([label_to_int[label] for label in labels])

# Split data into train and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(images_preprocessed, labels_encoded, test_size=0.2, random_state=42)

# Define input shape and number of classes
input_shape = (100, 100, 3)  # Assuming input images are resized to 100x100 pixels with 3 channels (RGB)
num_classes = len(np.unique(labels))  # Calculate the number of classes dynamically

# Create the model
model = fruit_classifier.create_fruit_classifier(input_shape, num_classes)

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")

# Save the model weights
print("Current Directory:", os.getcwd())
model.save_weights("/Users/somanshumishra/Desktop/ML/ImageClassification/model_weights.weights.h5")

# Visualize training history
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
