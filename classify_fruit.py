# classify_fruit.py

import fruit_classifier
import cv2
import numpy as np


def load_model(model_weights_path):
    # Load the trained model
    model = fruit_classifier.create_fruit_classifier(input_shape, num_classes)
    model.load_weights(model_weights_path)  # Load the saved model weights
    return model

def preprocess_image(image_path):
    # Preprocess the new image
    image = cv2.imread(image_path)
    image = cv2.resize(image, (100, 100))  # Resize image to match the input shape of the model
    image = image.astype('float32') / 255.0  # Normalize pixel values
    return image.reshape(1, 100, 100, 3)  # Reshape image to match the input shape of the model

def classify_image(model, image_path):
    # Preprocess the new image
    new_image = preprocess_image(image_path)

    # Use the model to make predictions
    predictions = model.predict(new_image)

    # Get the predicted class label
    predicted_class_index = np.argmax(predictions)
    if predicted_class_index == 0:
        predicted_class = "apple"
    elif predicted_class_index == 1:
        predicted_class = "banana"
    else:
        predicted_class = "unknown"

    return predicted_class

# if __name__ == "__main__":
    # Define paths and parameters
input_shape = (100, 100, 3)  # Input shape of the model
num_classes = 2  # Number of classes in the model

model_weights_path = "/Users/somanshumishra/Desktop/ML/ImageClassification/model_weights.weights.h5"  # Path to the saved model weights

    # Load the model
model = load_model(model_weights_path)

    # Define the path to the new image you want to classify
new_image_path = "/Users/somanshumishra/Desktop/ML/ImageClassification/tarin1.jpg"

    # Classify the new image
predicted_class = classify_image(model, new_image_path)

    # Display the prediction result
print("Predicted class:", predicted_class)
