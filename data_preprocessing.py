# data_preprocessing.py
import os
import cv2
import numpy as np

def load_data(data_dir):
    """
    Load images and labels from the specified directory.
    
    Args:
    - data_dir: Directory containing subdirectories for each class of images.
    
    Returns:
    - images: List of image arrays.
    - labels: List of corresponding labels.
    """
    images = []
    labels = []
    classes = os.listdir(data_dir)
    print("classes -------- " , classes)
    for class_name in classes:
        print("class_name" ,class_name)
        class_dir = os.path.join(data_dir, class_name)
        if os.path.isdir(class_dir):
            for image_name in os.listdir(class_dir):
                image_path = os.path.join(class_dir, image_name)
                image = cv2.imread(image_path)
                if image is not None:
                    image = cv2.resize(image, (100, 100))  # Resize images to a uniform size
                    images.append(image)
                    labels.append(class_name)  # Use folder name as the label
                else:
                    print(f"Unable to load image: {image_path}")
    return np.array(images), np.array(labels)

def preprocess_data(images, labels):
    """
    Preprocess images and labels.
    
    Args:
    - images: List of image arrays.
    - labels: List of corresponding labels.
    
    Returns:
    - images_preprocessed: Preprocessed image arrays.
    - labels: Labels.
    """
    images_preprocessed = images.astype('float32') / 255.0  # Normalize pixel values to [0, 1]
    return images_preprocessed, labels

def split_data(images, labels, test_size=0.2):
    """
    Split data into training and testing sets.
    
    Args:
    - images: Preprocessed image arrays.
    - labels: Labels.
    - test_size: Fraction of the dataset to include in the testing set.
    
    Returns:
    - X_train: Training images.
    - y_train: Training labels.
    - X_test: Testing images.
    - y_test: Testing labels.
    """
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=test_size, random_state=42)
    return X_train, y_train, X_test, y_test
