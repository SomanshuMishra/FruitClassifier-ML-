import tensorflow as tf
from tensorflow.keras import layers, models


# Define the CNN architecture
def create_fruit_classifier(input_shape, num_classes):
    model = models.Sequential()

    # Specify input shape using an Input layer
    model.add(layers.Input(shape=input_shape))

    # Convolutional layers
    model.add(layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    # Flatten layer to transition from convolutional to fully connected layers
    model.add(layers.Flatten())

    # Dense (fully connected) layers
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))  # Output layer with softmax activation for multi-class classification

    return model
# Set input shape and number of classes
input_shape = (100, 100, 3)  # Assuming input images are resized to 100x100 pixels with 3 channels (RGB)
num_classes = 3  # Assuming we have 3 classes: apple, orange, banana

# Create the model
model = create_fruit_classifier(input_shape, num_classes)

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Display model summary
model.summary()
