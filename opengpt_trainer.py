# Import necessary libraries
import tkinter as tk
from tkinter import filedialog
import tensorflow as tf
import requests  # for downloading images from the internet
import io  # for reading and writing images

from keras.preprocessing.image_dataset import load_image

from opengpt_user import contains_airplane


def create_model():
    # Create a neural network using TensorFlow
    model1 = tf.keras.Sequential()
    model1.add(tf.keras.layers.Conv2D(32, (3, 3), input_shape=(224, 224, 3)))
    model1.add(tf.keras.layers.Activation('relu'))
    model1.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

    model1.add(tf.keras.layers.Conv2D(32, (3, 3)))
    model1.add(tf.keras.layers.Activation('relu'))
    model1.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

    model1.add(tf.keras.layers.Conv2D(64, (3, 3)))
    model1.add(tf.keras.layers.Activation('relu'))
    model1.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

    model1.add(tf.keras.layers.Flatten())
    model1.add(tf.keras.layers.Dense(64))
    model1.add(tf.keras.layers.Activation('relu'))
    model1.add(tf.keras.layers.Dropout(0.5))
    model1.add(tf.keras.layers.Dense(1))
    model1.add(tf.keras.layers.Activation('sigmoid'))

    # Compile the model
    model1.compile(loss='binary_crossentropy',
                   optimizer='rmsprop',
                   metrics=['accuracy'])

    return model1


# Create a neural network using TensorFlow
model = create_model()
# Initialize the GUI
root = tk.Tk()
root.geometry('600x400')
# Create a button for searching for airplane images on the internet
search_button = tk.Button(root, text="Search for airplane images", command=lambda: search())
search_button.pack()
# Create a label for displaying the chosen image file
file_label = tk.Label(root)
file_label.pack()
# Create radio buttons for choosing whether the image contains an airplane
yes_button = tk.Radiobutton(root, text="Yes", value=1, variable=contains_airplane)
yes_button.pack()
no_button = tk.Radiobutton(root, text="No", value=0, variable=contains_airplane)
no_button.pack()
# Create an entry field for the airplane registration
registration_field = tk.Entry(root)
registration_field.pack()
# Create a button for training the neural network
train_button = tk.Button(root, text="Train", command=lambda: train())
train_button.pack()
# Start the GUI
root.mainloop()


# Search for airplane images on the internet
def search():
    # Use a search engine to find airplane images
    results = search_for_airplane_images()
    # Download the first image from the search results
    response = requests.get(results[0])
    image = tk.Image(io.BytesIO(response.content))
    # Display the image in the GUI
    file_label.config(image=image)


# Train the neural network using the chosen image and expected answers
def train(file_path=None):
    # Load the image file
    image = load_image(file_path)

    # Preprocess the image
    image = preprocess_image(image)

    # Get the airplane registration from the entry field
    registration = registration_field.get()

    # Train the neural network
    model.fit(image, (contains_airplane, registration), epochs=1, batch_size=1)

    # Save the trained model
    model.save("airplane_recognition_model.h5")
