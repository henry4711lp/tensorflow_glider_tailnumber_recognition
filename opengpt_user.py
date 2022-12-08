# Import necessary libraries
import tensorflow as tf
import cv2 # for capturing video frames
import tkinter as tk # for the GUI

from PIL import ImageTk

# Load the trained neural network
model = tf.keras.models.load_model("airplane_recognition_model.h5")

# Initialize the GUI
root = tk.Tk()
root.geometry('800x600')

# Create a label for displaying the video feed
frame_label = tk.Label(root)
frame_label.pack()

# Create a label for displaying the results of the neural network
results_label = tk.Label(root)
results_label.pack()

# Start the video capture
cap = cv2.VideoCapture(0)

while True:
    # Capture a frame from the video
    ret, frame = cap.read()

    # Preprocess the frame
    frame = preprocess_frame(frame)

    # Use the neural network to predict whether the frame contains an airplane and its registration
    contains_airplane, registration = model.predict(frame)

    # Draw a blue rectangle around the airplane
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), thickness=2)

    # Draw a red rectangle around the registration
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), thickness=2)

    # Convert the frame to a Tkinter-compatible image
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = ImageTk.PhotoImage(frame)

    # Display the frame in the GUI
    frame_label.config(image=frame)

    # Display the results of the neural network in the GUI
    results_label.config(text="Contains airplane: {}\nRegistration: {}".format(contains_airplane, registration))

# Release the video capture
cap.release()
