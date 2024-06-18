import streamlit as st
import cv2
import numpy as np
import time
from PIL import Image

# Load your pre-trained model (replace 'your_model.h5' with your model file)
# model = load_model('your_model.h5')

# For the purpose of this example, let's use a dummy function for classification
def classify_image(image):
    # Dummy classification logic (replace with your model's prediction logic)
    return "Example Class"

# Define function to capture image
def capture_image():
    cap = cv2.VideoCapture(0)
    time.sleep(1)  # Allow the camera to warm up
    ret, frame = cap.read()
    cap.release()

    # Convert the captured frame to RGB format
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame

# Define function to overlay frame
def overlay_frame(image):
    height, width, _ = image.shape
    overlay = image.copy()
    top_left = (width // 4, height // 4)
    bottom_right = (3 * width // 4, 3 * height // 4)
    cv2.rectangle(overlay, top_left, bottom_right, (0, 255, 0), 3)
    return overlay

st.title("Image Classification App with Frame Overlay")

# Display the camera feed
st.subheader("Live Camera Feed")

# Check if the user has clicked the capture button
if 'captured' not in st.session_state:
    st.session_state.captured = False

# Capture button
if st.button("Capture Image"):
    # Capture the image from the webcam
    captured_image = capture_image()
    
    # Overlay the frame on the captured image
    framed_image = overlay_frame(captured_image)
    
    # Set the session state to indicate that an image has been captured
    st.session_state.captured = True
    st.session_state.captured_image = framed_image

    # Classify the captured image
    classification_result = classify_image(captured_image)
    st.session_state.classification_result = classification_result

# Display the captured image with the frame if an image has been captured
if st.session_state.captured:
    st.image(st.session_state.captured_image, caption="Captured Image with Frame")
    st.subheader("Classification Result")
    st.write(st.session_state.classification_result)
else:
    # Display the live camera feed
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()

    # Convert the captured frame to RGB format
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Overlay the frame on the live camera feed
    live_framed_image = overlay_frame(frame)

    # Display the live camera feed with the frame
    st.image(live_framed_image, caption="Live Camera Feed with Frame")
