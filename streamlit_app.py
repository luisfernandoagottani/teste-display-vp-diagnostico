import streamlit as st
import cv2
import numpy as np
import time

# Define function to capture image
def capture_image():
    # Use OpenCV to capture the image from the webcam
    cap = cv2.VideoCapture(0)
    time.sleep(1)  # Allow the camera to warm up
    ret, frame = cap.read()
    cap.release()

    # Convert the captured frame to RGB format
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame

# Define function to add frame to the image
def add_frame_to_image(image, frame_path):
    # Read the frame image
    frame = cv2.imread(frame_path, cv2.IMREAD_UNCHANGED)
    
    # Resize the frame to match the camera image
    frame = cv2.resize(frame, (image.shape[1], image.shape[0]))
    
    # Split the frame into BGR and Alpha channels
    b, g, r, a = cv2.split(frame)
    
    # Create an alpha mask
    alpha_mask = a / 255.0
    
    # Combine the frame and the image
    for c in range(0, 3):
        image[:, :, c] = (1. - alpha_mask) * image[:, :, c] + alpha_mask * frame[:, :, c]

    return image

st.title("Camera Preview with Frame")

# Display the camera feed
st.subheader("Live Camera Feed")

# Capture button
if st.button("Capture Image"):
    # Capture the image from the webcam
    captured_image = capture_image()
    
    # Add frame to the captured image
    frame_path = 'frame.png'  # Path to your frame image
    image_with_frame = add_frame_to_image(captured_image, frame_path)
    
    # Display the captured image with the frame
    st.image(image_with_frame, caption="Captured Image with Frame")
else:
    # Display the live camera feed
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()

    # Convert the captured frame to RGB format
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Add frame to the live camera feed
    frame_path = 'frame.png'  # Path to your frame image
    live_image_with_frame = add_frame_to_image(frame, frame_path)

    # Display the live camera feed with the frame
    st.image(live_image_with_frame, caption="Live Camera Feed with Frame")
