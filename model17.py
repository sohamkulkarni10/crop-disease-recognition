import streamlit as st
import os
from PIL import Image
import tensorflow as tf
import numpy as np
from datetime import datetime

# Load the trained model
model = tf.keras.models.load_model("trained_model.h5")

# Define class names (update this list based on your dataset)
class_names = ["Disease A", "Disease B", "Healthy"]

# Function to preprocess the image
def preprocess_image(image_path):
    image = Image.open(image_path)
    image = image.resize((224, 224))  # Resize image to match model input size
    image = np.array(image) / 255.0  # Normalize pixel values to [0, 1]
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Function to predict the class of the image
def predict_image(image_path):
    processed_image = preprocess_image(image_path)
    predictions = model.predict(processed_image)
    predicted_class = class_names[np.argmax(predictions)]
    confidence = np.max(predictions) * 100
    return predicted_class, confidence

# Create directories for storing images if they don't exist
os.makedirs("testing_images", exist_ok=True)

def save_captured_image(image_data):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = f"testing_images/captured_image_{timestamp}.jpg"
    with open(file_path, "wb") as f:
        f.write(image_data)
    return file_path

# Streamlit UI
def main():
    st.title("Crop and Disease Recognition")

    st.sidebar.title("Navigation")
    options = ["Home", "Predict Disease"]
    choice = st.sidebar.radio("Go to:", options)

    if choice == "Home":
        st.write("### Welcome to the Crop and Disease Recognition System")
        st.write("This application helps you identify crop diseases by analyzing images of leaves.")
        st.write("You can either upload an image or capture a live photo to predict the crop and its disease.")

    elif choice == "Predict Disease":
        st.write("### Predict Crop Disease")

        # File upload option
        uploaded_file = st.file_uploader("Upload an image of the leaf:", type=["jpg", "jpeg", "png"], key="file_upload")

        # Live capture option
        st.write("Or, capture a live photo:")
        captured_image = st.camera_input("Take a photo:", key="camera_capture")

        # Process uploaded file
        if uploaded_file is not None:
            image_path = f"testing_images/{uploaded_file.name}"
            with open(image_path, "wb") as f:
                f.write(uploaded_file.read())

            st.image(image_path, caption="Uploaded Image", use_column_width=True)
            if st.button("Predict", key="upload_predict_button"):
                predicted_class, confidence = predict_image(image_path)
                st.success(f"Predicted: {predicted_class} with {confidence:.2f}% confidence.")

        # Process captured image
        elif captured_image is not None:
            image_path = save_captured_image(captured_image.getvalue())

            st.image(image_path, caption="Captured Image", use_column_width=True)
            if st.button("Predict", key="capture_predict_button"):
                predicted_class, confidence = predict_image(image_path)
                st.success(f"Predicted: {predicted_class} with {confidence:.2f}% confidence.")

if __name__ == "__main__":
    main()
