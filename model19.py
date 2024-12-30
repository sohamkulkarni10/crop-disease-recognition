import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import cv2
import os

# Prediction function
def model_prediction(test_image):
    model = tf.keras.models.load_model('trained_model.h5')
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128), color_mode="rgb")
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])
    prediction = model.predict(input_arr)
    result_index = np.argmax(prediction)
    return result_index

# Sidebar
st.sidebar.title("DASHBOARD")
app_mode = st.sidebar.selectbox("SELECT PAGE", ["Home page", "Disease prediction"])

# Home page
if app_mode == "Home page":
    st.header("PLANT DISEASE RECOGNITION SYSTEM")
    image_path = "crop.jpeg"
    st.image(image_path, use_container_width=True)
    st.markdown("""
    Crop diseases are significant challenges in agriculture, impacting crop yield, quality, and overall food security worldwide. These diseases are caused by various pathogens, including fungi, bacteria, viruses, and nematodes, as well as environmental factors like nutrient deficiencies and unfavorable weather conditions. Early detection and accurate diagnosis are crucial for effective management and control, often involving the use of pesticides, crop rotation, resistant crop varieties, and advanced techniques like deep learning-based image analysis for disease prediction and prevention.
    """)

# Disease prediction page
elif app_mode == "Disease prediction":
    st.header("Disease Recognition")
    
    # Option to upload an image
    test_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    # Capture Photograph Button
    if st.button("Capture Photograph"):
        cam = cv2.VideoCapture(0)
        st.info("Camera opened. Please capture the photograph.")

        test_folder = "test"  # Specify the folder to save images
        if not os.path.exists(test_folder):  # Ensure the test folder exists
            os.makedirs(test_folder)

        ret, frame = cam.read()
        if ret:
            img_name = "captured_image.png"  # Name of the captured image
            img_path = os.path.join(test_folder, img_name)
            cv2.imwrite(img_path, frame)
            st.success(f"Image successfully captured and saved to {img_path}")
            st.image(frame, channels="BGR", caption="Captured Image")

            # Predict using the captured image
            result_index = model_prediction(img_path)
            class_name = [
                'Apple___Apple_scab',
                'Apple___Black_rot',
                'Apple___Cedar_apple_rust',
                'Apple___healthy',
                'Blueberry___healthy',
                'Cherry_(including_sour)___Powdery_mildew',
                'Cherry_(including_sour)___healthy',
                'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
                'Corn_(maize)___Common_rust_',
                'Corn_(maize)___Northern_Leaf_Blight',
                'Corn_(maize)___healthy',
                'Grape___Black_rot',
                'Grape___Esca_(Black_Measles)',
                'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
                'Grape___healthy',
                'Orange___Haunglongbing_(Citrus_greening)',
                'Peach___Bacterial_spot',
                'Peach___healthy',
                'Pepper,_bell___Bacterial_spot',
                'Pepper,_bell___healthy',
                'Potato___Early_blight',
                'Potato___Late_blight',
                'Potato___healthy',
                'Raspberry___healthy',
                'Soybean___healthy',
                'Squash___Powdery_mildew',
                'Strawberry___Leaf_scorch',
                'Strawberry___healthy',
                'Tomato___Bacterial_spot',
                'Tomato___Early_blight',
                'Tomato___Late_blight',
                'Tomato___Leaf_Mold',
                'Tomato___Septoria_leaf_spot',
                'Tomato___Spider_mites Two-spotted_spider_mite',
                'Tomato___Target_Spot',
                'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
                'Tomato___Tomato_mosaic_virus',
                'Tomato___healthy'
            ]
            st.success(f"Model's prediction is {class_name[result_index]}")

            # Optionally remove the captured image after prediction
            if os.path.exists(img_path):
                os.remove(img_path)
        else:
            st.error("Failed to capture the image.")

        cam.release()

    # Prediction from uploaded image
    if test_image:
        # Save the uploaded image temporarily
        temp_image_path = "temp_uploaded_image.png"
        with open(temp_image_path, "wb") as f:
            f.write(test_image.getbuffer())

        # Predict using the uploaded image
        result_index = model_prediction(temp_image_path)
        class_name = [
            'Apple___Apple_scab',
            'Apple___Black_rot',
            'Apple___Cedar_apple_rust',
            'Apple___healthy',
            'Blueberry___healthy',
            'Cherry_(including_sour)___Powdery_mildew',
            'Cherry_(including_sour)___healthy',
            'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
            'Corn_(maize)___Common_rust_',
            'Corn_(maize)___Northern_Leaf_Blight',
            'Corn_(maize)___healthy',
            'Grape___Black_rot',
            'Grape___Esca_(Black_Measles)',
            'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
            'Grape___healthy',
            'Orange___Haunglongbing_(Citrus_greening)',
            'Peach___Bacterial_spot',
            'Peach___healthy',
            'Pepper,_bell___Bacterial_spot',
            'Pepper,_bell___healthy',
            'Potato___Early_blight',
            'Potato___Late_blight',
            'Potato___healthy',
            'Raspberry___healthy',
            'Soybean___healthy',
            'Squash___Powdery_mildew',
            'Strawberry___Leaf_scorch',
            'Strawberry___healthy',
            'Tomato___Bacterial_spot',
            'Tomato___Early_blight',
            'Tomato___Late_blight',
            'Tomato___Leaf_Mold',
            'Tomato___Septoria_leaf_spot',
            'Tomato___Spider_mites Two-spotted_spider_mite',
            'Tomato___Target_Spot',
            'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
            'Tomato___Tomato_mosaic_virus',
            'Tomato___healthy'
        ]
        st.success(f"Model's prediction is {class_name[result_index]}")

        # Remove the temporary image after prediction
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)
