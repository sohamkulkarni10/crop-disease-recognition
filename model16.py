import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import cv2
import os
import uuid


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
    st.markdown("""     Crop diseases are significant challenges in agriculture, impacting crop yield, quality, and overall food security worldwide. These diseases are caused by various pathogens, including fungi, bacteria, viruses, and nematodes, as well as environmental factors like nutrient deficiencies and unfavorable weather conditions. Common crop diseases include leaf spot, blight, rust, and wilt, which affect major crops like wheat, rice, maize, and vegetables. Early detection and accurate diagnosis are crucial for effective management and control, often involving the use of pesticides, crop rotation, resistant crop varieties, and advanced techniques like deep learning-based image analysis for disease prediction and prevention.
                 
Crop diseases pose a severe threat to global agriculture, leading to substantial economic losses and jeopardizing food security. These diseases can affect plants at any stage of growth, reducing crop yield, quality, and market value. They are caused by various factors, including pathogens like fungi, bacteria, viruses, and nematodes, as well as abiotic stresses such as nutrient deficiencies, pollution, and climate extremes. For example, fungal diseases like powdery mildew and rust can spread rapidly under humid conditions, while bacterial diseases like blight and canker thrive in wet environments. Viral infections, such as the mosaic virus, are often transmitted by insect vectors, compounding the difficulty of controlling outbreaks.

Crop diseases can manifest as symptoms such as leaf discoloration, wilting, stunted growth, fruit rot, and even plant death. Staple crops like rice, wheat, maize, and potatoes are particularly vulnerable, with diseases like rice blast, wheat rust, and potato late blight causing devastating losses. Effective management requires an integrated approach, including crop rotation, use of resistant varieties, proper irrigation, and timely application of fungicides or insecticides. Modern advancements, such as machine learning and image-based diagnostic tools, enable farmers to detect diseases early, improving precision in treatment and minimizing environmental impact. By combining traditional agricultural practices with cutting-edge technology, the agricultural sector can better combat crop diseases and ensure sustainable food production.
                """)

# Disease prediction page
elif app_mode == "Disease prediction":
    st.header("Disease Recognition")
    test_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if st.button("Show Image"):
        if test_image:
            st.image(test_image, use_container_width=True)

# Function to capture and save an image
def capture_and_save_image():
    # Define the folder to save captured images
    save_folder = "./captured_images"

    # Ensure the folder exists
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # Initialize the camera
    cam = cv2.VideoCapture(0)

    if not cam.isOpened():
        st.error("Could not access the camera. Please ensure it's connected.")
        return

    st.info("Camera started. Press 'Capture Image' to save an image or 'Stop Camera' to exit.")

    img_counter = 0
    while True:
        ret, frame = cam.read()
        if not ret:
            st.error("Failed to capture a frame. Exiting...")
            break

        # Display the live feed in Streamlit
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        st.image(frame_rgb, channels="RGB", caption="Live Camera Feed")

        # Buttons for capturing and stopping
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Capture Image"):
                # Generate a unique filename for the image
                img_name = f"image_{uuid.uuid4().hex}.png"
                print('Saved image ',img_name)
                img_path = os.path.join(save_folder, img_name)

                # Save the image
                cv2.imwrite(img_path, frame)
                st.success(f"Image saved as: {img_path}")
                img_counter += 1

        with col2:
            if st.button("Stop Camera"):
                st.info("Camera stopped.")
                break

    # Release the camera
    cam.release()

# Streamlit App UI
st.title("Live Camera Capture App")
if st.button("Start Camera"):
    capture_and_save_image()
    
    # Prediction
if st.button("Predict"):
    if not test_image:
        st.error("Please upload an image or capture a photograph first.")
    else:
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
            #if os.path.exists(temp_image_path):
                #os.remove(temp_image_path)
