import streamlit as st
import pickle
import numpy as np
from PIL import Image
import cv2

with open("svmmodel.pkl", "rb") as f:
    svmmodel = pickle.load(f)


st.title("Image Classifier App Using SVM")
st.write("Upload an image and let the model predict its class.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Convert PIL image to numpy array
    image_np = np.array(image)

    # Convert to grayscale using OpenCV
    gray_img = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

    # Resize the grayscale image to (64, 64)
    img_size = (64, 64)
    resized_img = cv2.resize(gray_img, img_size)

    # Flatten and normalize
    img_array = resized_img.flatten() / 255.0
    img_array = img_array.reshape(1, -1)


    st.write(f"Processed image shape: {img_array.shape}")


    

    svm_model_prediction = svmmodel.predict(img_array)

    label_map = {0: 'Arborio', 1: 'Basmati', 2: 'Ipsala', 3: 'Jasmine', 4: 'Karacadag'}

    st.write(f"Prediction from SVM model: **{label_map[svm_model_prediction[0]]}**")