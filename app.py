import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# -------------------- PAGE CONFIGURATION --------------------
st.set_page_config(
    page_title="Parkinson's Disease Detector",
    layout="centered",
    initial_sidebar_state="auto"
)

# -------------------- CUSTOM STYLING --------------------
st.markdown("""
    <style>
        /* Background color */
        .stApp {
            background-color: #f5f7fa;
        }

        /* Title Styling */
        h1 {
            font-size: 3rem;
            color: #2c3e50;
            text-align: center;
        }

        /* Subheader Styling */
        h2, h3 {
            color: #34495e;
        }

        /* Upload box styling */
        .upload-style .stFileUploader {
            border: 3px dashed #3498db !important;
            padding: 30px !important;
            border-radius: 15px;
            background-color: #ecf0f1;
            text-align: center;
        }

        /* Center text */
        .center-text {
            text-align: center;
        }

        /* Info box bold text */
        .stInfo {
            font-weight: bold !important;
        }

        /* Hide footer and hamburger */
        footer, header .st-emotion-cache-1dp5vir {
            visibility: hidden;
        }
    </style>
""", unsafe_allow_html=True)

# -------------------- HEADER --------------------
st.title("Parkinson's Disease Detector")
st.subheader("Clock Drawing Test")
st.markdown("Upload a clock drawing to receive a prediction using our trained image classification model.")

# -------------------- INSTRUCTIONS --------------------
st.subheader("Drawing Instructions")
st.markdown("""
To ensure accurate predictions, please follow these instructions:

- Draw an **analog clock** showing the time **7 o'clock**.
- Include **all numbers** from 1 to 12.
- Make sure the **hour and minute hands** are clear.
- Keep the drawing **clean and centered**.
- If drawn on paper, take a **well-lit photo** with no shadows or blur.
""")

# -------------------- UPLOAD SECTION --------------------
st.markdown("### Upload Your Clock Drawing Below")
with st.container():
    with st.container():
        with st.container():
            uploaded_file = st.file_uploader(
                label="Upload Your Clock Drawing Here",
                type=["jpg", "jpeg", "png"],
                label_visibility="visible"
            )

# -------------------- DISPLAY UPLOADED IMAGE --------------------
if uploaded_file:
    st.subheader("Uploaded Image")
    image = Image.open(uploaded_file)
    st.image(image, caption="Clock Drawing Preview", use_column_width=True)

    # -------------------- IMAGE PREPROCESSING --------------------
    image = image.resize((224, 224))  # Adjust based on model input size
    image_array = tf.keras.preprocessing.image.img_to_array(image)
    image_array = tf.expand_dims(image_array, 0) / 255.0  # Normalize

    # -------------------- LOAD MODEL --------------------
    model = tf.keras.models.load_model("model.h5")

    # -------------------- PREDICTION --------------------
    prediction = model.predict(image_array)
    predicted_class = np.argmax(prediction)
    confidence_score = np.max(prediction)

    # -------------------- CLASS MAPPING --------------------
    class_labels = ["Healthy", "Parkinson's Disease", "Unclear/Invalid Drawing"]
    result_text = class_labels[predicted_class]

    # -------------------- RESULT OUTPUT --------------------
    st.subheader("Prediction Result")
    st.success(f"**Prediction: {result_text}**")
    st.info(f"**The system is about {confidence_score:.0%} sure about this result.**")

# -------------------- EXAMPLE DRAWING --------------------
st.subheader("Example Clock Drawing")
st.image("clock_example.png", caption="Example of a Correct Clock Drawing", use_column_width=True)






