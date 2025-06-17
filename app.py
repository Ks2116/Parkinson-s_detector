import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import tensorflow.keras as keras
import os

# --- Configuration ---
# These paths assume keras_model.h5 and labels.txt are in the same directory as app.py
MODEL_PATH = "keras_model.h5"
LABELS_PATH = "labels.txt"
IMAGE_SIZE = (224, 224)        # Teachable Machine's default input size for images

# --- Function to load and predict ---
@st.cache_resource # Cache the model loading for performance
def load_model():
    """Loads the Teachable Machine Keras model."""
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found: {MODEL_PATH}")
        st.info("Please ensure 'keras_model.h5' is in the same directory as 'app.py'")
        return None
    try:
        model = keras.models.load_model(MODEL_PATH, compile=False)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

@st.cache_data # Cache the labels loading for performance
def load_labels():
    """Loads the class labels from labels.txt."""
    if not os.path.exists(LABELS_PATH):
        st.error(f"Labels file not found: {LABELS_PATH}")
        st.info("Please ensure 'labels.txt' is in the same directory as 'app.py'")
        return None
    try:
        with open(LABELS_PATH, "r") as f:
            class_names = [line.strip() for line in f.readlines()]
        return class_names
    except Exception as e:
        st.error(f"Error loading labels: {e}")
        return None

def preprocess_image(image):
    """Preprocesses the uploaded image for the model."""
    image = ImageOps.fit(image, IMAGE_SIZE, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
    data = np.expand_dims(normalized_image_array, axis=0)
    return data

def predict_parkinsons(image, model, class_names):
    """Makes a prediction using the loaded model."""
    if model is None: # Defensive check if model failed to load
        return "Error: Model not loaded", 0.0

    preprocessed_image = preprocess_image(image)
    prediction = model.predict(preprocessed_image)

    predicted_class_index = np.argmax(prediction)
    predicted_class = class_names[predicted_class_index]
    confidence_score = prediction[0][predicted_class_index]

    return predicted_class, confidence_score

# --- Streamlit UI ---
st.set_page_config(page_title="Parkinson's Detector (Clock Test)", layout="centered")

st.title("Parkinson's Disease Detector (Clock Test)")
st.write("Upload an image of a clock drawing to get a prediction.")
st.markdown("---")

# Load the model and labels
model = load_model()
class_names = load_labels()

if model is None or class_names is None:
    st.stop() # Stop if model or labels couldn't be loaded (e.g., file not found)

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Clock Drawing', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    predicted_class, confidence_score = predict_parkinsons(image, model, class_names)

    st.success(f"**Prediction:** {predicted_class}")
    st.info(f"**Confidence:** {confidence_score:.2f}")

    # Adjust class name based on your labels.txt
    if "Parkinson's" in predicted_class or "parkinson" in predicted_class.lower():
        st.warning("Based on the clock drawing, there might be indicators of Parkinson's disease. Please note this is an experimental AI detection and **should not replace professional medical advice.** Consult a healthcare professional for a proper diagnosis.")
    else:
        st.success("The clock drawing appears typical.")

st.markdown("---")
st.caption("Disclaimer: This tool is for educational and experimental purposes only. It is not a substitute for professional medical diagnosis or treatment.")
