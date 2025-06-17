# Cell 1: Create app.py
# This command writes your Streamlit application code to a file named app.py
%%writefile app.py
import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import tensorflow.keras as keras
import os
 
# --- Configuration ---
# These paths assume keras_model.h5 and labels.txt are in the same directory as app.py
MODEL_PATH = "keras_model.h5"
LABELS_PATH = "labels.txt"
IMAGE_SIZE = (224, 224)         # Teachable Machine's default input size for images
 
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
 
```python
# Cell 2: Create requirements.txt
# This command writes your Python dependencies to a file named requirements.txt
%%writefile requirements.txt
streamlit
tensorflow
Pillow
numpy
```python
# Cell 3: Install condacolab and restart runtime
# This cell installs condacolab.
# **IMPORTANT**: After running this cell, Colab will automatically restart the runtime.
# You will need to run the subsequent cells AFTER the restart is complete.
!pip install -q condacolab
import condacolab
condacolab.install()
```python
# Cell 4: Set up the Conda environment and install dependencies
# Run this cell AFTER the runtime has restarted from Cell 3.
%%shell
# Activate the conda shell hook for proper command execution
eval "$(conda shell.bash hook)"
 
# Create a new conda environment with Python 3.9 (a version compatible with TensorFlow 2.x)
# You can try Python 3.10 if you prefer, but 3.9 is generally stable for TF.
conda create -n tf_env python=3.9 -y
 
# Activate the newly created environment
conda activate tf_env
 
# Install specific TensorFlow version and other dependencies from requirements.txt
# TensorFlow 2.10.0 is a good choice for Python 3.9.
pip install tensorflow==2.10.0
pip install -r requirements.txt
 
# Verify the Python and TensorFlow versions (optional, for debugging)
python --version
pip show tensorflow
```python
# Cell 5: Run your Streamlit app
# Run this cell AFTER Cell 4 has completed successfully.
%%shell
# Activate the conda environment again to run the Streamlit app within it
eval "$(conda shell.bash hook)"
conda activate tf_env
 
# Run your Streamlit application
# This will output a public URL that you can click to access your app.
streamlit run app.py
