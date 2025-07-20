import streamlit as st 
from PIL import Image, ImageOps
import numpy as np
import tensorflow.keras as keras
import os
import time

# --- Configuration ---
MODEL_PATH = "keras_model.h5"
LABELS_PATH = "labels.txt"
IMAGE_SIZE = (224, 224)

# --- Page Settings ---
st.set_page_config(page_title="Parkinson's Clock Test", layout="centered")

# --- Validation Disclaimer at the Top ---
st.warning("\u26a0\ufe0f This application has not been clinically validated. Results must not be used as a substitute for a professional medical diagnosis.")

# --- Styling ---
st.markdown("""
<style>
/* Light Theme Lock */
html, body, [data-theme] {
    color-scheme: light !important;
    background-color: #f6f9fc !important;
}

@keyframes slowGradientShift {
  0% { background-position: 0% 50%; }
  50% { background-position: 100% 50%; }
  100% { background-position: 0% 50%; }
}

.stApp {
    background: linear-gradient(200deg, #eef6fa, #f8fbfe, #e9f3f7, #f6f9fc);
    background-size: 300% 300%;
    background-attachment: fixed;
    animation: slowGradientShift 3s ease infinite;
    font-family: 'Segoe UI', sans-serif;
    color: #355c60;
    padding-bottom: 5rem;
}

@keyframes waveAnimation {
  0% { background-position: 0 0; }
  100% { background-position: 1000px 0; }
}

.wave-background {
  position: fixed;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  background: url('https://svgshare.com/i/uY_.svg');
  background-repeat: repeat-x;
  background-size: 1000px 100%;
  animation: waveAnimation 20s linear infinite;
  opacity: 0.05;
  z-index: -1;
}

.banner {
    background: linear-gradient(135deg, #91c8ea, #d4eefc);
    color: #003049;
    text-align: center;
    padding: 2.5rem 1rem;
    border-radius: 16px;
    box-shadow: 0 12px 24px rgba(0,0,0,0.08);
    margin-bottom: 3rem;
}

h1 { font-size: 2.6rem; font-weight: 700; }
h2, h3 { color: #264653; font-weight: 600; }
p, li {
    font-size: 1.05rem;
    line-height: 1.6;
    color: #344955;
}

.card {
    background-color: white;
    padding: 2rem;
    border-radius: 12px;
    box-shadow: 0 8px 20px rgba(0, 0, 0, 0.05);
    margin-bottom: 2rem;
}

img {
    border-radius: 12px;
    box-shadow: 0 6px 18px rgba(0,0,0,0.05);
}

[data-testid="stFileUploader"] {
    border: 2px dashed #90caf9;
    border-radius: 12px;
    padding: 1.5rem;
    background-color: rgba(255,255,255,0.97);
    margin-bottom: 1.5rem;
}

.disclaimer {
    font-size: 0.9rem;
    color: #607d8b;
    margin-top: 4rem;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="wave-background"></div>', unsafe_allow_html=True)

# --- Load Model ---
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found: {MODEL_PATH}")
        return None
    try:
        return keras.models.load_model(MODEL_PATH, compile=False)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# --- Load Labels ---
@st.cache_data
def load_labels():
    if not os.path.exists(LABELS_PATH):
        st.error(f"Labels file not found: {LABELS_PATH}")
        return None
    try:
        with open(LABELS_PATH, "r") as f:
            return [line.strip() for line in f.readlines()]
    except Exception as e:
        st.error(f"Error loading labels: {e}")
        return None

# --- Preprocess Image ---
def preprocess_image(image):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = ImageOps.fit(image, IMAGE_SIZE, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized = (image_array.astype(np.float32) / 127.0) - 1
    return np.expand_dims(normalized, axis=0)

# --- Predict ---
def predict_parkinsons(image, model, class_names):
    if model is None:
        return "Error: Model not loaded", 0.0
    input_image = preprocess_image(image)
    prediction = model.predict(input_image)
    index = np.argmax(prediction)
    return class_names[index], prediction[0][index]

# --- Banner ---
st.markdown('<div class="banner"><h1>Parkinson\'s Disease Detector</h1><p>AI-powered tool for analyzing clock drawings</p></div>', unsafe_allow_html=True)

# --- Info Section ---
st.markdown("### What is the Clock Drawing Test?")

st.markdown("""
The **Clock Drawing Test** is a widely used screening tool... [truncated for brevity]
""")

# --- Drawing Instructions ---
st.markdown("""<hr style="border: 1px solid #dcdcdc; margin-top: 2rem; margin-bottom: 1rem;">""", unsafe_allow_html=True)
st.markdown("""
<div style='background-color: #ffffff; padding: 1.5rem; border-radius: 16px; box-shadow: 0 6px 12px rgba(0,0,0,0.05);'>
<h3 style='margin-top: 0;'>Drawing Instructions</h3>
<ul>
  <li>Draw an <strong>analog clock</strong> showing the time <strong>7 o'clock</strong>.</li>
  <li>Include <strong>all numbers</strong> from 1 to 12.</li>
  <li>Ensure the <strong>hour and minute hands</strong> are clear and correctly placed.</li>
  <li>Keep the drawing neat and centered.</li>
  <li>If drawn on paper, take a clear, well-lit photo with no shadows or blur.</li>
</ul>
</div>
""", unsafe_allow_html=True)

# --- Example Clock ---
st.markdown("### Example Clock Drawing")
try:
    img = Image.open("clock_example.png")
    st.image(img, caption="Sample Clock Drawing (7 o'clock)", width=220)
except:
    st.warning("Example image not found. Please place 'clock_example.png' in the same folder.")

# --- Upload ---
st.markdown("""
<div style='background-color: #ffffff; padding: 1.5rem; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.04); margin-top: 2rem;'>
<h3 style='margin-top: 0;'>Upload Your Clock Drawing</h3>
<p style='margin-bottom: 1rem;'>Please upload a clear photo...</p>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

# --- Load model & predict ---
model = load_model()
class_names = load_labels()
if model is None or class_names is None:
    st.stop()

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        st.markdown("### Uploaded Image")
        st.image(image, caption="Uploaded Clock Drawing", width=300)

        with st.spinner("Running AI analysis on your clock drawing..."):
            time.sleep(2.5)
            predicted_class, confidence_score = predict_parkinsons(image, model, class_names)

        st.success(f"**Prediction:** {predicted_class}")
        st.info(f"**The system is {confidence_score:.0%} confident in this result.**")

        if predicted_class.strip() == "May have Parkinson's Disease":
            st.warning("This drawing may show signs of Parkinson's disease...")
        elif predicted_class.strip() == "May have Alzheimer's Disease":
            st.warning("This drawing may show signs of Alzheimer's disease...")
        elif predicted_class.strip() == "Invalid Input":
            st.error("The uploaded image is not a valid clock drawing.")
        else:
            st.success("This clock drawing appears typical.")

    except Exception as e:
        st.error(f"Failed to open image: {e}")

# --- How it works ---
st.markdown("""<hr style="border: 1px solid #dcdcdc; margin-top: 2rem; margin-bottom: 1rem;">""", unsafe_allow_html=True)
with st.expander("\u2699\ufe0f **How This App Works**", expanded=False):
    st.markdown("""... (keep content as is)""")

# --- Learn More ---
st.markdown("### \ud83d\udd0e Learn More About Parkinson's Disease")
st.markdown("[WHO – Parkinson Disease](https://www.who.int/news-room/fact-sheets/detail/parkinson-disease)")
st.markdown("### \ud83d\udd0e Learn More About Alzheimer’s Disease")
st.markdown("[NIA – What Is Alzheimer’s Disease?](https://www.nia.nih.gov/health/alzheimers-and-dementia/what-alzheimers-disease)")

# --- Disclaimer ---
st.markdown("<p class='disclaimer'>Disclaimer: This tool is for educational and research purposes only and does not substitute professional medical advice.</p>", unsafe_allow_html=True)


