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

# --- Subtle Professional Styling ---
st.markdown("""
<style>
.stApp {
 background: linear-gradient(200deg, #c987c9, #d469a5, #e57490, #ff9696, #ffd97e);
 background-attachment: fixed;
    font-family: 'Segoe UI', sans-serif;
    color: #edf4f2;
    padding-bottom: 5rem;
}

.card {
    background-color: white;
    padding: 2rem;
    border-radius: 12px;
    box-shadow: 0 8px 20px rgba(0, 0, 0, 0.06);
    margin-bottom: 2rem;
}

.banner {
    background: linear-gradient(135deg, #5c6bc0, #42a5f5);
    color: white;
    text-align: center;
    padding: 2.5rem 1rem;
    border-radius: 16px;
    box-shadow: 0 12px 24px rgba(0,0,0,0.12);
    margin-bottom: 3rem;
}

h1 {
    font-size: 2.6rem;
    font-weight: 700;
    margin-bottom: 0.5rem;
}
h2, h3 {
    color: #37474f;
    font-weight: 600;
    margin-top: 2rem;
}

p, li {
    font-size: 1.05rem;
    line-height: 1.6;
    color: #263238;
}

img {
    border-radius: 12px;
    box-shadow: 0 6px 18px rgba(0,0,0,0.06);
}

[data-testid="stFileUploader"] {
    border: 2px dashed #90caf9;
    border-radius: 12px;
    padding: 1.5rem;
    background-color: rgba(255,255,255,0.95);
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

# --- Drawing Instructions ---
st.markdown("### Drawing Instructions")
st.write("""
To ensure accurate predictions, please follow these instructions:

- Draw an **analog clock** showing the time **7 o'clock**.
- Include **all numbers** from 1 to 12.
- Make sure the **hour and minute hands** are clear.
- Keep the drawing clean and centered.
- If drawn on paper, take a well-lit photo with no shadows or blur.
""")

# --- Example Clock ---
st.markdown("### Example Clock Drawing")
try:
    img = Image.open("clock_example.png")
    st.image(img, caption="Sample Clock Drawing (7 o'clock)", width=220)
except:
    st.warning("Example image not found. Please place 'clock_example.png' in the same folder.")

# --- Upload Drawing ---
st.markdown("### Upload Your Clock Drawing")
st.write("Please upload a clear photo of your 7 o'clock clock drawing.")
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

# --- Load Model & Labels ---
model = load_model()
class_names = load_labels()
if model is None or class_names is None:
    st.stop()

# --- Handle Uploaded File ---
if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)

        st.markdown("### Uploaded Image")
        st.image(image, caption="Uploaded Clock Drawing", width=300)

        with st.spinner("Analyzing image..."):
            time.sleep(2)
            predicted_class, confidence_score = predict_parkinsons(image, model, class_names)

        st.success(f"**Prediction:** {predicted_class}")
        st.info(f"**The system is {confidence_score:.0%} confident in this result.**")

        if predicted_class.strip() == "May have Parkinson's Disease":
            st.warning("This drawing may show signs of Parkinson's disease. Please consult a medical professional.")
        elif predicted_class.strip() == "May have Alzheimer's Disease":
            st.warning("This drawing may show signs of Alzheimer's disease. Consider consulting a doctor.")
        elif predicted_class.strip() == "Invalid Input":
            st.error("The uploaded image is not a valid clock drawing. Please upload a proper one.")
        else:
            st.success("This clock drawing appears typical.")

    except Exception as e:
        st.error(f"Failed to open image: {e}")

# --- How It Works ---
st.markdown("---")
with st.expander("How This App Works", expanded=False):
    st.markdown("""
**Step-by-step Process:**

- **Upload Clock Drawing**  
  Submit a hand-drawn or digital clock image.

- **Preprocessing**  
  The image is resized and normalized.

- **Prediction**  
  Our model classifies the image as:
    - May have Parkinson's Disease
    - May have Alzheimer's Disease
    - Typical
    - Invalid Input

- **Results**  
  You’ll receive a prediction and a confidence score.

> Note: This tool is experimental and not a replacement for clinical diagnosis.
""")

# --- Disclaimer ---
st.markdown("<p class='disclaimer'>Disclaimer: This tool is for educational and research purposes only and does not substitute professional medical advice.</p>", unsafe_allow_html=True)
