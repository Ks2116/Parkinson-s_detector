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

# --- Custom CSS ---
st.markdown("""
<style>
.stApp {
    background: linear-gradient(120deg, #f8fbff 0%, #ecf3f9 100%);
    font-family: 'Segoe UI', sans-serif;
    padding-bottom: 5rem;
}
h1 {
    font-size: 2.5rem;
    font-weight: 700;
    margin-bottom: 1.2rem;
    color: #1a2b3c;
    text-align: center;
}
h2, h3 {
    color: #2c3e50;
    margin-top: 2rem;
    font-weight: 600;
}
p, li {
    font-size: 1.05rem;
    line-height: 1.6;
    color: #34495e;
}
ul {
    padding-left: 1.2rem;
    margin-top: 0.5rem;
}
ul li {
    margin-bottom: 0.6rem;
}
[data-testid="stNotification"] {
    font-size: 1.05rem;
}
img {
    border-radius: 8px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.04);
}
.disclaimer {
    font-size: 0.9rem;
    color: #7f8c8d;
    margin-top: 4rem;
    text-align: center;
}
# --- Elegant Custom CSS ---
st.markdown("""
<style>
/* Overall App Styling */
.stApp {
    background: linear-gradient(135deg, #e3f2fd 0%, #f8f9fa 100%);
    font-family: 'Segoe UI', 'Helvetica Neue', sans-serif;
    padding-bottom: 5rem;
    color: #2c3e50;
}

/* Titles and Headers */
h1 {
    font-size: 2.8rem;
    font-weight: 800;
    margin-bottom: 1.2rem;
    color: #1b2e40;
    text-align: center;
}
h2, h3 {
    color: #2d3e50;
    margin-top: 2rem;
    font-weight: 600;
}

/* Text and Lists */
p, li {
    font-size: 1.08rem;
    line-height: 1.7;
    color: #34495e;
}
ul {
    padding-left: 1.5rem;
    margin-top: 0.5rem;
}
ul li {
    margin-bottom: 0.6rem;
}

/* Upload Box */
section[data-testid="stFileUploader"] > div {
    background: #f0f8ff;
    border: 2px dashed #4682b4;
    padding: 2rem;
    border-radius: 14px;
    text-align: center;
    box-shadow: 0 6px 14px rgba(0, 0, 0, 0.04);
    transition: all 0.3s ease;
}
section[data-testid="stFileUploader"] > div:hover {
    background: #e6f3ff;
    border-color: #1e6fd1;
}

/* Images */
img {
    border-radius: 10px;
    border: 1px solid #cfd8dc;
    box-shadow: 0 5px 15px rgba(0, 0, 0, 0.05);
}

/* Results Section */
.stAlert {
    border-radius: 10px !important;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.06);
    padding: 1.2rem !important;
}

/* Expander */
.streamlit-expanderHeader {
    font-size: 1.15rem;
    font-weight: 600;
    color: #2c3e50;
}

/* Disclaimer */
.disclaimer {
    font-size: 0.92rem;
    color: #6c757d;
    margin-top: 4rem;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)


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

# --- Main Interface ---
st.title("Parkinson's Disease Detector")
st.subheader("Clock Drawing Test")
st.write("Upload a clock drawing to receive a prediction using our trained image classification model.")

st.markdown("### Drawing Instructions")
st.write("""
To ensure accurate predictions, please follow these instructions:

- Draw an **analog clock** showing the time **7 o'clock**.
- Include **all numbers** from 1 to 12.
- Make sure the **hour and minute hands** are clear.
- Keep the drawing clean and centered.
- If drawn on paper, take a well-lit photo with no shadows or blur.
""")

st.markdown("### Example Clock Drawing")
try:
    img = Image.open("clock_example.png")
    st.image(img, caption="Sample Clock Drawing (7 o'clock)", width=220)
except:
    st.warning("Example image not found. Please place 'clock_example.png' in the same folder.")

# --- Upload Section ---
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
