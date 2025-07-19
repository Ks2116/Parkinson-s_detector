import streamlit as st 
from PIL import Image, ImageOps
import numpy as np
import tensorflow.keras as keras
import os

# --- Configuration ---
MODEL_PATH = "keras_model.h5"
LABELS_PATH = "labels.txt"
IMAGE_SIZE = (224, 224)

# --- Page Config & CSS Styling ---
st.set_page_config(page_title="Parkinson's Clock Test", layout="centered")

st.markdown("""
<style>
/* Global Background */
.stApp {
    background-color: #f4f6f9;
}

/* Container Box */
.main-box {
    background-color: #ffffff;
    padding: 3rem 2rem;
    border-radius: 12px;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
    max-width: 800px;
    margin: auto;
    font-family: 'Segoe UI', sans-serif;
    color: #2c3e50;
}

/* Typography */
h1, h2, h3 {
    color: #2c3e50;
}

ul li {
    margin-bottom: 0.5rem;
}

/* Upload Section */
.upload-section {
    background-color: #ffffff;
    padding: 1.5rem;
    border: 2px dashed #2E86C1;
    border-radius: 10px;
    text-align: center;
    margin-top: 2rem;
    margin-bottom: 2rem;
    font-family: 'Segoe UI', sans-serif;
}

.upload-section label {
    font-size: 1.2rem !important;
    font-weight: 600 !important;
    color: #2c3e50 !important;
}

/* Disclaimer */
.disclaimer {
    font-size: 0.85rem;
    color: #555;
}
</style>
""", unsafe_allow_html=True)

# --- Model and Label Loaders ---
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found: {MODEL_PATH}")
        return None
    try:
        model = keras.models.load_model(MODEL_PATH, compile=False)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

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

# --- Image Preprocessing ---
def preprocess_image(image):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = ImageOps.fit(image, IMAGE_SIZE, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
    return np.expand_dims(normalized_image_array, axis=0)

# --- Prediction ---
def predict_parkinsons(image, model, class_names):
    if model is None:
        return "Error: Model not loaded", 0.0
    preprocessed_image = preprocess_image(image)
    prediction = model.predict(preprocessed_image)
    index = np.argmax(prediction)
    return class_names[index], prediction[0][index]

# --- Main Content Box ---
st.markdown('<div class="main-box">', unsafe_allow_html=True)

st.title("Parkinson's Disease Detector")
st.subheader("Clock Drawing Test")
st.write("Upload a clock drawing to receive a prediction using our trained image classification model.")

# Drawing Instructions
st.markdown("### Drawing Instructions")
st.write("""
To ensure accurate predictions, please follow these instructions:

- Draw an **analog clock** showing the time **7 o'clock**.
- Include **all numbers** from 1 to 12.
- Make sure the **hour and minute hands** are clear.
- Keep the drawing clean and centered.
- If drawn on paper, take a well-lit photo with no shadows or blur.
""")

# Example Drawing
st.markdown("### Example Clock Drawing")
img = Image.open("clock_example.png")
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.image(img, caption="Sample Clock Drawing (7 o'clock)", width=250)

st.markdown('</div>', unsafe_allow_html=True)

# --- Upload Section ---
st.markdown('<div class="upload-section">', unsafe_allow_html=True)
uploaded_file = st.file_uploader("Upload Your Clock Drawing Here", type=["jpg", "jpeg", "png"])
st.markdown('</div>', unsafe_allow_html=True)

# --- Load Model and Predict ---
model = load_model()
class_names = load_labels()
if model is None or class_names is None:
    st.stop()

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    st.markdown("### Uploaded Image")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(image, caption="Uploaded Clock Drawing", use_column_width=True)

    st.write("Classifying...")

    predicted_class, confidence_score = predict_parkinsons(image, model, class_names)

    st.success(f"**Prediction:** {predicted_class}")
    st.info(f"**The system is {confidence_score:.0%} sure about this result.**")

    # Conditional messaging
    if predicted_class.strip() == "May have Parkinson's Disease":
        st.warning("This drawing may show signs of Parkinson's disease. Please consult a medical professional.")
    elif predicted_class.strip() == "May have Alzheimer's Disease":
        st.warning("This drawing may show signs of Alzheimer's disease. Consider consulting a doctor.")
    elif predicted_class.strip() == "Invalid Input":
        st.error("The uploaded image is not a valid clock drawing. Please upload a proper one.")
    else:
        st.success("This clock drawing appears typical.")

# --- How It Works Section ---
st.markdown("---")
with st.expander("How This App Works", expanded=False):
    st.markdown("""
**Step-by-step Process:**

- **Upload Clock Drawing**  
  Submit a hand-drawn or digital clock image.

- **Preprocessing**  
  The image is resized and prepared for prediction.

- **Prediction**  
  The system classifies the image as:
    - May have Parkinson's
    - May have Alzheimer's
    - Invalid input
    - Typical

- **Results**  
  You will receive a prediction along with how sure the system is about it.

> Note: This tool is experimental and not a replacement for clinical diagnosis.
""")

# --- Disclaimer ---
st.markdown("<p class='disclaimer'>Disclaimer: This tool is for educational and research purposes only and does not substitute professional medical advice.</p>", unsafe_allow_html=True)







