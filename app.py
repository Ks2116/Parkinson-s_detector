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
@keyframes elegantWave {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
.stApp {
    background: linear-gradient(-45deg, #d1e4f6, #b3cde0, #dee2e6, #a3bce2);
    background-size: 500% 500%;
    animation: elegantWave 15s ease infinite;
    font-family: 'Segoe UI', sans-serif;
    color: #2c3e50;
    padding-bottom: 4rem;
}
.container {
    background-color: rgba(255, 255, 255, 0.96);
    padding: 2.5rem 2rem;
    border-radius: 16px;
    box-shadow: 0 12px 30px rgba(0,0,0,0.06);
    max-width: 900px;
    margin: 0 auto;
}
.banner {
    background: linear-gradient(135deg, #4fc3f7, #7e57c2);
    color: white;
    text-align: center;
    padding: 2rem;
    border-radius: 20px;
    box-shadow: 0 10px 24px rgba(0,0,0,0.1);
    margin-bottom: 2rem;
}
h1 {
    font-size: 2.4rem;
    font-weight: 600;
    margin: 0;
}
h2 {
    font-size: 1.6rem;
    font-weight: 600;
    color: #37474f;
}
hr {
    border: none;
    border-top: 1px solid #ccc;
    margin: 2rem 0;
}
img {
    border-radius: 10px;
    box-shadow: 0 4px 16px rgba(0,0,0,0.05);
}
[data-testid="stFileUploader"] {
    border: 2px dashed #90caf9;
    border-radius: 12px;
    padding: 1.5rem;
    background-color: rgba(255,255,255,0.9);
    margin-bottom: 1rem;
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
st.markdown('<div class="banner"><h1>Parkinson\'s Disease Detector</h1><p>AI-powered analysis of clock drawings</p></div>', unsafe_allow_html=True)

# --- Main Container ---
with st.container():
    st.markdown('<div class="container">', unsafe_allow_html=True)

    st.subheader("Step 1: Upload Your Clock Drawing")
    st.write("Upload a drawing of a clock showing the time 7:00 for analysis.")

    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

    st.markdown("<hr>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Drawing Instructions")
        st.write("""
- Draw an analog clock showing **7:00**.
- Include all numbers (1–12).
- Clearly show hour and minute hands.
- Keep the image clear, centered, and well-lit.
        """)

    with col2:
        st.subheader("Example Clock")
        try:
            example_img = Image.open("clock_example.png")
            st.image(example_img, caption="Sample Clock Drawing", use_column_width=True)
        except:
            st.warning("Example image not found (clock_example.png).")

    st.markdown("<hr>", unsafe_allow_html=True)

    model = load_model()
    class_names = load_labels()
    if model is None or class_names is None:
        st.stop()

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            st.subheader("Step 2: Image Preview")
            st.image(image, caption="Uploaded Clock Drawing", width=300)

            with st.spinner("Analyzing..."):
                time.sleep(2)
                predicted_class, confidence_score = predict_parkinsons(image, model, class_names)

            st.markdown("<hr>", unsafe_allow_html=True)
            st.subheader("Prediction Result")
            st.success(f"**Prediction:** {predicted_class}")
            st.info(f"**Confidence Score:** {confidence_score:.0%}")

            if predicted_class.strip() == "May have Parkinson's Disease":
                st.warning("This result suggests possible signs of Parkinson's disease. Please consult a medical professional.")
            elif predicted_class.strip() == "May have Alzheimer's Disease":
                st.warning("This result suggests possible signs of Alzheimer's disease. Please consult a medical professional.")
            elif predicted_class.strip() == "Invalid Input":
                st.error("Invalid input. Please upload a proper clock drawing.")
            else:
                st.success("The drawing appears typical.")

        except Exception as e:
            st.error(f"Failed to process image: {e}")

    st.markdown("<hr>", unsafe_allow_html=True)

    with st.expander("How This Works"):
        st.markdown("""
1. **Upload Drawing**: Submit a hand-drawn or scanned clock image.
2. **Preprocessing**: The image is resized and normalized.
3. **Model Prediction**: A trained AI model analyzes the image.
4. **Result**: A prediction and confidence score are shown.

*Note: This tool is experimental and not a substitute for medical diagnosis.*
        """)

    st.markdown("</div>", unsafe_allow_html=True)

# --- Disclaimer ---
st.markdown("<p class='disclaimer'>Disclaimer: This tool is for educational and research purposes only. It does not substitute professional medical advice.</p>", unsafe_allow_html=True)
