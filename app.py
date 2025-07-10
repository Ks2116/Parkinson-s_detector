import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import tensorflow.keras as keras
import os

# --- Configuration ---
MODEL_PATH = "keras_model.h5"
LABELS_PATH = "labels.txt"
IMAGE_SIZE = (224, 224)

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

# --- UI ---
st.set_page_config(page_title="Parkinson's Detector (Clock Test)", layout="centered")
st.title("Parkinson's Disease Detector (Clock Test)")
st.write("Upload an image of a clock drawing to get a prediction.")
st.markdown("---")
# Drawing instructions (inserted section)
st.markdown("### 🖊️ Drawing Instructions")
st.write("""
You may draw the clock using **pen on paper** or **digitally (e.g., on an iPad or tablet)**.

Please follow these guidelines to ensure accurate predictions:
- Draw an **analog clock** showing the time **7 o'clock**.
- Include **all numbers** from 1 to 12 on the clock face.
- Ensure the **hour and minute hands** are clearly visible.
- Keep the drawing as neat and clear as possible.
- If drawn on paper, take a well-lit photo with **no shadows or blur** before uploading.
""")

# --- Example Clock Drawing ---
st.markdown("### 🧭 Example Clock Drawing")
st.image("clock_example.png", caption="Example of a correctly drawn clock (7 o'clock)", width=350)


st.markdown("---")
model = load_model()
class_names = load_labels()
if model is None or class_names is None:
    st.stop()

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Clock Drawing", use_column_width=True)
    st.write("Classifying...")

    predicted_class, confidence_score = predict_parkinsons(image, model, class_names)

    st.success(f"**Prediction:** {predicted_class}")
    st.info(f"**Confidence:** {confidence_score:.2f}")

    # Conditional messaging based on the prediction
    if predicted_class.strip() == "1 PD's":
        st.warning("⚠️ This clock drawing may show signs of Parkinson's disease. Please consult a healthcare professional.")
    elif predicted_class.strip() == "2 Alzheimer's":
        st.warning("⚠️ This clock drawing may show signs of Alzheimer's disease. Please consult a healthcare professional.")
    elif predicted_class.strip() == "3 Invalid Input":
        st.error("❌ The uploaded image doesn't appear to be a valid clock drawing. Please try again with a proper clock drawing.")
    else:
        st.success("✅ This clock drawing appears typical.")

st.markdown("---")
with st.expander("⚙️ How This App Works", expanded=False):
    st.markdown("""
<style>
ul.custom-expander li {
    margin-bottom: 0.4rem;
}
</style>

🧠 **Step-by-step Process:**

- 🖼️ **Upload Clock Drawing**  
  Submit a hand-drawn clock image for analysis.

- 🧹 **Preprocessing**  
  Image is cleaned and resized for model input.

- 🤖 **Prediction**  
  Our AI model checks for:
  • Parkinson’s  
  • Alzheimer’s  
  • Invalid Input

- 📊 **Get Results**  
  You'll see a prediction label and confidence score.

> ⚠️ *This tool is experimental. For professional evaluation, consult a healthcare provider.*
""", unsafe_allow_html=True)


st.caption("Disclaimer: This tool is experimental and not a replacement for medical advice.")





