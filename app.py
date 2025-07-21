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
st.warning("⚠️ This application has not been clinically validated. Results must not be used as a substitute for a professional medical diagnosis.")

# --- Subtle Professional Styling ---
st.markdown("""
<style>
@keyframes slowGradientShift {
  0% { background-position: 0% 50%; }
  50% { background-position: 100% 50%; }
  100% { background-position: 0% 50%; }
}

html, body, .stApp {
    background: linear-gradient(200deg, #eef6fa, #f8fbfe, #e9f3f7, #f6f9fc);
    background-size: 300% 300%;
    background-attachment: fixed;
    animation: slowGradientShift 3s ease infinite;
    font-family: 'Segoe UI', sans-serif;
    color: #355c60;
    padding: 2rem 10vw 5rem 10vw;
    box-sizing: border-box;
}

html[data-theme="dark"], body[data-theme="dark"], .stApp[data-theme="dark"] {
    background: #111111 !important;
    color: #f0f0f0 !important;
}

.card {
    background-color: white;
    padding: 2rem;
    border-radius: 12px;
    box-shadow: 0 8px 20px rgba(0, 0, 0, 0.05);
    margin-bottom: 2rem;
}

.banner {
    background: linear-gradient(135deg, #91c8ea, #d4eefc);
    color: #fff;
    text-align: center;
    padding: 2.5rem 1rem;
    border-radius: 16px;
    box-shadow: 0 12px 24px rgba(0,0,0,0.08);
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

@keyframes waveAnimation {
  0% {
    background-position: 0 0;
  }
  100% {
    background-position: 1000px 0;
  }
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
  opacity: 0.06;
  z-index: -1;
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

st.markdown("### What is the Clock Drawing Test?")

st.markdown("""
The **Clock Drawing Test** is a widely used screening tool that helps check how well a person's brain is working. It usually involves drawing an analog clock showing a specific time — such as **7 o'clock** — with all the numbers and the clock hands in the right places.

This task may seem simple, but it actually involves several mental skills, such as:
- Understanding and following instructions  
- Visual-spatial skills (how things fit together visually)  
- Motor coordination (hand movement and control)  
- Memory and reasoning

These types of brain functions are called **cognitive abilities**, which basically means how we think, learn, remember, and solve problems.

Doctors often use this test to spot early warning signs of brain-related conditions like **Parkinson’s disease** and **Alzheimer’s disease**.

While the Clock Drawing Test alone can't diagnose these conditions, it can give helpful clues when used together with other medical checks.

---

 **Want to learn more?**  
You can read more about the Clock Drawing Test from a trusted health source [here](https://pmc.ncbi.nlm.nih.gov/articles/PMC5619222/)
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

# --- Upload Drawing ---
st.markdown("""
<div style='background-color: #ffffff; padding: 1.5rem; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.04); margin-top: 2rem;'>
<h3 style='margin-top: 0;'>Upload Your Clock Drawing</h3>
<p style='margin-bottom: 1rem;'>Please upload a clear photo of a hand-drawn analog clock showing <strong>7 o'clock</strong>. Ensure the image is well-lit and free of shadows.</p>
</div>
""", unsafe_allow_html=True)

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

        with st.spinner("Running AI analysis on your clock drawing..."):
            time.sleep(2.5)
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

if prediction == "May have Parkinson's Disease":
    st.markdown("---")
    with st.container():
        st.markdown(
            """
            <div style='padding: 1.5rem; border-radius: 15px; background-color: #f3f4f6; box-shadow: 0 2px 6px rgba(0, 0, 0, 0.05);'>
                <h3 style='color: #374151;'>🧭 Your Result May Suggest Signs Related to Parkinson’s</h3>
                <p style='font-size: 1.05rem; color: #4B5563;'>
                    This result does <strong>not</strong> mean you have Parkinson’s Disease. This tool simply identified patterns that <em>might</em> resemble those seen in individuals with Parkinson’s.
                </p>

                <h4 style='color: #111827;'>If you're feeling unsure, here's what you can do:</h4>
                <ul style='color: #374151; line-height: 1.6;'>
                    <li>You don’t need to panic — this is just a potential indication, not a diagnosis.</li>
                    <li>Consider checking in with a <strong>neurologist</strong> or general practitioner if you have concerns.</li>
                    <li>They may offer more detailed assessments such as movement tests or imaging, if appropriate.</li>
                </ul>

                <h4 style='color: #111827;'>Why a follow-up could be helpful:</h4>
                <ul style='color: #374151; line-height: 1.6;'>
                    <li>It helps clarify things and gives you peace of mind.</li>
                    <li>Early check-ups allow better tracking and management, even if everything is fine.</li>
                </ul>

                <p style='margin-top: 1rem; font-size: 0.95rem; color: #6B7280;'>
                    This app is just a starting point — exploring your concerns with a professional is always a safe and proactive choice.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

elif prediction == "May have Alzheimer's Disease":
    st.markdown("---")
    with st.container():
        st.markdown(
            """
            <div style='padding: 1.5rem; border-radius: 15px; background-color: #f3f4f6; box-shadow: 0 2px 6px rgba(0, 0, 0, 0.05);'>
                <h3 style='color: #374151;'>🧭 Your Result May Suggest Patterns Linked to Alzheimer’s</h3>
                <p style='font-size: 1.05rem; color: #4B5563;'>
                    This result does <strong>not</strong> confirm Alzheimer’s. It simply suggests there may be some visual features similar to those observed in Alzheimer’s cases.
                </p>

                <h4 style='color: #111827;'>Next steps you might want to consider:</h4>
                <ul style='color: #374151; line-height: 1.6;'>
                    <li>No need to worry — this is just an early indicator for your awareness.</li>
                    <li>If you're experiencing memory issues or are concerned, consider speaking to a <strong>doctor</strong> or <strong>memory care specialist</strong>.</li>
                    <li>They may suggest cognitive testing or scans to provide more clarity.</li>
                </ul>

                <h4 style='color: #111827;'>Why a gentle follow-up could help:</h4>
                <ul style='color: #374151; line-height: 1.6;'>
                    <li>Understanding more about your cognitive health is always useful.</li>
                    <li>Even a brief consultation can offer guidance and reassurance.</li>
                </ul>

                <p style='margin-top: 1rem; font-size: 0.95rem; color: #6B7280;'>
                    You're doing the right thing by taking an interest in your brain health. A calm conversation with a professional can help you feel more at ease.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )


# --- How It Works ---
st.markdown("""<hr style="border: 1px solid #dcdcdc; margin-top: 2rem; margin-bottom: 1rem;">""", unsafe_allow_html=True)

with st.expander("⚙️ **How This App Works**", expanded=False):
    st.markdown("""
**Step-by-step Process:**

- **Upload Clock Drawing**  
  Submit a hand-drawn or digital clock image.

- **Preprocessing**   
  The image is adjusted to the input format required by the AI model and normalized.

- **Prediction**  
  Our model classifies the image as:
    - May have Parkinson's Disease
    - May have Alzheimer's Disease
    - Typical
    - Invalid Input
    
&nbsp;
- **Results**  
  You’ll receive a prediction and a confidence score.

> Note: This tool is experimental and not a replacement for clinical diagnosis.
""", unsafe_allow_html=True)

# --- Learn More Section ---
st.markdown("""<hr style="border: 1px solid #dcdcdc; margin-top: 2rem; margin-bottom: 1rem;">""", unsafe_allow_html=True)
st.markdown("### 🔎 Learn More About Parkinson's Disease")
st.write("To understand more about Parkinson’s Disease, its symptoms, causes, and available treatments, read the World Health Organization's fact sheet:")
st.markdown("[World Health Organization – Parkinson Disease (WHO)](https://www.who.int/news-room/fact-sheets/detail/parkinson-disease)")

st.markdown("### 🔎 Learn More About Alzheimer’s Disease")
st.write("To learn more about Alzheimer’s Disease, including its signs, stages, causes, and care options, visit the National Institute on Aging’s official resource:")
st.markdown("[National Institute on Aging – What Is Alzheimer’s Disease?](https://www.nia.nih.gov/health/alzheimers-and-dementia/what-alzheimers-disease)")

# --- Disclaimer ---
st.markdown("<p class='disclaimer'>Disclaimer: This tool is for educational and research purposes only and does not substitute professional medical advice.</p>", unsafe_allow_html=True)



