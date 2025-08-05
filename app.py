import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import tensorflow.keras as keras
import os
import time
from datetime import datetime
import pytz
import io
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader
from textwrap import wrap


singapore_time = datetime.now(pytz.timezone('Asia/Singapore')).strftime("%Y-%m-%d %H:%M")

# --- Configuration ---
MODEL_PATH = "keras_model.h5"
LABELS_PATH = "labels.txt"
IMAGE_SIZE = (224, 224)

# --- Page Settings ---
st.set_page_config(page_title="Parkinson's Clock Test", layout="centered")

# --- Validation Disclaimer at the Top ---
st.warning(
    "⚠️ This application has not been clinically validated. Results must not be used as a substitute for a professional medical diagnosis."
)

# --- Subtle Professional Styling ---
st.markdown(
    """
<style>
@keyframes slowGradientShift {
  0% { background-position: 0% 50%; }
  50% { background-position: 100% 50%; }
  100% { background-position: 0% 50%; }
}

html, body, .stApp {
    background: linear-gradient(200deg, #eef6fa, #40E0D0, #F4F5F0, #f8fbfe, #e9f3f7, #f6f9fc, #FFFFFF);
    background-size: 300% 300%;
    background-attachment: fixed;
    animation: slowGradientShift 20s ease infinite;
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
    color: #ffffff;
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

/* Wave container */
.wave-container {
  position: fixed;
  bottom: 0;
  left: 0;
  width: 300%;
  height: 160px;
  z-index: -1;
  overflow: hidden;
}

/* Animate the SVG path */
.wave-container svg {
  width: 200%;
  height: 100%;
  animation: waveMotion 10s ease-in-out infinite;
}

@keyframes waveMotion {
  0%   { transform: translateX(0); }
  50%  { transform: translateX(-30%); }
  100% { transform: translateX(0); }
}
</style>

<!-- Animated turquoise wave -->
<div class="wave-container">
<svg viewBox="0 0 1200 120" preserveAspectRatio="none">
  <path d="M0,40 C300,80 900,0 1200,40 L1200,120 L0,120 Z" 
        fill="#40E0D0" opacity="0.45">
    <animateTransform attributeName="transform"
                      attributeType="XML"
                      type="translate"
                      from="0,0" to="-200,0"
                      dur="6s"
                      repeatCount="indefinite" />
  </path>
</svg>
</div>
""",
    unsafe_allow_html=True,
)


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
st.markdown(
    '<div class="banner"><h1>Parkinson\'s Disease Detector</h1><p>AI-powered tool for analyzing clock drawings</p></div>',
    unsafe_allow_html=True,
)

st.markdown("### What is the Clock Drawing Test?")

st.markdown(
    """
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
"""
)

# --- Drawing Instructions ---
st.markdown(
    """<hr style="border: 1px solid #dcdcdc; margin-top: 2rem; margin-bottom: 1rem;">""",
    unsafe_allow_html=True,
)
st.markdown(
    """
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
""",
    unsafe_allow_html=True,
)

# --- Example Clock ---
st.markdown("### Example Clock Drawing")
try:
    img = Image.open("clock_example.png")
    st.image(img, caption="Sample Clock Drawing (7 o'clock)", width=220)
except Exception:
    st.warning("Example image not found. Please place 'clock_example.png' in the same folder.")

# --- Upload Drawing ---
st.markdown(
    """
<div style='background-color: #ffffff; padding: 1.5rem; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.04); margin-top: 2rem;'>
<h3 style='margin-top: 0;'>Upload Your Clock Drawing</h3>
<p style='margin-bottom: 1rem;'>Please upload a clear photo of a hand-drawn analog clock showing <strong>7 o'clock</strong>. Ensure the image is well-lit and free of shadows.</p>
</div>
""",
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

# --- Load Model & Labels ---
model = load_model()
class_names = load_labels()
if model is None or class_names is None:
    st.stop()

# --- Handle Uploaded File and Prediction in try-except block ---
if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)

        st.markdown("### Uploaded Image")
        st.image(image, caption="Uploaded Clock Drawing", width=300)

        with st.spinner("Running AI analysis on your clock drawing..."):
            time.sleep(2.5)
            predicted_class, confidence_score = predict_parkinsons(image, model, class_names)

            st.markdown(
                "<p style='color: #444; font-size: 1rem; font-style: italic; margin-top: 1rem;'> Analysis complete — please scroll down to view the full results.</p>",
                unsafe_allow_html=True,
            )

        st.success(f"**Prediction:** {predicted_class}")
        st.info(f"**The system is {confidence_score:.0%} confident in this result.**")

    
        # Prepare summary text
        summary_text = f"""
Parkinson's Clock Test Result
Date: {singapore_time}

Prediction: {predicted_class}
Confidence: {confidence_score:.0%}

Note:
This result does NOT confirm a medical diagnosis.
It is based on patterns seen in the clock drawing and is meant for awareness only.

"""

        guidance_block = ""

        # Guidance and display
        if predicted_class.strip() == "May have Parkinson's Disease":
            guidance = """
Your Result May Suggest Signs Related to Parkinson's Disease

This result does NOT necessarily mean that you have Parkinson's Disease.
It simply indicates patterns that may resemble those found in some Parkinson's cases.

Here's what you can do next:
- Stay calm — this is only a screening tool, not a diagnosis.
- Consider consulting a neurologist or primary care doctor.
- Further testing like motor assessments or brain imaging may be recommended.

Why a check-in could help:
- It can clarify things and reduce unnecessary worry.
- Early professional advice is valuable, even if everything turns out fine.

This tool is a first step — following up with a doctor can bring peace of mind.
"""
            guidance_block = guidance
            st.warning(
                "This drawing may show signs of Parkinson's disease. Please consult a medical professional."
            )
            st.markdown(
                f"""<div style='padding: 1.5rem; border-radius: 15px; background-color: #f3f4f6; box-shadow: 0 2px 6px rgba(0, 0, 0, 0.05);'>
                <h3 style='color: #374151;'>🧭 Your Result May Suggest Signs Related to Parkinson’s Disease</h3>
                <p style='font-size: 1.05rem; color: #4B5563;'>
                    This result does <strong>Not Necessarily</strong> mean that you have Parkinson’s Disease. It simply indicates patterns that <em>may</em> resemble those found in some Parkinson’s cases.
                </p>
                <h4 style='color: #111827;'>Here’s what you can do next:</h4>
                <ul style='color: #374151; line-height: 1.6;'>
                    <li>Stay calm — this is only a screening tool, not a diagnosis.</li>
                    <li>Consider consulting a <strong>neurologist</strong> or primary care doctor.</li>
                    <li>Further testing like motor assessments or brain imaging may be recommended.</li>
                </ul>
                <h4 style='color: #111827;'>Why a check-in could help:</h4>
                <ul style='color: #374151; line-height: 1.6;'>
                    <li>It can clarify things and reduce unnecessary worry.</li>
                    <li>Early professional advice is valuable, even if everything turns out fine.</li>
                </ul>
                <p style='margin-top: 1rem; font-size: 0.95rem; color: #6B7280;'>
                    This tool is a first step — following up with a doctor can bring peace of mind.
                </p>
            </div>""",
                unsafe_allow_html=True,
            )

        elif predicted_class.strip() == "May have Alzheimer's Disease":
            guidance = """
🧭 Your Result May Suggest Patterns Linked to Alzheimer's Disease

This result does NOT confirm Alzheimer's.
It only points to some signs that may resemble those found in Alzheimer's-related drawings.

Here's what you can do next:
- Stay calm — this is just an early suggestion, not a diagnosis.
- Consider consulting with a doctor or memory specialist.
- They may recommend cognitive screening or additional follow-ups.

Why early awareness matters:
- It supports peace of mind and informed decisions.
- Even brief medical input can be empowering and helpful.

You're being proactive about your cognitive health — that's a great first step.
"""
            guidance_block = guidance
            st.warning("This drawing may show signs of Alzheimer's disease. Consider consulting a doctor.")
            st.markdown(
                f"<div style='padding: 1.5rem; border-radius: 15px; background-color: #f3f4f6; box-shadow: 0 2px 6px rgba(0, 0, 0, 0.05);'>{guidance.replace(chr(10), '<br>')}</div>",
                unsafe_allow_html=True,
            )

        elif predicted_class.strip() == "Invalid Input":
            guidance = """
⚠️ The uploaded image is not a valid clock drawing.

Please ensure that:
- The clock includes all numbers from 1 to 12.
- The hands are showing exactly 7 o'clock.
- The drawing is clear, well-lit, and not blurry or distorted.

Try uploading a new image that follows the drawing instructions carefully.
"""
            guidance_block = guidance
            st.error("The uploaded image is not a valid clock drawing. Please upload a clear and complete one.")

        else:
            guidance = """
✅ This Clock Drawing Appears Typical

No unusual signs were detected in this drawing.

Still, if you ever feel unsure or notice changes in thinking, memory, or coordination, it's perfectly okay to speak with a healthcare provider.

Regular checkups and awareness of cognitive health are always encouraged.
"""
            guidance_block = guidance
            st.success("This clock drawing appears typical. No unusual signs were detected.")

        # Append guidance to summary
        summary_text += guidance_block

        # Bonus tip
        bonus_tip = """
---

Regardless of the result, maintaining a healthy lifestyle including regular exercise, balanced diet, mental stimulation, and social interaction supports brain health.

If you have concerns or questions, always reach out to healthcare professionals.
"""

        summary_text += bonus_tip
        # --- Create PDF ---
        pdf_buffer = io.BytesIO()
        c = canvas.Canvas(pdf_buffer, pagesize=letter)
        width, height = letter

        margin = 50
        line_height = 14
        cursor_y = height - 50

        # Title
        c.setFont("Helvetica-Bold", 16)
        c.drawString(margin, cursor_y, "Parkinson's Clock Drawing Test Result")
        cursor_y -= 2 * line_height

        # Date & Prediction
        c.setFont("Helvetica", 10)
        c.drawString(margin, cursor_y, f"Date: {singapore_time}")
        cursor_y -= line_height
        c.drawString(margin, cursor_y, f"Prediction: {predicted_class}")
        cursor_y -= line_height
        c.drawString(margin, cursor_y, f"Confidence: {confidence_score:.0%}")
        cursor_y -= 2 * line_height

        # Guidance block
        text_obj = c.beginText(margin, cursor_y)
        text_obj.setFont("Helvetica", 10)
        text_obj.setLeading(line_height)
        for line in guidance_block.strip().split("\n"):
            if cursor_y <= 100:
                c.drawText(text_obj)
                c.showPage()
                text_obj = c.beginText(margin, height - 50)
                text_obj.setFont("Helvetica", 10)
                text_obj.setLeading(line_height)
                cursor_y = height - 50
            text_obj.textLine(line)
            cursor_y -= line_height
        c.drawText(text_obj)
        cursor_y -= 2 * line_height

        # Insert uploaded image
        if cursor_y < 300:
            c.showPage()
            cursor_y = height - 50

        try:
            img_buffer = io.BytesIO()
            image.convert("RGB").save(img_buffer, format="PNG")
            img_buffer.seek(0)
            c.drawString(margin, cursor_y, "Uploaded Clock Drawing:")
            cursor_y -= line_height
            img_width = 250
            img_height = 250
            c.drawImage(ImageReader(img_buffer), margin, cursor_y - img_height, width=img_width, height=img_height)
            cursor_y -= (img_height + 2 * line_height)
        except Exception as e:
            c.drawString(margin, cursor_y, "Error displaying uploaded image.")
            cursor_y -= 2 * line_height

        # Write bonus tip
        bonus_lines = bonus_tip.strip().split("\n")
        text2 = c.beginText(margin, cursor_y)
        text2.setFont("Helvetica", 10)
        text2.setLeading(line_height)

        for line in bonus_lines:
             wrapped_lines = wrap(line, width=90)  # Wrap lines to fit page
        for wline in wrapped_lines:
            if cursor_y <= 100:
                c.drawText(text2)
                c.showPage()
                text2 = c.beginText(margin, height - 50)
                text2.setFont("Helvetica", 10)
                text2.setLeading(line_height)
                cursor_y = height - 50
            text2.textLine(line)
            cursor_y -= line_height

        c.drawText(text2)
        c.showPage()
        c.save()
        pdf_buffer.seek(0)

        # PDF download button
        st.download_button(
            label="📄 Download Result as PDF",
            data=pdf_buffer.getvalue(),
            file_name="clock_test_result.pdf",
            mime="application/pdf",
        )

      
    except Exception as e:
        st.error(f"⚠️ Failed to open or analyze image: {e}")
        st.stop()


# --- Feedback Section ---
st.markdown(
    """
<div style='padding: 1.5rem; border-radius: 15px; background-color: #f9fafb; border: 1px solid #e5e7eb; margin-top: 2rem;'>
    <h4 style='color: #111827;'>📋 We’d Appreciate Your Feedback</h4>
    <p style='color: #4B5563; font-size: 1.05rem;'>
        Your input helps us improve the quality and clarity of this tool.<br>
        If you’ve just used the app, please take a moment to complete our short feedback form.
    </p>
    <a href='https://docs.google.com/forms/d/1SbZXHxdveEXCWB0oYYqg_Vx_x2CtuO7o8zQrDc2Lf5w' target='_blank' style='text-decoration: none; font-weight: 500; color: #2563eb;'>👉 Click here to provide feedback</a>
    <p style='color: #6B7280; font-size: 0.95rem; margin-top: 1rem;'>We appreciate your time and feedback as we continue to improve this screening tool.</p>
</div>
""",
    unsafe_allow_html=True,
)

# --- Go Back to Webpage Button ---
if st.button("🔙 Go back to Webpage"):
    js = "window.open('https://detectparkinson.wixsite.com/mysite')"  # opens in new tab
    st.markdown(f"<script>{js}</script>", unsafe_allow_html=True)


# --- How It Works ---
st.markdown(
    """<hr style="border: 1px solid #dcdcdc; margin-top: 2rem; margin-bottom: 1rem;">""",
    unsafe_allow_html=True,
)

with st.expander("⚙️ **How This App Works**", expanded=False):
    st.markdown(
        """
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
""",
        unsafe_allow_html=True,
    )

# --- Learn More Section ---
st.markdown(
    """<hr style="border: 1px solid #dcdcdc; margin-top: 2rem; margin-bottom: 1rem;">""",
    unsafe_allow_html=True,
)
st.markdown("### 🔎 Learn More About Parkinson's Disease")
st.write(
    "To understand more about Parkinson’s Disease, its symptoms, causes, and available treatments, read the World Health Organization's fact sheet:"
)
st.markdown("[World Health Organization – Parkinson Disease (WHO)](https://www.who.int/news-room/fact-sheets/detail/parkinson-disease)")

st.markdown("### 🔎 Learn More About Alzheimer’s Disease")
st.write(
    "To learn more about Alzheimer’s Disease, including its signs, stages, causes, and care options, visit the National Institute on Aging’s official resource:"
)
st.markdown(
    "[National Institute on Aging – What Is Alzheimer’s Disease?](https://www.nia.nih.gov/health/alzheimers-and-dementia/what-alzheimers-disease)"
)

# --- Go Back to Webpage Button ---
if st.button("🔙 Go back to Webpage"):
    js = "window.open('https://detectparkinson.wixsite.com/mysite')"  # opens in new tab
    st.markdown(f"<script>{js}</script>", unsafe_allow_html=True)


# --- Disclaimer ---
st.markdown(
    "<p class='disclaimer'>Disclaimer: This tool is for educational and research purposes only and does not substitute professional medical advice.</p>",
    unsafe_allow_html=True,
)




