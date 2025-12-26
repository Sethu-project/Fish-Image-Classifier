import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# -------------------------------
# MODEL LOADING
# -------------------------------
MODEL_PATH = r"D:\GUVI DS 2025\Fish Image Classification\mobilenet_best.h5"

if not os.path.exists(MODEL_PATH):
    st.error("Model file not found.")
    st.stop()

model = tf.keras.models.load_model(MODEL_PATH)

# -------------------------------
# CONSTANTS
# -------------------------------
IMG_SIZE = (224, 224)
CONFIDENCE_THRESHOLD = 70.0
ENTROPY_THRESHOLD = 1.5

CLASS_NAMES = [
    'animal fish',
    'animal fish bass',
    'fish sea_food black_sea_sprat',
    'fish sea_food gilt_head_bream',
    'fish sea_food hourse_mackerel',
    'fish sea_food red_mullet',
    'fish sea_food red_sea_bream',
    'fish sea_food sea_bass',
    'fish sea_food shrimp',
    'fish sea_food striped_red_mullet',
    'fish sea_food trout'
]

# -------------------------------
# STREAMLIT UI
# -------------------------------
st.set_page_config(page_title="Fish Classifier", page_icon="🐟", layout="centered")

st.title("🐟 Fish Species Classification")
st.image(
    "https://upload.wikimedia.org/wikipedia/commons/1/19/Clown_fish_in_the_Andaman_Coral_Reef.jpg",
    #caption="Fish Classifier - Upload your fish image below",
    use_container_width=True
)
st.write("Upload an image to classify the fish species")

uploaded_file = st.file_uploader(
    "Upload Fish Image",
    type=["jpg", "png", "jpeg"]
)

# -------------------------------
# MAIN LOGIC (ONLY RUNS AFTER UPLOAD)
# -------------------------------
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # -------------------------------
    # PREPROCESSING
    # -------------------------------
    img = image.resize(IMG_SIZE)
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # -------------------------------
    # PREDICTION
    # -------------------------------
    preds = model.predict(img_array, verbose=0)
    class_idx = np.argmax(preds)
    confidence = float(np.max(preds) * 100)

    # -------------------------------
    # REJECTION LOGIC (FINAL)
    # -------------------------------
    entropy = -np.sum(preds[0] * np.log(preds[0] + 1e-10))

    if confidence < CONFIDENCE_THRESHOLD:
        st.error("❌ Image Rejected (Low Confidence)")
        st.info(f"Confidence: {confidence:.2f}%")
        st.stop()

    if entropy > ENTROPY_THRESHOLD:
        st.error("❌ Image Rejected (High Uncertainty)")
        st.info(f"Entropy Score: {entropy:.2f}")
        st.stop()

    # -------------------------------
    # ACCEPTED PREDICTION
    # -------------------------------
    st.success(f"✅ Prediction: {CLASS_NAMES[class_idx]}")
    st.info(f"Confidence: {confidence:.2f}%")
    st.progress(confidence / 100)