# Fish-Image-Classifier
“Streamlit app for fish species classification using the Best Model.”

# 🐟 Fish Image Classification (Best Model Deployment)

This project is a **Streamlit web application** that classifies fish species using the **best trained Mobilenet model**.  
Out of five trained models, the one with the highest accuracy and reliability (`mobilenet_best.h5`) was selected and deployed for testing.  

The app allows users to upload an image, processes it, and predicts the fish species.  
It also includes **rejection logic** to filter out uncertain or low-confidence predictions.

---

## 🚀 Features
- Upload an image and classify the fish species.
- Uses the **best Mobilenet model** (`mobilenet_best.h5`).
- Rejects images if:
  - Confidence < 70% (low confidence).
  - Entropy > 1.5 (high uncertainty).
- Simple, interactive UI built with **Streamlit**.
- Displays prediction, confidence score, and progress bar if accepted.

---

## 📂 Project Structure
fish-image-classification/ ├── app.py    
# Streamlit app code ├── mobilenet_best.h5  
# Best trained Mobilenet model ├── requirements.txt 
# Dependencies └── README.md 
# Project documentation

---

## ⚙️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/username/fish-image-classification.git
   cd fish-image-classification

   pip install -r requirements.txt

   streamlit run app.py

Requirements:
streamlit
tensorflow
numpy
pillow
