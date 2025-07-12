import streamlit as st
st.set_page_config(page_title="Plant Disease Detector", page_icon="🌿", layout="centered")
import json

import tensorflow as tf
import numpy as np
from PIL import Image

# --- CACHED: Load models once ---
@st.cache_resource
def load_models():
    model1 = tf.keras.models.load_model("my_modell.keras")                  # CNN (256x256)
    model2 = tf.keras.models.load_model("plant_disease_mobilenet_128.keras") # MobileNet (128x128)
    model3 = tf.keras.models.load_model("plant_disease_resnet152v2.keras")   # ResNet152V2 (128x128)
    return model1, model2, model3

model1, model2, model3 = load_models()


def load_disease_info():
    with open("disease_info.json", "r") as f:
        return json.load(f)

disease_info = load_disease_info()


# --- Class Labels ---
class_names = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
    'Corn_(maize)___healthy', 'Corn_(maize)___Northern_Leaf_Blight', 'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)', 'Grape___healthy', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Potato___Early_blight', 'Potato___healthy', 'Potato___Late_blight',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___healthy',
    'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
    'Tomato___Tomato_mosaic_virus', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus'
]

# --- Image Preprocessing ---
def preprocess_image(image_file):
    image = Image.open(image_file).convert("RGB")
    image = image.resize((256, 256))
    img_array = tf.keras.preprocessing.image.img_to_array(image) / 255.0
    return np.expand_dims(img_array, axis=0)

def preprocess_image128(image_file):
    image = Image.open(image_file).convert("RGB")
    image = image.resize((128, 128))
    img_array = tf.keras.preprocessing.image.img_to_array(image) / 255.0
    return np.expand_dims(img_array, axis=0)

# --- Ensemble Prediction ---
def ensemble_predict(image, weights=[0.2, 0.5, 0.3]):
    img256 = preprocess_image(image)
    img128 = preprocess_image128(image)

    preds1 = model1.predict(img256)[0]
    preds2 = model2.predict(img128)[0]
    preds3 = model3.predict(img128)[0]

    final_preds = (weights[0] * preds1 +
                   weights[1] * preds2 +
                   weights[2] * preds3)

    predicted_class = class_names[np.argmax(final_preds)]
    confidence = np.max(final_preds)
    return predicted_class, confidence

# --- Streamlit UI ---

st.sidebar.title("🔍 Navigation")
app_mode = st.sidebar.radio("Go to", ["🏠 Home", "🔬 Predict"])

# --- HOME PAGE ---
if app_mode == "🏠 Home":
    st.title("🌿 Plant Disease Detection System")
    st.markdown("""
    Welcome to our intelligent **Plant Disease Recognition App**! 🧠🌱

    This tool uses an **ensemble of three deep learning models**:
    - 🧪 **CNN Model (256×256)** — Custom convolutional network
    - ⚡ **MobileNetV2 (128×128)** — Fast and efficient
    - 🏗️ **ResNet152V2 (128×128)** — Deep and powerful

    ### 🔬 How It Works
    - Upload a plant leaf image
    - Each model predicts probabilities of 25 plant disease classes
    - We combine their predictions using a **weighted average**
    - The most confident disease prediction is shown

    ### 🧠 What Makes It Special?
    - ✅ Combines the strengths of 3 models
    - ⚖️ Balanced and accurate results
    - 🕒 Fast, interactive, and reliable

    👉 Switch to the **Prediction tab** to test it out!
    """)
    st.markdown("### 🧱 Model Architectures")
    
    st.subheader("📐 CNN Model Architecture")
    st.image("my_modell.keras.png", use_column_width=True)

    st.subheader("⚡ MobileNetV2 Architecture")
    st.image("plant_disease_mobilenet_128.keras.png", use_column_width=True)

    st.subheader("🏗️ ResNet152V2 Architecture")
    st.image("plant_disease_resnet152v2.keras.png", use_column_width=True)

    
# --- PREDICTION PAGE ---
elif app_mode == "🔬 Predict":
    st.title("🔍 Predict Plant Disease")

    uploaded_image = st.file_uploader("📤 Upload a plant leaf image:", type=["jpg", "jpeg", "png"])

    if uploaded_image:
        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

        if st.button("🎯 Predict"):
            pred_class, confidence = ensemble_predict(uploaded_image)
            st.success(f"🌱 Predicted Disease: **{pred_class}**")
            st.info(f"Confidence: **{confidence:.2%}**")
            # Show cure and prevention if available
            info = disease_info.get(pred_class, None)
            if info:
               st.markdown("### 💊 Suggested Cure")
               st.write(info.get("cure", "Not available."))

               st.markdown("### 🛡️ Prevention Tips")
               st.write(info.get("prevention", "Not available."))
