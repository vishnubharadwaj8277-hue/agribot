import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import requests
from dotenv import load_dotenv
import os
from groq import Groq
import io
import cv2  # <--- This import is needed for the validation

# --- Import all required functions ---
from project_bot import render_project_bot
from utils import (
    apply_custom_css, t, get_kannada_audio_bytes,
    check_login, render_sidebar
)

# -----------------------------
# --- Main App Logic ---
# -----------------------------
apply_custom_css() 
check_login()      
render_sidebar()   
# -----------------------------

lang = st.session_state.lang

load_dotenv()
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

@st.cache_resource
def load_model():
    model_path = "FinalTest_inceptionv3.h5" 
    if not os.path.exists(model_path):
        st.error(f"Model file not found at {model_path}.")
        return None
    custom_objects = {"mse": tf.keras.losses.MeanSquaredError()}
    return tf.keras.models.load_model(model_path, custom_objects=custom_objects)

model = load_model()
class_labels = {0: "Brown Spot", 1: "Healthy Plant", 2: "Leaf Blast", 3: "Sheath Blight"}

# ========================================
# CROP VALIDATION FUNCTION (This is what you asked for)
# ========================================

def is_crop_image(img):
    """
    Simple check: Is this a crop/plant image or something else?
    Returns: True if crop/plant, False if not
    """
    img_array = np.array(img)
    img_hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
    
    # Check 1: Does it have plant colors (green, yellow, brown)?
    lower_green = np.array([20, 15, 15])
    upper_green = np.array([95, 255, 255])
    green_mask = cv2.inRange(img_hsv, lower_green, upper_green)
    
    lower_yellow = np.array([8, 15, 15])
    upper_yellow = np.array([40, 255, 255])
    yellow_mask = cv2.inRange(img_hsv, lower_yellow, upper_yellow)
    
    plant_mask = cv2.bitwise_or(green_mask, yellow_mask)
    plant_ratio = np.sum(plant_mask > 0) / plant_mask.size
    
    # At least 8% of the image should be plant-colored
    if plant_ratio < 0.08:
        return False
    
    # Check 2: Is it too metallic/gray? (bikes, cars, buildings)
    lower_gray = np.array([0, 0, 30])
    upper_gray = np.array([180, 60, 230])
    gray_mask = cv2.inRange(img_hsv, lower_gray, upper_gray)
    gray_ratio = np.sum(gray_mask > 0) / gray_mask.size
    
    if gray_ratio > 0.40:
        return False
    
    # Check 3: Is it a person? (skin tone detection)
    lower_skin = np.array([0, 15, 60])
    upper_skin = np.array([25, 180, 255])
    skin_mask = cv2.inRange(img_hsv, lower_skin, upper_skin)
    skin_ratio = np.sum(skin_mask > 0) / skin_mask.size
    
    if skin_ratio > 0.12:
        return False
    
    # Check 4: Is it too bright? (paper, walls, sky)
    brightness = img_hsv[:,:,2]
    very_bright_ratio = np.sum(brightness > 235) / brightness.size
    if very_bright_ratio > 0.45:
        return False
    
    # Check 5: Is it too dark? (night photos, black objects)
    very_dark_ratio = np.sum(brightness < 25) / brightness.size
    if very_dark_ratio > 0.55:
        return False
    
    return True

# ========================================
# END OF VALIDATION FUNCTION
# ========================================

@st.cache_data(ttl=300)
def get_weather(city="Bangalore"):
    if not OPENWEATHER_API_KEY: 
        return (25, 60, 0, "http://openweathermap.org/img/wn/01d@2x.png", "Clear")
    try:
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={OPENWEATHER_API_KEY}&units=metric"
        response = requests.get(url, timeout=10)
        data = response.json()
        if data.get("cod") == 200:
            temp = round(data["main"]["temp"])
            icon = data["weather"][0]["icon"]
            desc = data["weather"][0]["description"].title()
            icon_url = f"http://openweathermap.org/img/wn/{icon}@2x.png"
            return temp, data["main"]["humidity"], data.get("rain", {}).get("1h", 0), icon_url, desc
    except: 
        pass
    return 25, 60, 0, "http://openweathermap.org/img/wn/01d@2x.png", "Clear"

def get_treatment_from_llm(disease: str, lang: str):
    if not client: 
        return t("LLM not available.", lang), None
    prompt = f"4 short, practical cure & prevention steps for paddy {disease}. Bullets only."
    if lang == "Kannada": 
        prompt += " Answer in Kannada. Use • for bullets."
    try:
        chat = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}], 
            model="llama-3.3-70b-versatile", # Corrected model name
            temperature=0.3, 
            max_tokens=250
        )
        response = chat.choices[0].message.content.strip()
        lines = []
        for line in response.split('\n'):
            line = line.strip()
            if line.startswith(('•', '-', '*')):
                line = line.lstrip('•-* ').strip()
                if line: 
                    lines.append(f"• {line}")
        clean_text = "<br>".join(lines[:4])
        audio_text = " ".join([l.replace('•', '').strip() for l in lines[:4]])
        return clean_text, audio_text
    except Exception as e: 
        return t(f"Error: {e}", lang), None

def preprocess_image(img):
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_image(model, img):
    try:
        img_array = preprocess_image(img)
        # The model is still expected to return two outputs based on load_model
        # predictions[0] is for the disease class, predictions[1] is for severity (which we will discard)
        predictions = model.predict(img_array)
        class_probs = predictions[0]
        disease_idx = np.argmax(class_probs)
        disease = class_labels.get(disease_idx, "Unknown")
        # The scale/severity is now discarded. We return a placeholder (None or 0.0)
        return disease, None # Removed scale
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return "Unknown", None # Removed scale

weather_data = get_weather("Bangalore")
if weather_data:
    temp, hum, rain, icon_url, desc = weather_data
    _, col_w = st.columns([1, 6])
    with col_w:
        st.markdown(
            f"**{temp}°C** | {t('Humidity', lang)}: {hum}% | {t('Rain', lang)}: {rain}mm | "
            f"<img src='{icon_url}' alt='{desc}' width='25' height='25' "
            f"style='vertical-align: middle; margin-bottom: 5px;'> {t(desc, lang)}", 
            unsafe_allow_html=True
        )

st.markdown(
    f"<h1 style='text-align:center;'>{t('AgroScan - Paddy Disease Detector', lang)}</h1>", 
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    t("Upload Paddy Leaf Image", lang), 
    type=["jpg", "jpeg", "png"]
)

if uploaded_file and model:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption=t("Preview", lang), width=250)
    
    # --- THIS IS THE LOGIC YOU WANTED ---
    # Check if it's a crop image
    with st.spinner(t("Validating image...", lang)):
        if not is_crop_image(img):
            # NOT A CROP - Show error
            st.error(
                f"❌ {t('Please upload only CROP images. This appears to be a non-crop image (bike, person, building, etc.).', lang)}"
            )
            st.info(
                f"💡 {t('Tips:', lang)}\n\n"
                f"• {t('Upload images of crop leaves only', lang)}\n"
                f"• {t('Ensure good lighting and focus on the crop', lang)}\n"
                f"• {t('Avoid backgrounds with bikes, people, or buildings', lang)}"
            )
        else:
            # IS A CROP - Proceed with prediction
            # Note: predict_image no longer returns scale/severity
            with st.spinner(t("Analyzing...", lang)):
                disease, _ = predict_image(model, img) # Changed to disease, _

            # Changed to one column layout since severity is removed
            # col1, col2 = st.columns(2) # Removed
            
            # Start of the display logic (Now centered without two columns)
            
            bg_color = "#2e7d32, #4caf50" if disease == "Healthy Plant" else "#e65100, #ff8a65"
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, {bg_color}); color:white; padding:25px; 
                 border-radius:20px; text-align:center; box-shadow:0 10px 30px rgba(0,0,0,0.4); 
                 margin:10px 0 20px 0; height: 180px;">
                <h3 style="margin:0; color:white; font-size:1.8rem;">
                    {t('Disease Detected', lang)}
                </h3>
                <p style="font-size:32px; font-weight:700; margin:15px 0; color:white;">
                    {t(disease, lang)}
                </p>
            </div>
            """, unsafe_allow_html=True)

            # The second column containing Severity Level is completely removed

            st.markdown(f"### {t('Treatment & Prevention', lang)}")
            
            if disease == "Healthy Plant":
                st.balloons()
                st.success(t("Your plant is healthy! No treatment needed.", lang))
            elif disease in ["Brown Spot", "Leaf Blast", "Sheath Blight"]:
                with st.spinner(t("Getting cure advice...", lang)):
                    treatment_html, audio_text = get_treatment_from_llm(disease, lang)
                
                st.markdown(
                    f"""<div class='info-box' style='margin:15px 0;'>
                        <p style='line-height:1.9; font-size:16px; margin:0;'>{treatment_html}</p>
                    </div>""", 
                    unsafe_allow_html=True
                )

                if lang == "Kannada" and audio_text:
                    audio_bytes = get_kannada_audio_bytes(audio_text)
                    if audio_bytes: 
                        st.audio(audio_bytes, autoplay=True, format="audio/mp3")
            else:
                st.error(t("An error occurred during prediction. Please try another image.", lang))

elif not model:
    st.error(t("Model not loaded. Please check file path.", lang))

render_project_bot()
