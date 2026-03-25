# pages/1_Crop_Recommender.py
import streamlit as st
import requests
import os
from groq import Groq
from datetime import datetime
import folium
import streamlit.components.v1 as components
from folium.plugins import MarkerCluster
from dotenv import load_dotenv
import io
import joblib
import numpy as np
import pandas as pd

# --- Import shared functions ---
from utils import apply_custom_css, t, language_toggle, get_kannada_audio_bytes
from project_bot import render_project_bot # (NEW) Import floating bot

# --- Apply CSS and Language Toggle ---
apply_custom_css()
with st.sidebar:
    language_toggle()

# Get current language
lang = st.session_state.lang

# ----------------- Load .env -----------------
load_dotenv()
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ----------------- Groq Client -----------------
client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

# ----------------- Session State (Page Specific) -----------------
if "selected_crop" not in st.session_state: st.session_state.selected_crop = None
if "location" not in st.session_state: st.session_state.location = {"state": "", "district": "", "month": ""}
if "crops" not in st.session_state: st.session_state.crops = None
if "lat" not in st.session_state: st.session_state.lat = 12.9716
if "lon" not in st.session_state: st.session_state.lon = 77.5946

# ----------------- Weather API -----------------
@st.cache_data(ttl=300)
def get_weather(lat, lon):
    if not OPENWEATHER_API_KEY: return {"temp": 25, "humidity": 60, "rainfall": 0, "desc": "Clear", "icon": "01d"}
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&units=metric&appid={OPENWEATHER_API_KEY}"
        data = requests.get(url, timeout=10).json()
        if data.get("cod") == 200:
            return { "temp": round(data["main"]["temp"]), "humidity": data["main"]["humidity"], "rainfall": data.get("rain", {}).get("1h", 0), "desc": data["weather"][0]["description"].title(), "icon": data["weather"][0]["icon"] }
    except: pass
    return {"temp": 25, "humidity": 60, "rainfall": 0, "desc": "Clear", "icon": "01d"}

# ----------------- ML + LLM: Hybrid Recommendation -----------------
@st.cache_resource
def load_ml_components():
    try:
        model = joblib.load('ensemble_crop_model.joblib')
        scaler = joblib.load('crop_scaler.joblib')
        return model, scaler
    except Exception as e:
        st.error(f"Error loading ML model: {e}")
        return None, None

ml_model, ml_scaler = load_ml_components()

def get_crop_recommendations(n, p, k, ph, temp, hum, rain, state, district, month, lang):
    # 1. Use the trained Ensemble ML Model to predict the top 3 crops
    if ml_model and ml_scaler:
        # Prepare input matching the scaler's format (using pandas to avoid warnings)
        input_data = pd.DataFrame([[n, p, k, temp, hum, ph, rain]], 
                                  columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'])
        input_scaled = ml_scaler.transform(input_data)
        
        # Get probabilities to find the top 3 recommendations
        probabilities = ml_model.predict_proba(input_scaled)[0]
        top_3_indices = np.argsort(probabilities)[-3:][::-1]
        classes = ml_model.classes_
        
        predicted_crops = [classes[idx].capitalize() for idx in top_3_indices]
    else:
        predicted_crops = ["Rice", "Maize", "Groundnut"] # Fallback if model missing

    # 2. Use the LLM to generate the short reasons in the correct language
    if not client: 
        return [f"1. {predicted_crops[0]} - Default", f"2. {predicted_crops[1]} - Default", f"3. {predicted_crops[2]} - Default"], " ".join(predicted_crops)
        
    prompt = f"The machine learning model has recommended these 3 crops: {predicted_crops[0]}, {predicted_crops[1]}, and {predicted_crops[2]}. For an Indian farmer with Soil (N={n}, P={p}, K={k}, pH={ph}) and Weather ({temp}C, {hum}% humidity, {rain}mm rain) in {state}, {district} during {month}. Give a short 1-sentence reason for EACH crop. Format EXACTLY like this:\n1. {predicted_crops[0]} - [short reason]\n2. {predicted_crops[1]} - [short reason]\n3. {predicted_crops[2]} - [short reason]"
    
    if lang == "Kannada": prompt += " Answer in Kannada. Use 1. 2. 3."
    
    try:
        chat = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}], 
            model="llama-3.3-70b-versatile", 
            temperature=0.3, 
            max_tokens=300
        )
        response = chat.choices[0].message.content.strip()
        crops = [line.strip() for line in response.split('\n') if line.strip().startswith(('1.', '2.', '3.'))]
        
        # Ensure we always return 3 items
        while len(crops) < 3: crops.append(f"{len(crops)+1}. Unknown - Error")
        
        audio_text = " ".join([c.split('-')[0].replace('1.', '').replace('2.', '').replace('3.', '').strip() for c in crops[:3]])
        return crops[:3], audio_text
    except Exception as e: 
        st.error(f"LLM Error: {e}")
        
    return [f"1. {predicted_crops[0]} - ML Recommendation", f"2. {predicted_crops[1]} - ML Recommendation", f"3. {predicted_crops[2]} - ML Recommendation"], " ".join(predicted_crops)

# ----------------- LLM: Full Crop Guide -----------------
def get_crop_guide(crop, state, district, month, lang):
    if not client: return t("Guide not available in demo mode.", lang)
    prompt = f"Complete growing guide for {crop} in {state}, {district} during {month}. Include: Soil preparation, Sowing time, Seed rate, Spacing, Irrigation, Fertilizer (NPK), Pest control, Harvesting, Yield per acre, Market tips. Use bullets."
    if lang == "Kannada": prompt += " Answer in Kannada."
    try:
        chat = client.chat.completions.create(messages=[{"role": "user", "content": prompt}], model="llama-3.3-70b-versatile", temperature=0.3, max_tokens=800)
        return chat.choices[0].message.content.strip()
    except Exception as e: return t(f"Error: {e}", lang)

# ----------------- CROP MAP DATA (RESTORED) -----------------
FAMOUS_CROPS = {
    "Punjab": "Wheat 🌾", "Haryana": "Rice 🌾", "Uttar Pradesh": "Sugarcane 🍬", "Bihar": "Maize 🌽", "West Bengal": "Rice 🌾", "Odisha": "Rice 🌾", "Maharashtra": "Cotton ☁️", "Gujarat": "Groundnut 🥜", "Karnataka": "Ragi 🌾", "Kerala": "Coconut 🥥", "Tamil Nadu": "Rice 🌾", "Madhya Pradesh": "Soybean 🌱", "Andhra Pradesh": "Chillies 🌶️", "Telangana": "Cotton ☁️", "Rajasthan": "Bajra 🌾", "Assam": "Tea 🍃"
}
STATE_COORDS = {
    "Punjab": (31.15, 75.34), "Haryana": (29.06, 76.08), "Uttar Pradesh": (26.84, 80.94), "Bihar": (25.59, 85.13), "West Bengal": (22.57, 88.36), "Odisha": (20.27, 85.84), "Maharashtra": (19.07, 72.88), "Gujarat": (22.30, 70.80), "Karnataka": (12.97, 77.59), "Kerala": (10.85, 76.27), "Tamil Nadu": (13.08, 80.27), "Madhya Pradesh": (23.25, 77.41), "Andhra Pradesh": (15.91, 79.74), "Telangana": (17.39, 78.49), "Rajasthan": (26.91, 75.79), "Assam": (26.20, 92.93)
}
INDIA_STATES_DISTRICTS = {
    "Andhra Pradesh": ["Anantapur", "Chittoor", "Guntur", "Krishna", "Kurnool", "Visakhapatnam"],
    "Karnataka": [ "Bagalkote", "Ballari", "Belagavi", "Bengaluru Rural", "Bengaluru Urban", "Bidar", "Chamarajanagara", "Chikkaballapura", "Chikkamagaluru", "Chitradurga", "Dakshina Kannada", "Davanagere", "Dharwad", "Gadag", "Hassan", "Haveri", "Kalaburagi", "Kodagu", "Kolar", "Koppal", "Mandya", "Mysuru", "Raichur", "Ramanagara", "Shivamogga", "Tumakuru", "Udupi", "Uttara Kannada", "Vijayapura", "Yadgir" ],
    "Kerala": ["Alappuzha", "Ernakulam", "Idukki", "Kannur", "Kollam", "Kottayam", "Kozhikode", "Malappuram", "Palakkad", "Thiruvananthapuram"],
    "Maharashtra": ["Ahmednagar", "Aurangabad", "Kolhapu", "Mumbai City", "Mumbai Suburban", "Nagpur", "Nashik", "Pune", "Satara", "Thane"],
    "Tamil Nadu": ["Chennai", "Coimbatore", "Kanchipuram", "Kanyakumari", "Madurai", "Salem", "Tiruchirappalli", "Vellore"],
    "Uttar Pradesh": ["Agra", "Aligarh", "Allahabad", "Bareilly", "Ghaziabad", "Gorakhpur", "Kanpur", "Lucknow", "Meerut", "Varanasi"]
}
MONTHS_LIST = [ "January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December" ]

# ----------------- WEATHER BAR -----------------
lat, lon = st.session_state.lat, st.session_state.lon; weather = get_weather(lat, lon)
temp, hum, rain = weather["temp"], weather["humidity"], weather["rainfall"]; desc, icon = weather["desc"], weather["icon"]
icon_url = f"https://openweathermap.org/img/wn/{icon}@2x.png"
_, col_w = st.columns([1, 6]);
with col_w:
    c1, c2 = st.columns([2, 1])
    with c1: st.markdown(f"**{temp}°C** | {t('Humidity', lang)}: {hum}% | {t('Rain', lang)}: {rain}mm", unsafe_allow_html=True)
    with c2: st.markdown(f'<img src="{icon_url}" alt="{desc}" width="25" height="25"> {desc}', unsafe_allow_html=True)

# ----------------- Main UI -----------------
st.markdown(f"<h1 style='text-align:center;'>{t('AI Crop Recommender', lang)}</h1>", unsafe_allow_html=True)
tab1, tab2 = st.tabs([ f"📍 {t('Recommend Crops', lang)}", f"🗺️ {t('Crop Map', lang)}" ]) # Only two tabs

# ----------------- TAB 1: RECOMMEND CROPS -----------------
with tab1:
    st.markdown(f"<h3 style='text-align:center;'>{t('Enter Soil & Location Data', lang)}</h3>", unsafe_allow_html=True)
    states_list = sorted(list(INDIA_STATES_DISTRICTS.keys()))
    col1, col2, col3 = st.columns(3)
    with col1: selected_state_rec = st.selectbox( t("State", lang), ["Select State"] + states_list, key="state_select_rec" )
    with col2:
        district_options = [t("Select a state first", lang)]; district_disabled = True
        if selected_state_rec != "Select State": district_options = ["Select District"] + sorted(INDIA_STATES_DISTRICTS[selected_state_rec]); district_disabled = False
        selected_district = st.selectbox( t("District", lang), options=district_options, disabled=district_disabled, key="district_select_unified" )
    with col3:
        current_month = datetime.now().strftime("%B")
        default_month_index = MONTHS_LIST.index(current_month) + 1 if current_month in MONTHS_LIST else 0
        selected_month = st.selectbox(t("Month", lang), ["Select Month"] + MONTHS_LIST, index=default_month_index, key="month_select")
    if st.button(t("Save Location", lang)):
         if selected_state_rec == "Select State" or selected_district in ["Select District", t("Select a state first", lang)] or selected_month == "Select Month": st.error(t("Please select a valid State, District, and Month.", lang))
         else: st.session_state.location = {"state": selected_state_rec.upper(), "district": selected_district.upper(), "month": selected_month.upper()}; st.success(t("Location saved!", lang) + f" ({selected_state_rec}, {selected_district})")
    st.markdown(f"### {t('Soil & Weather Data', lang)}")
    col1, col2, col3 = st.columns(3); col4, col5, col6 = st.columns(3)
    with col1: n = st.number_input(t("Nitrogen (N)", lang), 0.0, value=50.0, step=1.0)
    with col2: p = st.number_input(t("Phosphorus (P)", lang), 0.0, value=25.0, step=1.0)
    with col3: k = st.number_input(t("Potassium (K)", lang), 0.0, value=25.0, step=1.0)
    with col4: ph = st.number_input(t("pH", lang), 0.0, 14.0, value=6.5, step=0.1)
    with col5: temp_in = st.number_input(t("Temperature (°C)", lang), 0.0, value=float(temp), step=0.5)
    with col6: hum_in = st.number_input(t("Humidity (%)", lang), 0.0, value=float(hum), step=1.0)
    rainfall = st.number_input(t("Rainfall (mm)", lang), 0.0, value=100.0, step=10.0)
    if st.button(t("Get Crop Recommendations", lang), type="primary"):
        if not st.session_state.location["state"]: st.error(t("Please save a location first.", lang))
        else:
            crops, audio_text = get_crop_recommendations( n, p, k, ph, temp_in, hum_in, rainfall, st.session_state.location["state"], st.session_state.location["district"], st.session_state.location["month"], lang )
            st.session_state.crops = crops
            if lang == "Kannada" and audio_text: 
                audio_bytes = get_kannada_audio_bytes(audio_text)
                if audio_bytes: st.audio(audio_bytes, autoplay=True, format="audio/mp3")

    if st.session_state.get("crops"):
        st.markdown(f"### {t('Top 3 Recommended Crops', lang)}")
        rank_labels = [t("Highly Recommended", lang), t("Good Choice", lang), t("Viable Option", lang)]; colors = ["#2e7d32", "#f9a825", "#e65100"]
        recommendations = st.session_state.crops
        while len(recommendations) < 3: recommendations.append("Error fetching recommendation")
        for i, line in enumerate(recommendations[:3]):
             parts = line.split('-', 1)
             crop_name = parts[0].strip().lstrip('1234567890. ') if len(parts) > 0 else "Unknown"
             reason = parts[1].strip() if len(parts) > 1 else "No reason provided"
             col_a, col_b = st.columns([1, 4])
             with col_a: st.markdown(f"""<div style='background:{colors[i]}; color:white; padding:12px; border-radius:12px; text-align:center; font-weight:600;'>{rank_labels[i]}</div>""", unsafe_allow_html=True)
             with col_b:
                 is_error = "error" in crop_name.lower() or "unknown" in crop_name.lower()
                 if st.button(crop_name, key=f"crop_{i}", use_container_width=True, disabled=is_error):
                      st.session_state.selected_crop = crop_name
             st.markdown(f"<small style='color:#1B5E20;'>{reason}</small>", unsafe_allow_html=True)
        if st.session_state.get("selected_crop"):
            guide = get_crop_guide( st.session_state.selected_crop, st.session_state.location["state"], st.session_state.location["district"], st.session_state.location["month"], lang )
            st.markdown(f"### {t('Complete Guide for', lang)} **{st.session_state.selected_crop}**")
            st.markdown(f"""<div style='background:rgba(255,255,255,0.95); padding:25px; border-radius:15px; color:#1B5E20; line-height:2;'>{guide.replace('•', '<br>•')}</div>""", unsafe_allow_html=True)
            if lang == "Kannada": 
                audio_bytes = get_kannada_audio_bytes(guide[:500]) # Limit guide to 500 chars for audio
                if audio_bytes: st.audio(audio_bytes, autoplay=True, format="audio/mp3")

# ----------------- TAB 2: CROP MAP (STATIC) -----------------
with tab2:
    st.markdown(f"<h3 style='text-align:center;'>{t('Famous Crops by State (India)', lang)}</h3>", unsafe_allow_html=True); st.markdown(f"<p style='text-align:center;'>{t('This is a static map showing major crops.', lang)}</p>", unsafe_allow_html=True)
    m = folium.Map(location=[22.97, 78.65], zoom_start=5); marker_cluster = MarkerCluster().add_to(m)
    for state, crop in FAMOUS_CROPS.items():
        coords = STATE_COORDS.get(state)
        if coords: popup = f"<b>{state}</b><br>{t('Famous Crop', lang)}: {t(crop, lang)}"; folium.Marker( location=coords, popup=popup, tooltip=f"{state}: {crop}", icon=folium.Icon(color='green', icon='leaf')).add_to(marker_cluster)
    map_html = m._repr_html_()
    components.html(map_html, height=600)

# (NEW) Render the floating bot at the end
render_project_bot()
