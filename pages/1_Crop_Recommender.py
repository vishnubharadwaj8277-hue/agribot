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


# ----------------- Load/Save User Data -----------------
def load_user_data():
    """Loads soil/location data from local session for the current user."""
    # Return defaults if no data
    return {
        "location": {"state": "Select State", "district": "Select a state first", "month": datetime.now().strftime("%B")},
        "soil": {"n": 50.0, "p": 25.0, "k": 25.0, "ph": 6.5, "rainfall": 100.0}
    }

def save_user_data(data_to_save):
    """Saves soil/location data to local session for the current user."""
    st.session_state.user_data = data_to_save

# ----------------- Load initial data -----------------
if "user_data" not in st.session_state:
    st.session_state.user_data = load_user_data()

# ----------------- Session State (Page Specific) -----------------
if "selected_crop" not in st.session_state: st.session_state.selected_crop = None
if "crops" not in st.session_state: st.session_state.crops = None
if "lat" not in st.session_state: st.session_state.lat = 12.9716
if "lon" not in st.session_state: st.session_state.lon = 77.5946
if "market_prediction" not in st.session_state: st.session_state.market_prediction = None
# Removed: if "stress_test_result" not in st.session_state: st.session_state.stress_test_result = None

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

def call_groq_api(prompt, max_tokens=300):
    if not client: return t("LLM service not available.", lang)
    try:
        chat = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}], 
            model="llama-3.3-70b-versatile", # Using the fast Llama 3 on Groq
            temperature=0.3, 
            max_tokens=max_tokens
        )
        return chat.choices[0].message.content.strip()
    except Exception as e:
        return t(f"LLM Error: {e}", lang)

def get_crop_recommendations(n, p, k, ph, temp, hum, rain, state, district, month, lang):
    prompt = f"Recommend 3 crops for Indian farmer. Soil: N={n}, P={p}, K={k}, pH={ph}. Weather: {temp} deg C, {hum}% humidity, {rain} mm rain. Location: {state}, {district}, {month}. Rank: 1=best, 2=good, 3=viable. Format:\n1. [CROP] - [short reason]\n2. [CROP] - [short reason]\n3. [CROP] - [short reason]"
    if lang == "Kannada": prompt += " Answer in Kannada. Use 1. 2. 3."
    response = call_groq_api(prompt)
    if response:
        crops = [line.strip() for line in response.split('\n') if line.strip().startswith(('1.', '2.', '3.'))]
        while len(crops) < 3: crops.append(f"{len(crops)+1}. Unknown - Error")
        if len(crops) >= 3:
            audio_text = " ".join([c.split('-')[0].replace('1.', '').replace('2.', '').replace('3.', '').strip() for c in crops[:3]])
            return crops[:3], audio_text
    return ["1. Rice - Error", "2. Maize - Error", "3. Groundnut - Error"], "Rice Maize Groundnut"

def get_crop_guide(crop, state, district, month, lang):
    prompt = f"Complete growing guide for {crop} in {state}, {district} during {month}. Include: Soil preparation, Sowing time, Seed rate, Spacing, Irrigation, Fertilizer (NPK), Pest control, Harvesting, Yield per acre, Market tips. Use bullets."
    if lang == "Kannada": prompt += " Answer in Kannada."
    return call_groq_api(prompt, max_tokens=800)

# --- NEW FEATURE FUNCTIONS ---

def get_market_prediction(crop, state, district, month, lang):
    """Generates market price and procurement advice."""
    prompt = f"""
    You are a commodity market expert for Indian agriculture. 
    Your goal is to give a farmer a quick, actionable market and procurement forecast for {crop} in {district}, {state} for the upcoming harvest period in {month}. 
    Analyze current (simulated) global trends, MSP data, and local harvest cycles.
    Format your response as three distinct, short, bulleted points.
    1. [Price Trend Prediction]
    2. [Procurement/MSP Advice]
    3. [Selling Strategy Tip]
    """
    if lang == "Kannada": prompt += " Answer concisely in Kannada."
    return call_groq_api(prompt, max_tokens=300)

# Removed: run_farm_stress_test function as requested

# --- STATIC DATA (Expanded) ---
FAMOUS_CROPS = { "Punjab": "Wheat 🌾", "Haryana": "Rice 🌾", "Uttar Pradesh": "Sugarcane 🍬", "Bihar": "Maize 🌽", "West Bengal": "Rice 🌾", "Odisha": "Rice 🌾", "Maharashtra": "Cotton ☁️", "Gujarat": "Groundnut 🥜", "Kerala": "Coconut 🥥", "Tamil Nadu": "Rice 🌾", "Madhya Pradesh": "Soybean 🌱", "Andhra Pradesh": "Chillies 🌶️", "Telangana": "Cotton ☁️", "Rajasthan": "Bajra 🌾", "Assam": "Tea 🍃" }
STATE_COORDS = { "Punjab": (31.15, 75.34), "Haryana": (29.06, 76.08), "Uttar Pradesh": (26.84, 80.94), "Bihar": (25.59, 85.13), "West Bengal": (22.57, 88.36), "Odisha": (20.27, 85.84), "Maharashtra": (19.07, 72.88), "Gujarat": (22.30, 70.80), "Karnataka": (14.52, 75.72), "Kerala": (10.85, 76.27), "Tamil Nadu": (13.08, 80.27), "Madhya Pradesh": (23.25, 77.41), "Andhra Pradesh": (15.91, 79.74), "Telangana": (17.39, 78.49), "Rajasthan": (26.91, 75.79), "Assam": (26.20, 92.93) }

# EXPANDED INDIA_STATES_DISTRICTS list for Recommender Dropdown
INDIA_STATES_DISTRICTS = { 
    "Andhra Pradesh": ["Anantapur", "Chittoor", "Guntur", "Krishna", "Kurnool", "Visakhapatnam"],
    "Assam": ["Cachar", "Dhubri", "Kamrup", "Lakhimpur"],
    "Bihar": ["Gaya", "Patna", "Muzaffarpur", "Nalanda", "Purnia"],
    "Gujarat": ["Ahmedabad", "Bhavnagar", "Jamnagar", "Kachchh", "Rajkot", "Surat"],
    "Haryana": ["Ambala", "Bhiwani", "Faridabad", "Hisar", "Karnal", "Gurgaon"],
    "Karnataka": [ 
        "Bagalkote", "Ballari", "Belagavi", "Bengaluru Rural", "Bengaluru Urban", "Bidar", 
        "Chamarajanagara", "Chikkaballapura", "Chikkamagaluru", "Chitradurga", "Dakshina Kannada", 
        "Davanagere", "Dharwad", "Gadag", "Hassan", "Haveri", "Kalaburagi", "Kodagu", "Kolar", 
        "Koppal", "Mandya", "Mysuru", "Raichur", "Ramanagara", "Shivamogga", "Tumakuru", 
        "Udupi", "Uttara Kannada", "Vijayapura", "Yadgir" 
    ],
    "Kerala": ["Alappuzha", "Ernakulam", "Idukki", "Kannur", "Kollam", "Kottayam", "Kozhikode", "Malappuram", "Palakkad", "Thiruvananthapuram"],
    "Madhya Pradesh": ["Bhopal", "Gwalior", "Indore", "Jabalpur", "Ujjain"],
    "Maharashtra": ["Ahmednagar", "Aurangabad", "Kolhapu", "Mumbai City", "Mumbai Suburban", "Nagpur", "Nashik", "Pune", "Satara", "Thane"],
    "Odisha": ["Cuttack", "Ganjam", "Khurda", "Puri", "Sambalpur"],
    "Punjab": ["Amritsar", "Bathinda", "Firozpur", "Ludhiana", "Patiala"],
    "Rajasthan": ["Ajmer", "Bikaner", "Jaipur", "Jodhpur", "Udaipur"],
    "Tamil Nadu": ["Chennai", "Coimbatore", "Kanchipuram", "Kanyakumari", "Madurai", "Salem", "Tiruchirappalli", "Vellore"], 
    "Telangana": ["Hyderabad", "Karimnagar", "Khammam", "Warangal"],
    "Uttar Pradesh": ["Agra", "Aligarh", "Allahabad", "Bareilly", "Ghaziabad", "Gorakhpur", "Kanpur", "Lucknow", "Meerut", "Varanasi"],
    "West Bengal": ["Bankura", "Birbhum", "Hooghly", "Kolkata", "Nadia", "Paschim Midnapore"]
}

MONTHS_LIST = [ "January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December" ]

# --- ENHANCED KARNATAKA DISTRICT CROP DATA FOR MAP (ALL DISTRICTS IN LIST) ---
# This dictionary now contains all districts listed in INDIA_STATES_DISTRICTS["Karnataka"]
KARNATAKA_DISTRICT_CROPS = {
    # South/Cauvery Basin
    "Mandya": {"coords": (12.53, 76.90), "crops": "Sugarcane, Paddy, Coconut"},
    "Mysuru": {"coords": (12.30, 76.65), "crops": "Ragi, Turmeric, Paddy"},
    "Chamarajanagara": {"coords": (11.93, 77.12), "crops": "Turmeric, Banana, Maize"},
    "Bengaluru Urban": {"coords": (12.97, 77.59), "crops": "Ragi, Rice, Vegetables"},
    "Bengaluru Rural": {"coords": (13.00, 77.40), "crops": "Ragi, Mango, Pulses"},
    "Ramanagara": {"coords": (12.75, 77.26), "crops": "Sericulture, Ragi, Coconut"},
    "Kolar": {"coords": (13.13, 78.13), "crops": "Tomato, Groundnut, Pulses"},
    "Chikkaballapura": {"coords": (13.43, 77.72), "crops": "Groundnut, Grapes, Ragi"},
    # Malnad/Coastal
    "Kodagu": {"coords": (12.33, 75.74), "crops": "Coffee, Cardamom, Paddy"},
    "Chikkamagaluru": {"coords": (13.31, 75.77), "crops": "Coffee, Arecanut, Paddy"},
    "Hassan": {"coords": (13.01, 76.10), "crops": "Coffee, Potato, Paddy"},
    "Shivamogga": {"coords": (13.92, 75.56), "crops": "Arecanut, Paddy, Maize"},
    "Dakshina Kannada": {"coords": (12.91, 74.85), "crops": "Arecanut, Coconut, Rice"},
    "Udupi": {"coords": (13.34, 74.74), "crops": "Coconut, Rice, Arecanut"},
    "Uttara Kannada": {"coords": (14.65, 74.61), "crops": "Cashew, Paddy, Spice"},
    # North/Central Karnataka
    "Davanagere": {"coords": (14.46, 75.92), "crops": "Paddy, Maize, Cotton"},
    "Chitradurga": {"coords": (14.22, 76.40), "crops": "Groundnut, Jowar, Maize"},
    "Tumakuru": {"coords": (13.34, 77.10), "crops": "Coconut, Ragi, Groundnut"},
    "Ballari": {"coords": (15.14, 76.92), "crops": "Paddy, Cotton, Jowar"},
    "Koppal": {"coords": (15.35, 76.08), "crops": "Paddy, Cotton, Tur Dal"},
    "Raichur": {"coords": (16.21, 77.34), "crops": "Paddy, Cotton, Jowar"},
    # Hyderabad-Karnataka Region (Kalyana Karnataka)
    "Kalaburagi": {"coords": (17.33, 76.83), "crops": "Red Gram (Tur), Jowar, Maize"},
    "Yadgir": {"coords": (16.76, 77.13), "crops": "Red Gram (Tur), Cotton, Jowar"},
    "Bidar": {"coords": (17.91, 77.51), "crops": "Red Gram (Tur), Jowar, Sugarcane"},
    # Belagavi Region
    "Belagavi": {"coords": (15.85, 74.50), "crops": "Sugarcane, Jowar, Groundnut"},
    "Dharwad": {"coords": (15.46, 75.00), "crops": "Jowar, Wheat, Bengal Gram"},
    "Haveri": {"coords": (14.79, 75.45), "crops": "Maize, Cotton, Paddy"},
    "Gadag": {"coords": (15.35, 75.62), "crops": "Jowar, Cotton, Groundnut"},
    "Bagalkote": {"coords": (16.18, 75.66), "crops": "Jowar, Bajra, Sugarcane"},
    "Vijayapura": {"coords": (16.82, 75.71), "crops": "Jowar, Sunflower, Grapes"},
}
# --- END ENHANCED KARNATAKA DATA ---

# ----------------- Weather Header -----------------
lat, lon = st.session_state.lat, st.session_state.lon; weather = get_weather(lat, lon)
temp, hum, rain = weather["temp"], weather["humidity"], weather["rainfall"]; desc, icon = weather["desc"], weather["icon"]
icon_url = f"https://openweathermap.org/img/wn/{icon}@2x.png" 
_, col_w = st.columns([1, 6]);
with col_w:
    st.markdown(f"**{temp}°C** | {t('Humidity', lang)}: {hum}% | {t('Rain', lang)}: {rain}mm | <img src='{icon_url}' alt='{desc}' width='25' height='25' style='vertical-align: middle; margin-bottom: 5px;'> {t(desc, lang)}", unsafe_allow_html=True)

st.markdown(f"<h1 style='text-align:center;'>{t('AI Crop Recommender', lang)}</h1>", unsafe_allow_html=True)
tab1, tab2 = st.tabs([ f"📍 {t('Recommend Crops', lang)}", f"🗺️ {t('Crop Map', lang)}" ]) 

with tab1:
    st.markdown(f"<h3 style='text-align:center;'>{t('Enter Soil & Location Data', lang)}</h3>", unsafe_allow_html=True)
    states_list = sorted(list(INDIA_STATES_DISTRICTS.keys()))
    
    saved_loc = st.session_state.user_data.get("location", {})
    saved_soil = st.session_state.user_data.get("soil", {})
    
    try:
        # Adjusted index lookup to handle the expanded list gracefully
        state_index = states_list.index(saved_loc.get("state", "Select State")) + 1
    except ValueError:
        state_index = 0

    col1, col2, col3 = st.columns(3)
    with col1: 
        selected_state_rec = st.selectbox( 
            t("State", lang), 
            ["Select State"] + states_list, 
            key="state_select_rec", 
            index=state_index
        )
    with col2:
        district_options = [t("Select a state first", lang)]; district_disabled = True; district_index = 0
        if selected_state_rec != "Select State": 
            district_options = ["Select District"] + sorted(INDIA_STATES_DISTRICTS.get(selected_state_rec, [])) # Use .get for safety
            district_disabled = False
            try:
                district_index = district_options.index(saved_loc.get("district", "Select District"))
            except ValueError:
                district_index = 0
        selected_district = st.selectbox( t("District", lang), options=district_options, disabled=district_disabled, key="district_select_unified", index=district_index)
    with col3:
        current_month = datetime.now().strftime("%B")
        saved_month = saved_loc.get("month", current_month)
        month_index = MONTHS_LIST.index(saved_month) + 1 if saved_month in MONTHS_LIST else 0
        selected_month = st.selectbox(t("Month", lang), ["Select Month"] + MONTHS_LIST, index=month_index, key="month_select")

    if st.button(t("Save Location", lang)):
         if selected_state_rec == "Select State" or selected_district in ["Select District", t("Select a state first", lang)] or selected_month == "Select Month": st.error(t("Please select a valid State, District, and Month.", lang))
         else: 
             st.session_state.user_data["location"] = {"state": selected_state_rec, "district": selected_district, "month": selected_month}
             save_user_data(st.session_state.user_data)
             st.success(t("Location saved!", lang) + f" ({selected_state_rec}, {selected_district})")
    
    st.markdown(f"### {t('Soil & Weather Data', lang)}")
    col1, col2, col3 = st.columns(3); col4, col5, col6 = st.columns(3)
    with col1: n = st.number_input(t("Nitrogen (N)", lang), 0.0, value=saved_soil.get("n", 50.0), step=1.0, key="n_input")
    with col2: p = st.number_input(t("Phosphorus (P)", lang), 0.0, value=saved_soil.get("p", 25.0), step=1.0, key="p_input")
    with col3: k = st.number_input(t("Potassium (K)", lang), 0.0, value=saved_soil.get("k", 25.0), step=1.0, key="k_input")
    with col4: ph = st.number_input(t("pH", lang), 0.0, 14.0, value=saved_soil.get("ph", 6.5), step=0.1, key="ph_input")
    with col5: temp_in = st.number_input(t("Temperature (°C)", lang), 0.0, value=float(temp), step=0.5, key="temp_input")
    with col6: hum_in = st.number_input(t("Humidity (%)", lang), 0.0, value=float(hum), step=1.0, key="hum_input")
    rainfall = st.number_input(t("Rainfall (mm)", lang), 0.0, value=saved_soil.get("rainfall", 100.0), step=10.0, key="rain_input")
    
    if st.button(t("Get Crop Recommendations", lang), type="primary"):
        if "state" not in st.session_state.user_data.get("location", {}): 
            st.error(t("Please save a location first.", lang))
        else:
            # Clear previous results when running a new recommendation
            st.session_state.market_prediction = None
            # Removed: st.session_state.stress_test_result = None
            st.session_state.user_data["soil"] = {"n": n, "p": p, "k": k, "ph": ph, "rainfall": rainfall}
            save_user_data(st.session_state.user_data)
            loc = st.session_state.user_data["location"]
            
            with st.spinner(t("Calculating top crop recommendations...", lang)):
                 crops, audio_text = get_crop_recommendations( n, p, k, ph, temp_in, hum_in, rainfall, loc["state"], loc["district"], loc["month"], lang )
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
             with col_a:
                 st.markdown(f"""
                 <div style='background:{colors[i]}; color:white; padding:12px; border-radius:12px; text-align:center; font-weight:600;'>
                     {rank_labels[i]}
                 </div>
                 """, unsafe_allow_html=True)
             with col_b:
                 is_error = "error" in crop_name.lower() or "unknown" in crop_name.lower()
                 # Set selected_crop directly on button click and clear market prediction
                 if st.button(crop_name, key=f"crop_{i}", use_container_width=True, disabled=is_error):
                      st.session_state.selected_crop = crop_name
                      st.session_state.market_prediction = None # Clear previous market prediction
                      # Removed: st.session_state.stress_test_result = None # Clear previous stress test
                      st.rerun() # <-- CORRECTED RERUN FUNCTION

             st.markdown(f"<small>{reason}</small>", unsafe_allow_html=True) 

        # --- Selected Crop Action Panel ---
        if st.session_state.get("selected_crop"):
            st.markdown("---")
            st.markdown(f"## 🌾 {t('Action Hub for', lang)} **{st.session_state.selected_crop}**")
            
            loc = st.session_state.user_data.get("location") 
            if loc and "state" in loc:
                
                # --- MARKET PREDICTION TAB (NEW FEATURE) ---
                st.markdown(f"### 📈 {t('Live Market Price & Procurement Prediction', lang)}")
                
                if st.button(t(f"Get Market Outlook for {st.session_state.selected_crop}", lang), key="get_market_btn"):
                     with st.spinner(t("Running instant market forecast...", lang)):
                         st.session_state.market_prediction = get_market_prediction( 
                             st.session_state.selected_crop, loc["state"], loc["district"], loc["month"], lang
                         )

                if st.session_state.market_prediction:
                    st.success(t("Forecast Ready! (Powered by Groq's low-latency LLM)", lang))
                    st.markdown(st.session_state.market_prediction)
                    
                    # Added: Audio output for market prediction in Kannada
                    if lang == "Kannada": 
                        # Only take the first 300 characters for audio synthesis
                        audio_bytes = get_kannada_audio_bytes(st.session_state.market_prediction[:300])
                        if audio_bytes: st.audio(audio_bytes, format="audio/mp3")


                st.markdown("---")
                
                # Removed the entire SIMULATED STRESS TEST FEATURE block as requested

                
                # --- GUIDE (Standard Feature) ---
                st.markdown(f"### 📚 {t('Complete Guide for', lang)} **{st.session_state.selected_crop}**")
                
                with st.spinner(t(f"Fetching guide for {st.session_state.selected_crop}...", lang)):
                    guide = get_crop_guide( st.session_state.selected_crop, loc["state"], loc["district"], loc["month"], lang )
                
                st.markdown("""
<style>
.info-box { color: #1B1B1B !important; background: rgba(255,255,255,0.95) !important; }
</style>
""", unsafe_allow_html=True)
                st.markdown(f"<div class='info-box'>{guide.replace('•', '<br>•')}</div>", unsafe_allow_html=True)
                
                if lang == "Kannada": 
                    audio_bytes = get_kannada_audio_bytes(guide[:500])
                    if audio_bytes: st.audio(audio_bytes, autoplay=True, format="audio/mp3")
            else:
                st.error(t("Please save your location first.", lang))

with tab2:
    st.markdown(f"<h3 style='text-align:center;'>{t('Major Crops by Region (India)', lang)}</h3>", unsafe_allow_html=True)
    st.markdown(f"<p style='text-align:center;'>{t('Markers show famous crops for states and detailed data for ALL major Karnataka districts.', lang)}</p>", unsafe_allow_html=True)
    
    # Initialize Map centered roughly on India
    m = folium.Map(location=[22.97, 78.65], zoom_start=5); 
    marker_cluster = MarkerCluster().add_to(m)
    
    # 1. Add General State Markers (Existing Logic, non-Karnataka states)
    for state, crop in FAMOUS_CROPS.items():
        coords = STATE_COORDS.get(state)
        if coords: 
            popup = f"<b>{state}</b><br>{t('Famous Crop', lang)}: {t(crop, lang)}"
            folium.Marker( 
                location=coords, 
                popup=popup, 
                tooltip=f"{state}: {crop}", 
                icon=folium.Icon(color='green', icon='leaf')
            ).add_to(marker_cluster)
            
    # 2. Add Granular Karnataka District Markers (New Enhanced Logic)
    for district, data in KARNATAKA_DISTRICT_CROPS.items():
        coords = data["coords"]
        crops = data["crops"]
        popup = f"<b>Karnataka: {district}</b><br>{t('Major Crops', lang)}: {t(crops, lang)}"
        folium.Marker(
            location=coords,
            popup=popup,
            tooltip=f"Karnataka: {district} - {crops}",
            icon=folium.Icon(color='blue', icon='map-pin') 
        ).add_to(marker_cluster)

    map_html = m._repr_html_()
    components.html(map_html, height=600)

render_project_bot()