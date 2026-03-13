# utils.py
import streamlit as st
from deep_translator import GoogleTranslator
from gtts import gTTS
from io import BytesIO
import time
from langdetect import detect
import re


# ----------------- Session State Init -----------------
def init_session_state():
    if "lang" not in st.session_state:
        st.session_state.lang = "English"
    if "theme" not in st.session_state:
        st.session_state.theme = "light"

# ----------------- Theme Toggle -----------------
def theme_toggle():
    """Renders the theme toggle button."""
    current_theme = st.session_state.theme
    is_dark = (current_theme == "dark")
    
    new_theme_state = st.toggle(
        "🌙 **Night Mode**", 
        value=is_dark, 
        key="theme_toggle_button"
    )
    
    if (is_dark and not new_theme_state):
        st.session_state.theme = "light"
        st.rerun()
    elif (not is_dark and new_theme_state):
        st.session_state.theme = "dark"
        st.rerun()

# ----------------- Custom CSS -----------------
def apply_custom_css():
    init_session_state()
    
    if st.session_state.theme == "light":
        # Light Theme
        st.markdown("""
        <style>
        .stApp {
            background: url("https://images.unsplash.com/photo-1500595046743-cd271d6942ee?q=80&w=2074&auto=format&fit=crop") no-repeat center center fixed;
            background-size: cover;
        }
        .stApp::before {
            content: "";
            position: fixed; top: 0; left: 0; width: 100%; height: 100%;
            background: linear-gradient(135deg, rgba(0,100,0,0.5), rgba(0,0,0,0.3));
            z-index: -1;
        }

            color: #1B5E20 !important;
            text-shadow: 0 1px 2px rgba(0,0,0,0.1);
        }
        [data-testid="stSidebar"] { 
            background-color: rgba(230, 245, 230, 0.8) !important;
            backdrop-filter: blur(5px);
        }
        [data-testid="stChatContainer"] {
            background-color: rgba(255,255,255,0.9);
            border-radius: 15px;
        }
        .stButton>button {
            background: linear-gradient(45deg,#2E7D32,#4CAF50);
            color: white; border: none; border-radius: 30px;
        }
        .info-box {
            background: rgba(255,255,255,0.95) !important;
            padding: 25px;
            border-radius: 15px;
            line-height: 2;
        }
        .info-box p, .info-box li {
            color: #1B5E20 !important;
        }
        </style>
        """, unsafe_allow_html=True)
    else:
        # Dark Theme
        st.markdown("""
        <style>
        .stApp {
            background-color: #0E1117;
            background-image: radial-gradient(rgba(255,255,255,0.1) 1px, transparent 1px);
            background-size: 20px 20px;
        }
        h1, h2, h3, h4, p, li, span, div, label {
            color: #FAFAFA !important;
        }
        [data-testid="stSidebar"] { 
            background-color: #0E1117 !important;
        }
        [data-testid="stChatContainer"] {
            background-color: #111111;
            border: 1px solid #333;
        }
        .info-box {
            background: #1E1E1E !important;
            border: 1px solid #333;
            padding: 25px;
            border-radius: 15px;
            line-height: 2;
        }
        .info-box p, .info-box li {
            color: #FAFAFA !important;
        }
        </style>
        """, unsafe_allow_html=True)

# ----------------- Translator -----------------
@st.cache_data
def t(text, lang="en"):
    if lang == "English":
        return text
    if lang == "Kannada":
        try:
            return GoogleTranslator(source='en', target='kn').translate(text)
        except:
            return text
    return text

# ----------------- Language Toggle -----------------
def language_toggle():
    init_session_state()
    lang_options = ["English", "Kannada"]
    current_index = 1 if st.session_state.lang == "Kannada" else 0
    new_lang = st.selectbox(
        label="Language / ಕನ್ನಡ", 
        options=lang_options,
        index=current_index, 
        key="lang_select_sidebar"
    )
    if new_lang != st.session_state.lang:
        st.session_state.lang = new_lang
        st.rerun()

# ----------------- Kannada Audio Generator -----------------
def get_kannada_audio_bytes(text: str):
    if not text:
        return None

    cleaned_text = re.sub(r'[\*\-•]', '', text)
    cleaned_text = cleaned_text.replace('\n', ' ')
    
    if not cleaned_text.strip():
        return None
        
    try:
        tts = gTTS(text=cleaned_text, lang='kn', slow=False)
        audio_bytes_io = BytesIO()
        tts.write_to_fp(audio_bytes_io)
        audio_bytes_io.seek(0)
        return audio_bytes_io.read()
    except Exception as e:
        print(f"gTTS Error: {e}")
        return None

# ----------------- Translation Helpers -----------------
def translate_to_english(text):
    try:
        lang = detect(text)
        if lang == "en":
            return text, "en"
        return GoogleTranslator(source=lang, target="en").translate(text), lang
    except:
        return text, "kn"

def translate_back(text, target_lang):
    try:
        if target_lang == "en":
            return text
        return GoogleTranslator(source="en", target=target_lang).translate(text)
    except:
        return text

# ----------------- Login Check -----------------
def check_login():
    pass

# ----------------- Sidebar with Logout -----------------
def render_sidebar():
    lang = st.session_state.get("lang", "English")
    with st.sidebar:
        st.markdown(f"### {t('Settings', lang)}")
        language_toggle()
        theme_toggle()
        
        st.markdown("---")
        st.markdown(f"**{t('Model', lang)}:** `llama-3.3-70b-versatile`")
        st.markdown(f"**{t('Provider', lang)}:** `GROQ`")
        
        if st.button(t("Clear Chat History", lang)):
            st.session_state.messages = []
            st.session_state.audio_bytes_for_message = {}
            st.rerun()
            
        st.markdown("---")
        st.info("Welcome to Agri-Bot! No login required.")
