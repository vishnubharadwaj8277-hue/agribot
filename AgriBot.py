# AgriBot.py
import os
import time
import base64
import requests
import streamlit as st
import speech_recognition as sr
from langdetect import detect
from groq import Groq
from dotenv import load_dotenv, find_dotenv
import io
from typing import List, Dict
from gtts import gTTS

# --- Import authentication, project bot, and utils ---
from project_bot import render_project_bot
from utils import (
    apply_custom_css, t, get_kannada_audio_bytes,
    check_login, render_sidebar,
    translate_to_english, translate_back
)

# -----------------------------
# Main App Logic
# -----------------------------
apply_custom_css()  # Apply CSS *before* login check
check_login()       # Login gate
render_sidebar()    # Sidebar with logout

# -----------------------------
# Load environment variables
# -----------------------------
load_dotenv(find_dotenv())
GROQ_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_KEY:
    st.error("GROQ_API_KEY is not set!")
    st.stop()

API_KEY = GROQ_KEY
PROVIDER = "GROQ"
DEFAULT_BASE = "https://api.groq.com/openai/v1"
DEFAULT_MODEL = "llama-3.3-70b-versatile"
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", DEFAULT_BASE)
MODEL = os.getenv("OPENAI_MODEL", DEFAULT_MODEL)
RETRIES = int(os.getenv("API_RETRIES", 2))

client = Groq(api_key=GROQ_KEY)
r = sr.Recognizer()


# -----------------------------
# Local session data
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_audio_hash" not in st.session_state:
    st.session_state.last_audio_hash = None
if "audio_bytes_for_message" not in st.session_state:
    st.session_state.audio_bytes_for_message = {}

# -----------------------------
# API Call with Chat History
# -----------------------------
def call_chat_api(message_history: List[Dict[str, str]], max_retries: int = RETRIES) -> str:
    if not API_KEY:
        raise EnvironmentError("Missing API key in .env")

    # STRICT AGRICULTURE SYSTEM PROMPT
    system_prompt = """
You are an expert agriculture and farming assistant for Indian farmers, especially in Karnataka.

STRICT RULES:
1. ONLY answer questions related to agriculture, farming, crops, soil, pests, irrigation, and rural topics.
2. If the question is NOT agriculture-related (e.g., politics, celebrities, tech, general knowledge), you MUST politely refuse and say:
   "ದಯವಿಟ್ಟು ಕೃಷಿ-ಸಂಬಂಧಿತ ಪ್ರಶ್ನೆಗಳನ್ನು ಮಾತ್ರ ಕೇಳಿ. ನಾನು ಕೃಷಿ ಸಹಾಯಕನಾಗಿದ್ದೇನೆ."
   or
   "Please ask only agriculture-related questions. I am an agriculture assistant."
3. Use conversation history for context.
4. Answer in Kannada if the user speaks Kannada, otherwise in English.
"""

    messages_payload = [{"role": "system", "content": system_prompt}]
    messages_payload.extend(message_history[-10:])

    url = OPENAI_API_BASE.rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    payload = {"model": MODEL, "messages": messages_payload, "temperature": 0.3, "max_tokens": 700}

    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=40)
            if resp.status_code == 200:
                return resp.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            last_error = str(e)
            time.sleep(1 * attempt)
    raise RuntimeError(f"API failed: {last_error}")

# -----------------------------
# Page Title
# -----------------------------
lang = st.session_state.lang
st.markdown(f"<h1 style='text-align:center;'>{t('Agri-Bot: Your Smart Farming Assistant', lang)}</h1>", unsafe_allow_html=True)
st.markdown(f"<h3 style='text-align:center;'>{t('Powered by AI', lang)}</h3>", unsafe_allow_html=True)

# -----------------------------
# Display Chat Messages
# -----------------------------
message_counter = 0
if st.session_state.messages is None:
    st.session_state.messages = []

for msg in st.session_state.messages:
    message_counter += 1
    msg_key = f"msg_{message_counter}"
    avatar = "🌱" if msg["role"] == "assistant" else "🧑‍🌾"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])
        if msg_key in st.session_state.audio_bytes_for_message:
            audio_bytes = st.session_state.audio_bytes_for_message[msg_key]
            if st.button(f"🔊 {t('Play Kannada', lang)}", key=f"play_btn_{msg_key}"):
                st.audio(audio_bytes, format="audio/mp3", autoplay=True)

# -----------------------------
# Input Section (Text + Voice)
# -----------------------------
user_input_text = st.chat_input(t("Type in Kannada or English…", lang))
user_input_voice = None

st.markdown("###")  # Spacer
audio_bytes_obj = st.audio_input(label=t("Or record your voice (press, speak, press again)", lang))

if audio_bytes_obj:
    audio_data_bytes = audio_bytes_obj.read()
    current_audio_hash = hash(audio_data_bytes)
    if current_audio_hash != st.session_state.last_audio_hash:
        st.session_state.last_audio_hash = current_audio_hash
        st.info(t("Processing voice input...", lang))
        try:
            wav_file = io.BytesIO(audio_data_bytes)
            with sr.AudioFile(wav_file) as source:
                audio_data = r.record(source)
            user_input_voice = r.recognize_google(audio_data, language="kn-IN")
        except Exception:
            st.error(t("Sorry, I could not understand the audio.", lang))

# -----------------------------
# Process User Input
# -----------------------------
user_input = user_input_text or user_input_voice

if user_input:
    try:
        orig_lang = detect(user_input)
    except:
        orig_lang = "en"

    st.session_state.messages.append({"role": "user", "content": user_input})
    message_counter += 1

    with st.chat_message("user", avatar="🧑‍🌾"):
        st.markdown(user_input)

    with st.spinner(t("Thinking…", lang)):
        try:
            history = st.session_state.messages
            answer = call_chat_api(history)
            final_answer = answer

            st.session_state.messages.append({"role": "assistant", "content": final_answer})
            message_counter += 1
            assistant_msg_key = f"msg_{message_counter}"

            if orig_lang == "kn":
                audio_bytes = get_kannada_audio_bytes(final_answer)
                if audio_bytes:
                    st.session_state.audio_bytes_for_message[assistant_msg_key] = audio_bytes

                # Chat history saving removed (no authentication)
                pass
            st.rerun()

        except Exception as e:
            st.error(f"Error: {e}")

# -----------------------------
# Render Floating Project Bot
# -----------------------------
render_project_bot()
