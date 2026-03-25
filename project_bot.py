# project_bot.py
import streamlit as st
from groq import Groq
from dotenv import load_dotenv
import os
from streamlit_modal import Modal 
import streamlit.components.v1 as components

# -----------------
# Load API Key
# -----------------
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    print("Warning: GROQ_API_KEY not found for project_bot.py")

client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

# -----------------
# The System Prompt
# -----------------
PROJECT_CONTEXT = """
You are a 'Project Helper' bot for an app called 'Agri-Bot'. 
Your ONLY job is to answer questions about the features of this app. 
Do NOT answer any other questions (like 'what is the weather' or 'who is president'). 
If asked something else, politely say "I can only answer questions about the Agri-Bot app."

Here is a summary of the app's features:

1.  **Agri-Bot (Main Page):** A voice-enabled chatbot for farmers to ask farming questions in English and Kannada.

2.  **Crop Recommender:** Recommends the Top 3 crops based on user's State, District, and soil data (N, P, K, pH). It also provides a full growing guide for any selected crop and includes a static Crop Map of India.

3.  **Policy Portal:** Lists real Karnataka Government agricultural policies (like PM KISAN, Organic Farming Policy). Users can click "Show Details" to read a summary in a scrollable box and click a button to read the official PDF.
"""

# -----------------
# The Chatbot's API Call
# -----------------
def call_project_bot_api(message_history):
    if not client:
        return "Chatbot API is not configured. Please check your .env file."
    
    messages_payload = [
        {"role": "system", "content": PROJECT_CONTEXT}
    ]
    messages_payload.extend(message_history[-6:]) # Add last 6 messages
    
    try:
        chat = client.chat.completions.create(
            messages=messages_payload,
            model="llama-3.3-70b-versatile",
            temperature=0.2, 
            max_tokens=300
        )
        return chat.choices[0].message.content.strip()
    except Exception as e:
        print(f"ProjectBot Error: {e}")
        return f"Sorry, I had an error: {e}"

# -----------------
# The function to render the floating bot
# -----------------
def render_project_bot():
    
    # 1. Add all CSS styling
    st.markdown("""
    <style>
    /* Style the modal header */
    div[data-testid="stModal"] > div:first-child > div:first-child {
        background-color: #E8F5E9; /* Light green header */
    }
    /* Style the modal title */
    div[data-testid="stModal"] > div:first-child > div:first-child > div:first-child {
        color: #1B5E20 !important; /* Dark green title */
        font-weight: 600;
    }
    
    /* Style the modal's official Close button */
    div[data-testid="stModal"] button[aria-label="Close"] {
        background-color: #FFFFFF;
        color: #1B5E20;
        border-radius: 50%;
        border: 1px solid #1B5E20;
    }
    div[data-testid="stModal"] button[aria-label="Close"]:hover {
        background-color: #D3EADC;
    }

    /* Style the floating button */
    button[data-testid="stButton"][key="open-chat-modal"] {
        position: fixed;
        bottom: 3rem;
        right: 1.5rem;
        background-color: #2E7D32;
        color: white;
        border: none;
        border-radius: 50%;
        width: 55px;
        height: 55px;
        font-size: 28px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.25);
        z-index: 1000;
        cursor: pointer;
    }
    button[data-testid="stButton"][key="open-chat-modal"]:hover {
        background-color: #1B5E20;
        transform: scale(1.1);
    }
    
    /* Hide the label for the text_input inside the form */
    div[data-testid="stForm"] label {
        display: none;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # 2. Define the Modal
    modal = Modal(
        "💬 Agri-Bot Help Center", 
        key="project-bot-modal",
        max_width=600
    )

    # 3. Create the floating button
    st.button("💬", key="open-chat-modal")

    # 4. Initialize chat history and processing flags
    if "project_bot_messages" not in st.session_state:
        st.session_state.project_bot_messages = [
            {"role": "assistant", "content": "Hi! How can I help you understand this app?"}
        ]
    
    # Flag to track if we're currently waiting for a response
    if "project_bot_awaiting_response" not in st.session_state:
        st.session_state.project_bot_awaiting_response = False

    # 5. Open the modal if the button is clicked
    if st.session_state.get("open-chat-modal"):
        modal.open()

    # 6. Define the content of the modal
    if modal.is_open():
        with modal.container():
            
            # 6a. Display past messages
            chat_box = st.container(height=350, border=False)
            with chat_box:
                for msg in st.session_state.project_bot_messages:
                    avatar = "🌱" if msg["role"] == "assistant" else "🧑‍🌾" 
                    with st.chat_message(msg["role"], avatar=avatar):
                        st.markdown(msg["content"])
            
            # 6b. Check if waiting for response - show spinner without rerun
            if st.session_state.project_bot_awaiting_response:
                st.info("⏳ Getting response...")
            
            # 6c. Use a form for the input *inside the modal*
            with st.form(key="bot_chat_form", clear_on_submit=True):
                user_text = st.text_input(
                    "Your message:", 
                    placeholder="Ask about app features...", 
                    label_visibility="collapsed",
                    key="bot_chat_input"
                )
                submitted = st.form_submit_button("Send", disabled=st.session_state.project_bot_awaiting_response)
            
            if submitted and user_text and not st.session_state.project_bot_awaiting_response:
                st.session_state.project_bot_messages.append({"role": "user", "content": user_text})
                st.session_state.project_bot_awaiting_response = True
                st.rerun()

            # 6d. Process API response ONLY if awaiting and last message is from user
            if (st.session_state.project_bot_awaiting_response and 
                len(st.session_state.project_bot_messages) > 0 and
                st.session_state.project_bot_messages[-1]["role"] == "user"):
                
                response = call_project_bot_api(st.session_state.project_bot_messages)
                st.session_state.project_bot_messages.append({"role": "assistant", "content": response})
                st.session_state.project_bot_awaiting_response = False
                st.rerun()