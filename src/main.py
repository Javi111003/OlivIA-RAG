import os
import re
import requests
import streamlit as st
from dotenv import load_dotenv
import json
import glob
from datetime import datetime

# --- CONFIGURATION ---
# Load environment variables from .env file.
load_dotenv()
API_KEY = os.getenv("API_KEY")

MODEL_NAME = "mistral-small-latest"
API_URL = "https://api.mistral.ai/v1/chat/completions"
HISTORY_DIR = "chat_history"

# --- HELPER FUNCTIONS ---

def setup():
    """Initial setup for the app."""
    # Create chat history directory if it doesn't exist
    os.makedirs(HISTORY_DIR, exist_ok=True)
    
    # Initialize session state for messages if it doesn't exist
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "system", "content": "You are a helpful and concise assistant."}]
    if "current_chat" not in st.session_state:
        st.session_state.current_chat = "new_chat"

def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

def save_chat_history(messages, filename):
    """Saves the chat history to a JSON file."""
    filepath = os.path.join(HISTORY_DIR, sanitize_filename(filename))
    with open(filepath, 'w') as f:
        json.dump(messages, f, indent=4)

def load_chat_history(filename):
    """Loads a chat history from a JSON file."""
    filepath = os.path.join(HISTORY_DIR, filename)
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return None

def get_chat_history_files():
    """Returns a sorted list of chat history files."""
    files = glob.glob(os.path.join(HISTORY_DIR, "*.json"))
    # Sort files by modification time, newest first
    files.sort(key=os.path.getmtime, reverse=True)
    return [os.path.basename(f) for f in files]


# --- PAGE SETUP ---
st.set_page_config(
    page_title="Sophisticated Chatbot",
    page_icon="✨",
    layout="centered",
    initial_sidebar_state="auto",
)
load_css(os.path.join(os.path.dirname(__file__), "style.css"))

# Run setup
setup()

def sanitize_filename(filename):
    # Remove invalid characters for Windows filenames
    return re.sub(r'[<>:"/\\|?*]', '', filename)

# --- API COMMUNICATION ---
def get_mistral_stream(messages):
    """Sends a request to the Mistral API and yields the response chunks."""
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    body = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": 0.7,
        "stream": True,
    }
    try:
        response = requests.post(API_URL, headers=headers, json=body, stream=True)
        response.raise_for_status()
        
        for line in response.iter_lines():
            if line:
                line_str = line.decode('utf-8')
                if line_str.startswith('data: '):
                    json_str = line_str[len('data: '):]
                    if json_str.strip() == '[DONE]':
                        break
                    try:
                        chunk = json.loads(json_str)
                        if 'choices' in chunk and chunk['choices'][0]['delta']['content']:
                            yield chunk['choices'][0]['delta']['content']
                    except json.JSONDecodeError:
                        continue
                        
    except requests.exceptions.RequestException as e:
        st.error(f"Error connecting to Mistral API: {e}")
        yield ""


# --- SIDEBAR ---
with st.sidebar:
    st.markdown("## ✨ Chat Controls")
    st.markdown("---")
    if st.button("🔄 New Chat"):
        st.session_state.messages = [{"role": "system", "content": "You are a helpful and concise assistant."}]
        st.session_state.current_chat = "new_chat"
        st.rerun()

    st.markdown("---")
    st.markdown("## 📜 Chat History")

    # Display saved chats
    history_files = get_chat_history_files()
    for file in history_files:
        col1, col2 = st.columns([3, 1])
        # Display chat name, remove .json
        chat_name_display = file[:-5].replace("_", " ").title()
        
        with col1:
            if st.button(chat_name_display, key=f"load_{file}", use_container_width=True):
                st.session_state.messages = load_chat_history(file)
                st.session_state.current_chat = file
                st.rerun()
        with col2:
            if st.button("🗑️", key=f"delete_{file}"):
                os.remove(os.path.join(HISTORY_DIR, file))
                if st.session_state.current_chat == file:
                     st.session_state.messages = [{"role": "system", "content": "You are a helpful and concise assistant."}]
                     st.session_state.current_chat = "new_chat"
                st.rerun()

    st.markdown("---")
    st.markdown("Built with [Mistral AI](https://mistral.ai) & [Streamlit](https://streamlit.io)")


# --- MAIN CHAT INTERFACE ---
title = st.session_state.current_chat[:-5].replace("_", " ").title() if st.session_state.current_chat != 'new_chat' else 'New Chat'
st.title(f"🤖 {title}")
st.caption("Powered by Mistral AI")

# Display chat history
for msg in st.session_state.messages[1:]:
    avatar = "👤" if msg["role"] == "user" else "🤖"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

# Handle user input
if user_input := st.chat_input("What can I help you with today?"):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_input)

    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Thinking..."):
            response_stream = get_mistral_stream(st.session_state.messages)
            full_response = st.write_stream(response_stream)

    if full_response:
        st.session_state.messages.append({"role": "assistant", "content": full_response})
        # Auto-save the conversation
        if st.session_state.current_chat == "new_chat":
            # Use the first user message to create a filename
            first_user_message = user_input[:30].replace(" ", "_").lower()
            filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{first_user_message}.json"
            st.session_state.current_chat = filename
        
        save_chat_history(st.session_state.messages, st.session_state.current_chat)
        st.rerun() # Rerun to update the history list