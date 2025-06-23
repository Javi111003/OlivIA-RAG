import os
import re
import requests
import streamlit as st
from dotenv import load_dotenv
import json
import glob
import asyncio
from datetime import datetime
from generator.llm_provider import MistralLLMProvider
from types import GeneratorType
from  agents.agentic_pipeline import AgenticPipeline
from core.conversation_manager import ConversationManager as cm

# Todo esto es la parte del Scrapy, la dejo arriba para que se vea 
import threading
from scrapy.crawler import CrawlerProcess
from scrapy.utils.project import get_project_settings
from web_crawler.MathCrawlerScraper.spiders.math_spider import MathSpider
from web_crawler.MathCrawlerScraper.spiders.dispatch_spider import DispatchSpider

# --- AQUI ESTA LA MAGIA DEL CRAWLER ---
# --- Se crea un hilo para echar a correr la arania y que esta no moleste el chat ---

#def run_dispatch_spider(query): # Usa esta funcion para llamar a la arana dinamica si no se encuentran resultados
#    """Ejecuta dinámicamente la araña Dispatch con una consulta dada."""
#    process = CrawlerProcess(get_project_settings())
#    process.crawl(DispatchSpider, query=query)
#    process.start()
#
#def run_math_spider(): 
#    process = CrawlerProcess(get_project_settings())
#    process.crawl(MathSpider)
#    process.start()
#
#def start_math_crawler_background(): # Esta funcion se mandara a correr en cuanto se ponga el programa
#    t = threading.Thread(target=run_math_spider, daemon=True)
#    t.start()

# --- CONFIGURATION ---
# Load environment variables from .env file.
load_dotenv()
API_KEY = os.getenv("API_KEY")

#MODEL_NAME = "mistral-small-latest"
#API_URL = "https://api.mistral.ai/v1/chat/completions"
HISTORY_DIR = "chat_history"
# --- PAGE SETUP ---
st.set_page_config(
    page_title="OlivIA - Agentic Math Tutor",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="auto",
)
llm = None 
try:
    llm = MistralLLMProvider()
    pipeline = AgenticPipeline(llm)  # CREAR EL PIPELINE
    st.success("✅ Sistema de agentes inicializado correctamente")
    
except Exception as e:
    st.error(f"Error initializing system: {e}")
    st.stop()

# --- HELPER FUNCTIONS ---

def setup():
    """Initial setup for the app."""
    # Create chat history directory if it doesn't exist
    os.makedirs(HISTORY_DIR, exist_ok=True)
    
    # Initialize session state for messages if it doesn't exist
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "system", "content": "Eres OlívIA, un tutor de matemáticas amigable y experto en preparación para exámenes de ingreso a la universidad. Siempre responde de manera clara, concisa y enfócate en la explicación de conceptos matemáticos."}]
    if "current_chat" not in st.session_state:
        st.session_state.current_chat = "new_chat"
        
async def process_with_agents(user_input: str) -> str:
    """Procesa la consulta del usuario a través del pipeline de agentes"""
    try:
        # Ejecutar el pipeline completo
        respuesta = await pipeline.run(user_input)
        return respuesta
    except Exception as e:
        st.error(f"Error en el pipeline de agentes: {e}")
        return f"Lo siento, ocurrió un error procesando tu consulta: {str(e)}"

def run_agent_pipeline(user_input: str) -> str:
    """Wrapper síncrono para ejecutar el pipeline async"""
    try:
        # Crear un nuevo loop de eventos si no existe
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        # Ejecutar el pipeline
        return loop.run_until_complete(process_with_agents(user_input))
    except Exception as e:
        return f"Error ejecutando pipeline: {str(e)}"

def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


try:
    load_css(os.path.join(os.path.dirname(__file__), "style.css"))
except:
    pass
# Run setup
setup()
#start_math_crawler_background()

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
    history_files = cm.get_chat_history_files()
    for file in history_files:
        col1, col2 = st.columns([3, 1])
        # Display chat name, remove .json
        chat_name_display = file[:-5].replace("_", " ").title()
        
        with col1:
            if st.button(chat_name_display, key=f"load_{file}", use_container_width=True):
                st.session_state.messages = cm.load_chat_history(file)
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

# Mostrar información del sistema
with st.expander("ℹ️ Información del Sistema"):
    st.markdown("""
    **OlivIA** utiliza un sistema de agentes especializados:
    
    1. 🔍 **Retriever**: Busca información relevante en la base de conocimientos
    2. 👨‍💼 **Supervisor**: Coordina y decide qué agente usar
    3. 🧮 **Experto Matemático**: Especializado en explicaciones matemáticas
    4. 📋 **Creador de Exámenes**: Genera preguntas y exámenes personalizados
    5. 📊 **Evaluador**: Verifica la calidad de las respuestas
    
    Tu consulta pasará por este pipeline completo para obtener la mejor respuesta posible.
    """)

# Display chat history
for msg in st.session_state.messages[1:]:
    avatar = "👤" if msg["role"] == "user" else "🤖"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

# Handle user input
if user_input := st.chat_input("¿Qué puedo ayudarte a entender hoy?"):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_input)

    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Thinking..."):
            #response_stream = llm.chat_completion(st.session_state.messages, stream=True)
            full_response = run_agent_pipeline(user_input)
        st.markdown(full_response)

    if full_response:
        st.session_state.messages.append({"role": "assistant", "content": full_response})
        # Auto-save the conversation
        if st.session_state.current_chat == "new_chat":
            # Use the first user message to create a filename
            first_user_message = user_input[:30].replace(" ", "_").lower()
            filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{first_user_message}.json"
            st.session_state.current_chat = filename
        
        cm.save_chat_history(st.session_state.messages, st.session_state.current_chat)
        st.rerun() # Rerun to update the history list