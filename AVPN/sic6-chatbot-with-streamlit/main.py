from dotenv import load_dotenv
import os, sys
import asyncio

# Patch untuk streamlit supaya ada event loop
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

load_dotenv()  # load .env

import streamlit as st
from RAG_Chatbot import show_rag_chatbot
from RAG_MovieChatbot import show_movie_rag_chatbot  # <-- chatbot movie baru

def main():
    st.sidebar.title("⚡ Pilih Chatbot")
    menu = st.sidebar.selectbox(
        "Pilih mode:",
        ["General RAG Chatbot", "Movie RAG Chatbot"]
    )

    if menu == "General RAG Chatbot":
        show_rag_chatbot()
    elif menu == "Movie RAG Chatbot":
        show_movie_rag_chatbot()

if __name__ == "__main__":
    main()
