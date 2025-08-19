import streamlit as st

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.messages import AIMessage, HumanMessage
from langchain_google_genai import (ChatGoogleGenerativeAI,
                                    GoogleGenerativeAIEmbeddings)
from utilities.rag import RAG
from utilities.indexing import Index
rag = RAG()


def show_rag_chatbot():
    st.title("🤖 RAG Chatbot (LangChain v0.3)")
    
    # File upload
    uploaded_file = st.file_uploader("Upload dokumen", type=["txt", "pdf"])
    
    if uploaded_file:
        # Process document
        # text = uploaded_file.read().decode()
        index = Index(uploaded_file)
        index.chunk_text()
        index.save_index()
        
        # Chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []
            
        # Tampilkan history
        for msg in st.session_state.messages:
            st.chat_message(msg["role"]).write(msg["content"])
            
        # Input user
        if prompt := st.chat_input("Apa pertanyaan Anda?"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.chat_message("user").write(prompt)
            
            response = rag.graph.invoke({"question": prompt})
            
            chat_history = []
            for msg in st.session_state.messages[:-1]:
                if msg["role"] == "user":
                    chat_history.append(HumanMessage(content=msg["content"]))
                else:
                    chat_history.append(AIMessage(content=msg["content"]))
            
            # Tampilkan response
            with st.chat_message("assistant"):
                st.write(response["answer"])
            
            st.session_state.messages.append({"role": "assistant", "content": response["answer"]})