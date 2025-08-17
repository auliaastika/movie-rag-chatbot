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


def load_css():
    st.markdown("""
        <style>
        /* Background */
        .stApp {
            background-color: #1e1e1e;
            color: #f5f5f5;
            font-family: 'Segoe UI', sans-serif;
        }

        /* Title */
        h1 {
            font-size: 2.5rem;
            color: #FF4B4B;
            text-shadow: 2px 2px 5px rgba(0,0,0,0.4);
        }
        h3 {
            color: #aaa;
        }

        /* Chat bubble */
        .stChatMessage {
            border-radius: 15px;
            padding: 12px;
            margin: 6px 0px;
        }
        .stChatMessage[data-testid="user"] {
            background-color: #2c2f33;
            color: #fff;
        }
        .stChatMessage[data-testid="assistant"] {
            background-color: #FF4B4B20;
            border: 1px solid #FF4B4B60;
            color: #fff;
        }

        /* Sidebar */
        [data-testid="stSidebar"] {
            background-color: #252526;
            border-right: 1px solid #333;
        }

        /* Buttons & uploader */
        .stFileUploader label {
            color: #FF4B4B !important;
            font-weight: bold;
        }
        button, .stDownloadButton {
            border-radius: 12px !important;
            background: linear-gradient(45deg, #FF4B4B, #FF6A6A);
            color: white !important;
            font-weight: bold;
            border: none;
        }
        </style>
    """, unsafe_allow_html=True)


def show_movie_rag_chatbot():
    load_css()

    st.markdown(
        """
        <div style="text-align:center;">
            <h1>🎬 Movie RAG Chatbot</h1>
            <h3>by <b>Aulia Astika</b></h3>
        </div>
        <hr style="margin:10px 0px 25px 0px; border: 1px solid #333;">
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        🌟 **Welcome to the Movie RAG Chatbot!** 🎥  
        Upload your dataset (**TXT / CSV / PDF**) and ask anything about your favorite movies!  
        This chatbot uses **LangChain + RAG** + **Google Generative AI** 🚀  
        """,
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader(
        "📂 Upload your movie dataset:",
        type=["txt", "csv", "pdf"]
    )
    
    if uploaded_file:
        index = Index(uploaded_file)
        index.chunk_text()
        index.save_index()

        if "messages" not in st.session_state:
            st.session_state.messages = []

        for msg in st.session_state.messages:
            st.chat_message(msg["role"]).write(msg["content"])

        if prompt := st.chat_input("💬 Ask me anything about the movies..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.chat_message("user").write(prompt)

            response = rag.graph.invoke({"question": prompt})

            chat_history = []
            for msg in st.session_state.messages[:-1]:
                if msg["role"] == "user":
                    chat_history.append(HumanMessage(content=msg["content"]))
                else:
                    chat_history.append(AIMessage(content=msg["content"]))

            with st.chat_message("assistant"):
                st.write(response["answer"])

            st.session_state.messages.append(
                {"role": "assistant", "content": response["answer"]}
            )

    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/744/744922.png", width=100)
        st.markdown("## 🎞 About this Chatbot")
        st.markdown(
            """
            Explore your **movies dataset** with AI-powered Q&A.  

            **Tech Stack**:  
            - ⚡ Streamlit  
            - 🔍 LangChain + RAG  
            - 🤖 Google Generative AI  
            """
        )
        st.markdown("---")
        st.markdown("👩‍💻 **Created by Aulia Astika**")
