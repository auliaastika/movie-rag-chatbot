import asyncio

from langgraph.graph import START, StateGraph
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_google_genai import (ChatGoogleGenerativeAI,
                                    GoogleGenerativeAIEmbeddings)
from langchain import hub

# Initialize State

from langchain_core.documents import Document
from typing_extensions import List, TypedDict
from langchain_groq import ChatGroq


class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

class RAG:
    def __init__(self):

        # Pastikan ada event loop
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            asyncio.set_event_loop(asyncio.new_event_loop())


        self.graph_builder = StateGraph(State).add_sequence([self.retrieve, self.generate])
        self.graph_builder.add_edge(START, "retrieve")
        self.graph = self.graph_builder.compile()
        
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
        self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

        self.prompt = hub.pull("rlm/rag-prompt")
        self.vector_store = None
        
    def load_vector_store(self):
        self.vector_store = FAISS.load_local(
            "faiss_index", self.embeddings, allow_dangerous_deserialization=True
        )

    def retrieve(self, state: State):
        if self.vector_store is None:
            self.load_vector_store()
        retrieved_docs = self.vector_store.similarity_search(state["question"])
        return {"context": retrieved_docs}

    def generate(self, state: State):
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])
        messages = self.prompt.invoke({"question": state["question"], "context": docs_content})
        response = self.llm.invoke(messages)
        return {"answer": response.content}
