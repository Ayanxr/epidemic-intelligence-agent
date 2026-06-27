import os
import json
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate

# 100% local CPU embeddings
from langchain_community.embeddings import FastEmbedEmbeddings

# Chroma config to turn off telemetry tracking completely
from chromadb.config import Settings

load_dotenv()

class EpidemicRAGEngine:
    def __init__(self, db_path="./chroma_db", data_dir="./data"):
        self.db_path = db_path
        self.data_dir = data_dir
        
        # Runs locally on your machine - no Hugging Face API key required
        self.embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
        
        # Turn off anonymized tracking to prevent terminal telemetry crashes
        self.chroma_settings = Settings(
            anonymized_telemetry=False,
            is_persistent=True
        )
        
        self.vector_db = None
        self.llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

    def initialize_global_knowledge(self):
        """Loads PDFs from data folder and builds the initial database safely."""
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            return

        print("🔄 Loading pre-fed guidelines...")
        loader = PyPDFDirectoryLoader(self.data_dir)
        documents = loader.load()
        
        if not documents:
            print("⚠️ No PDFs found in the data/ folder.")
            return

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
        chunks = text_splitter.split_documents(documents)
        
        self.vector_db = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.db_path,
            client_settings=self.chroma_settings
        )
        print("✅ Global Knowledge Base successfully loaded!")

    def load_user_file(self, file_path):
        """Ingests user uploaded PDFs securely."""
        if not self.vector_db:
            self.vector_db = Chroma(
                persist_directory=self.db_path, 
                embedding_function=self.embeddings,
                client_settings=self.chroma_settings
            )
            
        _, ext = os.path.splitext(file_path.lower())
        
        if ext == '.pdf':
            loader = PyPDFLoader(file_path)
            documents = loader.load()
        else:
            raise ValueError("Unsupported file format! Please upload a verified PDF document.")
            
        if not documents:
            raise ValueError("File content could not be parsed or document is empty.")

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=50)
        chunks = text_splitter.split_documents(documents)
        
        self.vector_db.add_documents(chunks)
        print(f"✅ Successfully ingested file chunks from: {os.path.basename(file_path)}")

    def retrieve_context_with_score(self, query):
        """Retrieves documentation and grades alignment via LLM-As-A-Judge."""
        if not self.vector_db:
            self.vector_db = Chroma(
                persist_directory=self.db_path, 
                embedding_function=self.embeddings,
                client_settings=self.chroma_settings
            )
            
        docs = self.vector_db.similarity_search(query, k=4)
        context = "\n\n".join([doc.page_content for doc in docs])
        
        if not context.strip():
            return {"accuracy_score": 0, "is_sufficient": False, "context": ""}

        eval_template = """
        You are a meticulous public health data auditor. Grade whether the context contains enough specific information to address the inquiry.
        
        User Query: {query}
        Context: {context}
        
        Return strictly JSON.
        {{
            "accuracy_score": <integer from 0 to 100>,
            "is_sufficient": <true if score >= 70 else false>,
            "reasoning": "<explanation>"
        }}
        """
        
        prompt = PromptTemplate.from_template(eval_template)
        chain = prompt | self.llm
        
        try:
            response = chain.invoke({"query": query, "context": context})
            evaluation = json.loads(response.content)
            evaluation["context"] = context
            return evaluation
        except Exception:
            return {"accuracy_score": 50, "is_sufficient": False, "context": context}