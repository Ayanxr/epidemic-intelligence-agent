import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

# Load environment variables from .env
load_dotenv()

class EpidemicRAGEngine:
    def __init__(self, db_path="./chroma_db", data_dir="./data"):
        self.db_path = db_path
        self.data_dir = data_dir
        
        # 1. Initialize Open-Source, Free Embeddings (FastEmbed runs locally without an API key)
        self.embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
        
        # 2. Initialize our Vector DB instance
        self.vector_db = None
        
        # 3. Initialize the Groq LLM for evaluation and generation
        # Using llama-3.1-8b-instant as it's fast and highly capable of structured JSON output
        self.llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

    def initialize_global_knowledge(self):
        """Loads pre-fed CDC/WHO PDFs from data folder and builds the initial database."""
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            print(f"Created {self.data_dir} directory. Please ensure your pre-fed PDFs are dropped here.")
            return

        print("🔄 Loading pre-fed CDC/WHO public health guidelines...")
        loader = PyPDFDirectoryLoader(self.data_dir)
        documents = loader.load()
        
        if not documents:
            print("⚠️ No PDFs found in the data/ folder. Skipping global ingestion.")
            return

        # Split texts into chunks optimized for semantic medical match retrieval
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
        chunks = text_splitter.split_documents(documents)
        
        print(f"🧩 Split guidelines into {len(chunks)} chunks. Generating embeddings...")
        
        # Create and persist the vector store
        self.vector_db = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.db_path
        )
        print("✅ Global Knowledge Base successfully loaded and stored in Vector DB!")

    def load_user_pdf(self, file_path):
        """Dynamically ingests a local incident report (PDF or TXT) uploaded by a user."""
        if not self.vector_db:
            self.vector_db = Chroma(persist_directory=self.db_path, embedding_function=self.embeddings)
            
        print(f"📥 Processing user-uploaded report: {file_path}")
        
        # Robust fallback checking for file extension types
        if file_path.endswith('.txt') or file_path.endswith('.pdf') and os.path.getsize(file_path) < 2000:
            # If a PDF is tiny or it's a text file, parse it safely as raw string data
            from langchain_core.documents import Document
            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()
                documents = [Document(page_content=text, metadata={"source": os.path.basename(file_path)})]
            except Exception:
                from langchain_community.document_loaders import PyPDFLoader
                loader = PyPDFLoader(file_path)
                documents = loader.load()
        else:
            from langchain_community.document_loaders import PyPDFLoader
            loader = PyPDFLoader(file_path)
            documents = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=50)
        chunks = text_splitter.split_documents(documents)
        
        # Tag chunks as 'user_upload' via metadata to keep track of sources
        for chunk in chunks:
            chunk.metadata["source"] = "user_upload"
            
        self.vector_db.add_documents(chunks)
        print(f"✅ User report ingested successfully into {len(chunks)} searchable chunks.")
        
    def retrieve_context_with_score(self, query):
        """Retrieves documents, grades their alignment to the query, and generates an accuracy score."""
        if not self.vector_db:
            self.vector_db = Chroma(persist_directory=self.db_path, embedding_function=self.embeddings)
            
        # Fetch the top 4 most relevant chunks
        docs = self.vector_db.similarity_search(query, k=4)
        context = "\n\n".join([doc.page_content for doc in docs])
        
        if not context.strip():
            return {"accuracy_score": 0, "is_sufficient": False, "context": ""}

        # The Self-Correction Judge Prompt
        eval_template = """
        You are a meticulous public health data auditor. Your job is to strictly grade whether the retrieved internal medical guidelines contain enough specific clinical information to completely address the user's inquiry.
        
        User Query: {query}
        
        Retrieved Internal Guidelines:
        {context}
        
        Evaluate carefully. If specific operational steps, treatment paths, or symptom identifiers requested are missing, lower the score.
        You must return your output strictly in JSON format with no extra text or explanations.
        
        Expected JSON Format:
        {{
            "accuracy_score": <integer from 0 to 100>,
            "is_sufficient": <true if score >= 70 else false>,
            "reasoning": "<brief sentence explanation of your scoring choice>"
        }}
        """
        
        prompt = PromptTemplate.from_template(eval_template)
        chain = prompt | self.llm
        
        try:
            # Enforce structured JSON generation
            response = chain.invoke({"query": query, "context": context})
            import json
            evaluation = json.loads(response.content)
            evaluation["context"] = context # Attach raw text for downstream steps
            return evaluation
        except Exception as e:
            print(f"Error during self-correction evaluation step: {e}")
            # Safe default fallback if JSON parsing fails
            return {"accuracy_score": 50, "is_sufficient": False, "context": context}

# Quick validation routine when executing the script directly
if __name__ == "__main__":
    engine = EpidemicRAGEngine()
    engine.initialize_global_knowledge()