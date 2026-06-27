import streamlit as st
import os
from src.rag_engine import EpidemicRAGEngine
from src.tools import PublicHealthSearchTool

st.set_page_config(
    page_title="Episurv AI: Epidemic Intelligence Agent",
    page_icon="🦠",
    layout="wide"
)

@st.cache_resource
def get_systems():
    engine = EpidemicRAGEngine()
    engine.initialize_global_knowledge()
    search_tool = PublicHealthSearchTool()
    return engine, search_tool

try:
    rag_engine, web_tool = get_systems()
except Exception as e:
    st.error(f"Failed to initialize systems. Check your GROQ_API_KEY. Error: {e}")

# --- UI HEADER ---
st.title("🦠 Episurv AI: Epidemic Intelligence Agent")
st.markdown("""
    This Agentic RAG system acts as an automated public health auditor. It evaluates queries against internal CDC/WHO manuals, 
    calculates an **Accuracy Score**, and autonomously triggers a targeted medical web-search fallback if internal data is insufficient.
""")

st.divider()

# --- SIDEBAR: FILE UPLOAD ---
with st.sidebar:
    st.header("📥 Local Incident Reports")
    st.write("Upload a situational clinic report or health dataset (PDF) to append to the active vector database.")
    
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])
    
    if uploaded_file is not None:
        with st.spinner("Processing and embedding user document..."):
            temp_dir = "./temp_uploads"
            if not os.path.exists(temp_dir):
                os.makedirs(temp_dir)
                
            tmp_filepath = os.path.join(temp_dir, uploaded_file.name)
            
            with open(tmp_filepath, "wb") as f:
                f.write(uploaded_file.getvalue())
            
            try:
                rag_engine.load_user_file(tmp_filepath)
                st.success(f"Successfully ingested: {uploaded_file.name}!")
            except Exception as e:
                st.error(f"Error parsing file: {e}")
            finally:
                if os.path.exists(tmp_filepath):
                    os.remove(tmp_filepath)
                    
# --- MAIN INTERFACE: QUERY ENGINE ---
user_query = st.text_input(
    "🔍 Enter tactical query or outbreak symptoms:",
    placeholder="e.g., What are the isolation rules for a sudden spike in Dengue with plasma leakage?"
)

if user_query:
    with st.spinner("Agent running evaluation routing loops..."):
        evaluation = rag_engine.retrieve_context_with_score(user_query)
        
        score = evaluation.get("accuracy_score", 0)
        is_sufficient = evaluation.get("is_sufficient", False)
        reasoning = evaluation.get("reasoning", "No rationale provided.")
        pdf_context = evaluation.get("context", "")

        col1, col2 = st.columns([1, 3])
        with col1:
            if is_sufficient:
                st.metric(label="RAG Accuracy Score", value=f"{score}%", delta="Sufficient (Internal Data)")
            else:
                st.metric(label="RAG Accuracy Score", value=f"{score}%", delta="- Insufficient (Fallback Triggered)", delta_color="inverse")
        with col2:
            st.info(f"**Auditor Reasoning:** {reasoning}")

        if is_sufficient:
            st.subheader("🤖 Agent Strategy: Pure Internal Retrieval")
            final_context = pdf_context
        else:
            st.subheader("🌐 Agent Strategy: Triggering Web-Search Tool (Self-Correction)")
            web_results = web_tool.search_verified_medical_web(user_query)
            
            with st.expander("See Raw Web Search Tool Output"):
                st.write(web_results)
                
            final_context = f"INTERNAL MANUAL CONTEXT:\n{pdf_context}\n\nLIVE VERIFIED WEB CONTEXT:\n{web_results}"

        generation_prompt = f"""
        You are an expert epidemic response strategist. Synthesize an actionable directive for health personnel using the following context.
        Ensure you outline symptom tracking criteria, isolation steps, or administrative instructions explicitly present in the data.
        
        Context Source Layer:
        {final_context}
        
        User Query: {user_query}
        
        Provide a clean, bulleted operational roadmap. If any data is unavailable, state it clearly.
        """
        
        final_response = rag_engine.llm.invoke(generation_prompt)
        st.subheader("📋 Actionable Public Health Directive")
        st.markdown(final_response.content)
        
        with st.expander("🔍 View Raw Vector DB Retrieved Chunks"):
            st.text(pdf_context if pdf_context else "No vector chunks found.")