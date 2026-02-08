import streamlit as st
import os
import tempfile
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.openai import OpenAI
from llama_parse import LlamaParse

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Mining Analyst AI", layout="wide")
st.title("⛏️ Mining Analyst AI (Powered by LlamaIndex)")

# --- SIDEBAR: CONFIGURATION ---
with st.sidebar:
    st.header("Configuration")
    
    # Check for secrets or environment variables first
    default_openai = os.environ.get("OPENAI_API_KEY", "")
    default_llama = os.environ.get("LLAMA_CLOUD_API_KEY", "")

    openai_key = st.text_input("OpenAI API Key", value=default_openai, type="password")
    llama_cloud_key = st.text_input("LlamaCloud API Key", value=default_llama, type="password")
    
    st.divider()
    
    model_choice = st.selectbox("Select Model", ["gpt-4o", "gpt-4o-mini"])
    st.info("Tip: GPT-4o is recommended for complex table reasoning.")

# --- MAIN APP LOGIC ---

if not openai_key or not llama_cloud_key:
    st.warning("Please enter your API Keys in the sidebar to proceed.")
    st.stop()

# Set API Keys for the session
os.environ["OPENAI_API_KEY"] = openai_key
os.environ["LLAMA_CLOUD_API_KEY"] = llama_cloud_key

# Initialize Model
llm = OpenAI(model=model_choice, temperature=0)
Settings.llm = llm

# --- FILE UPLOAD SECTION ---
uploaded_file = st.file_uploader("Upload Mining Report (PDF)", type=['pdf'])

# Initialize session state
if "query_engine" not in st.session_state:
    st.session_state.query_engine = None

if uploaded_file:
    # Check if this is a new file or the same one
    if "last_uploaded" not in st.session_state or st.session_state.last_uploaded != uploaded_file.name:
        
        with st.spinner(f"Parsing {uploaded_file.name}... Tables are being extracted..."):
            try:
                # 1. Save to temp file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_file_path = tmp_file.name

                # 2. LlamaParse Configuration
                parser = LlamaParse(
                    result_type="markdown", 
                    verbose=True,
                    language="en"
                )

                # 3. Load & Parse
                file_extractor = {".pdf": parser}
                documents = SimpleDirectoryReader(
                    input_files=[tmp_file_path], 
                    file_extractor=file_extractor
                ).load_data()

                # 4. Indexing
                index = VectorStoreIndex.from_documents(documents)

                # 5. Create Engine
                mining_prompt = (
                    "You are a Senior Mining Analyst. Your goal is to extract factual data from the report. "
                    "When asked for numbers (NPV, IRR, Grade), format them clearly. "
                    "If a table is provided in the context, preserve its structure in your output."
                )
                
                st.session_state.query_engine = index.as_query_engine(
                    system_prompt=mining_prompt,
                    similarity_top_k=5
                )
                
                st.session_state.last_uploaded = uploaded_file.name
                st.success("Analysis Ready!")
                
                # Cleanup
                os.remove(tmp_file_path)

            except Exception as e:
                st.error(f"An error occurred: {e}")

# --- CHAT INTERFACE ---
if st.session_state.query_engine:
    
    col1, col2, col3 = st.columns(3)
    if col1.button("📊 Extract Economics"):
        prompt = "Create a markdown table summarizing the Project Economics: Post-Tax NPV (5% and 8%), IRR, Initial Capex, and LOM AISC."
    elif col2.button("⚠️ Summarize Risks"):
        prompt = "What are the primary risks? Focus on Permitting, Jurisdiction (Country Risk), and Social License."
    elif col3.button("💎 Drill Results"):
        prompt = "List the top 5 drill intercepts (Grade x Width). Format as a bulleted list."
    else:
        prompt = ""

    # Chat Input
    if user_input := st.chat_input("Ask about the report...") or prompt:
        if prompt and not user_input:
            user_input = prompt
            
        with st.chat_message("user"):
            st.write(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                response = st.session_state.query_engine.query(user_input)
                st.markdown(str(response))