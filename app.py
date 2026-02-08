import streamlit as st
import os
import tempfile
import pandas as pd
import json
import re
from io import BytesIO
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.openai import OpenAI
from llama_parse import LlamaParse

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Mining Analyst AI", layout="wide")
st.title("⛏️ Mining Analyst AI")

# --- SIDEBAR ---
with st.sidebar:
    st.header("Configuration")
    
    default_openai = os.environ.get("OPENAI_API_KEY", "")
    default_llama = os.environ.get("LLAMA_CLOUD_API_KEY", "")

    openai_key = st.text_input("OpenAI API Key", value=default_openai, type="password")
    llama_cloud_key = st.text_input("LlamaCloud API Key", value=default_llama, type="password")
    
    st.divider()
    
    model_choice = st.selectbox("Select Model", ["gpt-4o", "gpt-4o-mini"])
    st.info("Tip: GPT-4o is CRITICAL for this multi-step reasoning.")

# --- HELPER: MERGE JSON ---
def extract_json_from_text(text):
    """Extracts JSON block from text."""
    match = re.search(r"```json\n(.*?)\n```", text, re.DOTALL)
    json_str = match.group(1) if match else text
    try:
        return json.loads(json_str)
    except:
        return {}

# --- MAIN LOGIC ---

if not openai_key or not llama_cloud_key:
    st.warning("Please enter your API Keys in the sidebar to proceed.")
    st.stop()

os.environ["OPENAI_API_KEY"] = openai_key
os.environ["LLAMA_CLOUD_API_KEY"] = llama_cloud_key

llm = OpenAI(model=model_choice, temperature=0)
Settings.llm = llm

# --- FILE UPLOAD ---
uploaded_file = st.file_uploader("Upload Mining Report (PDF)", type=['pdf'])

if "index" not in st.session_state:
    st.session_state.index = None
if "query_engine" not in st.session_state:
    st.session_state.query_engine = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_data" not in st.session_state:
    st.session_state.last_data = None

if uploaded_file:
    if "last_uploaded" not in st.session_state or st.session_state.last_uploaded != uploaded_file.name:
        with st.spinner(f"Parsing {uploaded_file.name}..."):
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_file_path = tmp_file.name

                # LlamaParse with improved instruction
                parser = LlamaParse(
                    result_type="markdown", 
                    verbose=True, 
                    language="en",
                    parsing_instruction="This is a mining report. Extract tables containing Drill Results, Reserves, and Economics precisely."
                )
                file_extractor = {".pdf": parser}
                documents = SimpleDirectoryReader(input_files=[tmp_file_path], file_extractor=file_extractor).load_data()
                
                index = VectorStoreIndex.from_documents(documents)
                st.session_state.index = index
                
                # Default Chat Engine
                st.session_state.query_engine = index.as_query_engine(similarity_top_k=5)
                
                st.session_state.last_uploaded = uploaded_file.name
                st.success("Analysis Ready!")
                os.remove(tmp_file_path)

            except Exception as e:
                st.error(f"Error: {e}")

# --- INTERFACE ---
if st.session_state.index:

    # History
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # --- DEFINING THE 3 TARGETED SCHEMAS ---
    
    # 1. General Info
    prompt_general = """
    Extract General Project Info. Return JSON:
    { "proj": "Project Name", "co": "Company", "loc": "Location", "dep_type": "Deposit Type", "mining": "Method", "mine_life_yr": null }
    """
    
    # 2. Resources (The hard part)
    prompt_geo = """
    Extract Resource & Reserve Estimates. Return JSON:
    { "res_t": "Total Resource Tonnes", "res_grade": {"Au": null, "Cu": null}, "rev_t": "Total Reserve Tonnes", "rev_grade": {"Au": null} }
    """
    
    # 3. Economics
    prompt_eco = """
    Extract Economic metrics (Base Case). Return JSON:
    { "capex_init": "Initial Capex", "capex_sus": "Sustaining Capex", "opex_mining_t": "Mining cost per tonne", "price": {"Au": null}, "npv": null, "irr": null }
    """

    col1, col2 = st.columns([3, 1])
    
    with col1:
        if st.button("🚀 Run Deep Extraction (Divide & Conquer)"):
            st.session_state.messages.append({"role": "user", "content": "Running Deep Extraction..."})
            with st.chat_message("user"):
                st.write("Running Deep Extraction...")

            # Create specific engine
            extraction_engine = st.session_state.index.as_query_engine(
                similarity_top_k=10, 
                response_mode="tree_summarize"
            )

            final_data = {}

            with st.chat_message("assistant"):
                # STEP 1: General
                with st.spinner("1/3 Extracting General Info..."):
                    res1 = extraction_engine.query(prompt_general)
                    final_data.update(extract_json_from_text(str(res1)))
                
                # STEP 2: Geology
                with st.spinner("2/3 Extracting Resources & Reserves..."):
                    res2 = extraction_engine.query(prompt_geo)
                    final_data.update(extract_json_from_text(str(res2)))
                
                # STEP 3: Economics
                with st.spinner("3/3 Extracting Economics..."):
                    res3 = extraction_engine.query(prompt_eco)
                    final_data.update(extract_json_from_text(str(res3)))

                # MERGE & DISPLAY
                try:
                    df = pd.json_normalize([final_data])
                    st.session_state.last_data = df
                    st.success("Deep Extraction Complete!")
                    st.dataframe(df)
                    st.session_state.messages.append({"role": "assistant", "content": "Deep Extraction Complete."})
                except Exception as e:
                    st.error(f"Merge Error: {e}")
                    st.write(final_data)

    # Chat Input
    if prompt := st.chat_input("Ask specific questions..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.query_engine.query(prompt)
                st.markdown(str(response))
                st.session_state.messages.append({"role": "assistant", "content": str(response)})

    # Download
    if st.session_state.last_data is not None:
        st.divider()
        st.subheader("📥 Export")
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            st.session_state.last_data.to_excel(writer, index=False, sheet_name='Summary')
        st.download_button("Download Excel (.xlsx)", output.getvalue(), "mining_data_export.xlsx")