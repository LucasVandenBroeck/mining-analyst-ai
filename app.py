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

# --- ROBUST JSON EXTRACTOR (The Fix) ---
def extract_json_from_text(text):
    """
    Aggressively hunts for JSON objects { ... } inside any text.
    Ignores markdown fences and conversational filler.
    """
    try:
        # 1. If strict markdown exists, take it
        match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
        if match:
            text = match.group(1)

        # 2. Find the outermost curly braces
        start = text.find('{')
        end = text.rfind('}')
        
        if start != -1 and end != -1:
            json_str = text[start:end+1]
            return json.loads(json_str)
        else:
            return {} # Return empty dict if no braces found
            
    except json.JSONDecodeError:
        return {} # Fail silently but safely

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

                parser = LlamaParse(result_type="markdown", verbose=True, language="en")
                file_extractor = {".pdf": parser}
                documents = SimpleDirectoryReader(input_files=[tmp_file_path], file_extractor=file_extractor).load_data()
                
                index = VectorStoreIndex.from_documents(documents)
                st.session_state.index = index
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

    # --- PROMPTS ---
    # Note: We now explicitly ask for nulls to ensure keys exist even if data is missing
    
    prompt_general = """
    Extract General Info. Return ONLY valid JSON:
    { "proj": "Project Name", "co": "Company", "loc": "Location", "dep_type": "Deposit Type", "mining": "Method", "mine_life_yr": null }
    """
    
    prompt_geo = """
    Extract Resource & Reserve Estimates. Return ONLY valid JSON:
    { "res_t": "Total Resource Tonnes", "res_grade": {"Au": null, "Cu": null}, "rev_t": "Total Reserve Tonnes", "rev_grade": {"Au": null} }
    """
    
    prompt_eco = """
    Extract Economic metrics (Base Case). Return ONLY valid JSON:
    { "capex_init": "Initial Capex", "capex_sus": "Sustaining Capex", "opex_mining_t": "Mining cost per tonne", "price": {"Au": null}, "npv": null, "irr": null }
    """

    col1, col2 = st.columns([3, 1])
    
    with col1:
        if st.button("🚀 Run Deep Extraction"):
            st.session_state.messages.append({"role": "user", "content": "Running Deep Extraction..."})
            with st.chat_message("user"):
                st.write("Running Deep Extraction...")

            # Use a fresh engine for extraction
            extraction_engine = st.session_state.index.as_query_engine(
                similarity_top_k=15, 
                response_mode="tree_summarize"
            )

            final_data = {}

            with st.chat_message("assistant"):
                
                # --- STEP 1 ---
                with st.status("1/3 Extracting General Info...", expanded=False) as status:
                    res1 = extraction_engine.query(prompt_general)
                    data1 = extract_json_from_text(str(res1))
                    if data1:
                        final_data.update(data1)
                        status.write("✅ Found General Info")
                    else:
                        status.write("⚠️ Failed to parse General Info")
                        status.write(str(res1)) # Debug output

                # --- STEP 2 ---
                with st.status("2/3 Extracting Geology...", expanded=False) as status:
                    res2 = extraction_engine.query(prompt_geo)
                    data2 = extract_json_from_text(str(res2))
                    if data2:
                        final_data.update(data2)
                        status.write("✅ Found Geology")
                    else:
                        status.write("⚠️ Failed to parse Geology")
                        status.write(str(res2))

                # --- STEP 3 ---
                with st.status("3/3 Extracting Economics...", expanded=False) as status:
                    res3 = extraction_engine.query(prompt_eco)
                    data3 = extract_json_from_text(str(res3))
                    if data3:
                        final_data.update(data3)
                        status.write("✅ Found Economics")
                    else:
                        status.write("⚠️ Failed to parse Economics")
                        status.write(str(res3))

                # MERGE & DISPLAY
                if final_data:
                    # Normalize flattens the nested dictionaries (e.g. price.Au)
                    df = pd.json_normalize([final_data])
                    st.session_state.last_data = df
                    st.success("Deep Extraction Complete!")
                    st.dataframe(df)
                    st.session_state.messages.append({"role": "assistant", "content": "Extraction Complete."})
                else:
                    st.error("No data could be extracted. Check the raw logs above.")

    # Download
    if st.session_state.last_data is not None:
        st.divider()
        st.subheader("📥 Export")
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            st.session_state.last_data.to_excel(writer, index=False, sheet_name='Summary')
        st.download_button("Download Excel (.xlsx)", output.getvalue(), "mining_data_export.xlsx")