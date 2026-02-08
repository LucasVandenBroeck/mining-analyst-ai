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
    st.info("Tip: GPT-4o is highly recommended for structured JSON extraction.")

# --- HELPER FUNCTION: EXTRACT & FLATTEN JSON ---
def extract_and_normalize_json(text):
    """
    Finds JSON in text, loads it, and flattens nested dictionaries 
    so they fit nicely into Excel columns.
    """
    try:
        # 1. Regex to find the JSON block
        match = re.search(r"```json\n(.*?)\n```", text, re.DOTALL)
        json_str = match.group(1) if match else text
        
        # 2. Parse JSON
        data = json.loads(json_str)
        
        # 3. Ensure it's a list (even if one item) for normalization
        if isinstance(data, dict):
            data = [data]
            
        # 4. Normalize (Flatten nested keys like 'price.Au')
        df = pd.json_normalize(data)
        return df
        
    except Exception as e:
        print(f"JSON Error: {e}")
        return None

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

# Session State Initialization
if "query_engine" not in st.session_state:
    st.session_state.query_engine = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_data" not in st.session_state:
    st.session_state.last_data = None

if uploaded_file:
    # Check if new file
    if "last_uploaded" not in st.session_state or st.session_state.last_uploaded != uploaded_file.name:
        
        with st.spinner(f"Parsing {uploaded_file.name}... (This happens only once per file)"):
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_file_path = tmp_file.name

                parser = LlamaParse(result_type="markdown", verbose=True, language="en")
                file_extractor = {".pdf": parser}
                documents = SimpleDirectoryReader(input_files=[tmp_file_path], file_extractor=file_extractor).load_data()
                index = VectorStoreIndex.from_documents(documents)

                # Base Prompt
                base_prompt = (
                    "You are a Mining Analyst. Your goal is to extract structured data. "
                    "Always answer based strictly on the provided text."
                )
                
                st.session_state.query_engine = index.as_query_engine(
                    system_prompt=base_prompt,
                    similarity_top_k=7  # Increased context window for full report scanning
                )
                
                st.session_state.last_uploaded = uploaded_file.name
                st.success("Analysis Ready!")
                os.remove(tmp_file_path)

            except Exception as e:
                st.error(f"Error: {e}")

# --- CHAT & EXPORT INTERFACE ---
if st.session_state.query_engine:

    # Display History
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # --- THE GOLDEN PROMPT TEMPLATE ---
    # We define the schema here to inject into the query
    json_structure_prompt = """
    Extract the project data into the following JSON schema.
    Return ONLY valid JSON inside ```json``` tags.
    If a value is not found, use null.
    
    Mapping Guide:
    - res_t: Resource Tonnage (Total M&I preferred)
    - res_grade: Resource Grades (e.g., {"Au": 1.5, "Ag": 20})
    - rev_t: Reserve Tonnage (Proven & Probable)
    - mining: Mining Method (Open Pit, Underground, etc.)
    - rec: Recovery Rates % (e.g., {"Au": 92.5})
    - prod_pa: Production Per Annum (Avg LOM)
    - capex: In Millions (initial and sustaining)
    - opex: Per tonne processed
    - price: Metal prices used in economic model
    
    Target JSON Structure:
    {
      "proj": "Project Name",
      "co": "Company Name",
      "loc": "Country/Location",
      "dep_type": "Deposit Type (e.g. Porphyry)",
      "res_t": null,
      "res_grade": {},
      "res_contained": {},
      "rev_t": null,
      "rev_grade": {},
      "rev_contained": {},
      "cutoff": null,
      "mining": "",
      "strip": null,
      "tpd": null,
      "tpy": null,
      "rec": {},
      "proc_method": "",
      "prod_pa": {},
      "capex_init": null,
      "capex_sus": null,
      "opex_mining_t": null,
      "opex_proc_t": null,
      "opex_ga_t": null,
      "price": {},
      "discount": null,
      "mine_life_yr": null
    }
    """

    # Input Area
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # The Main Extraction Button
        if st.button("🚀 Run Full Data Extraction"):
            # 1. VISUAL FEEDBACK
            st.session_state.messages.append({"role": "user", "content": "Running Full Data Extraction..."})
            with st.chat_message("user"):
                st.write("Running Full Data Extraction...")

            # 2. THE FIX: Create a specialized engine just for this heavy task
            # We increase 'similarity_top_k' to 20 to catch distributed data
            # We use 'response_mode="tree_summarize"' to force it to combine all 20 chunks intelligently
            extraction_engine = st.session_state.query_engine.index.as_query_engine(
                system_prompt=json_structure_prompt,
                similarity_top_k=20,     # <--- INCREASED from 7 to 20
                response_mode="tree_summarize" # <--- NEW: Combines multiple chunks better
            )

            # 3. GENERATE
            with st.chat_message("assistant"):
                with st.spinner("Scanning 20+ key sections of the report... (This may take 30s)"):
                    
                    # We send a very specific trigger phrase to matched the system prompt
                    response = extraction_engine.query("Extract all project parameters into the defined JSON schema.")
                    response_text = str(response)
                    
                    # 4. PARSE & DISPLAY
                    df = extract_and_normalize_json(response_text)
                    
                    if df is not None:
                        st.session_state.last_data = df
                        st.success("Extraction Complete! (Checked 20 document segments)")
                        st.dataframe(df)
                        
                        # Add success message to chat history
                        st.session_state.messages.append({"role": "assistant", "content": "Data Extracted Successfully. See the Download section below."})
                    else:
                        st.error("Extraction failed or returned invalid JSON.")
                        with st.expander("See Raw Output"):
                            st.text(response_text)

    # Chat Input (for other questions)
    if prompt := st.chat_input("Ask specific questions (e.g. 'What is the strip ratio?')"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.query_engine.query(prompt)
                st.markdown(str(response))
                st.session_state.messages.append({"role": "assistant", "content": str(response)})

    # Persistent Download Button
    if st.session_state.last_data is not None:
        st.divider()
        st.subheader("📥 Export")
        
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            st.session_state.last_data.to_excel(writer, index=False, sheet_name='Summary')
        excel_data = output.getvalue()

        st.download_button(
            label="Download Excel (.xlsx)",
            data=excel_data,
            file_name="mining_data_export.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )