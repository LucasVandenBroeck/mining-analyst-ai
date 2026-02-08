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
    st.info("Tip: GPT-4o is recommended for complex table reasoning.")

# --- HELPER FUNCTION: EXTRACT JSON ---
def extract_json(text):
    """
    Tries to find JSON data inside the AI's response.
    Returns a pandas DataFrame or None if it fails.
    """
    try:
        # Find JSON block between triple backticks if present
        match = re.search(r"```json\n(.*?)\n```", text, re.DOTALL)
        if match:
            json_str = match.group(1)
        else:
            # If no backticks, try to parse the whole text
            json_str = text
            
        data = json.loads(json_str)
        
        # If it's a list of dictionaries (standard table format)
        if isinstance(data, list):
            return pd.DataFrame(data)
        # If it's a single dictionary
        elif isinstance(data, dict):
            return pd.DataFrame([data])
        else:
            return None
    except Exception as e:
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

if "query_engine" not in st.session_state:
    st.session_state.query_engine = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_data" not in st.session_state:
    st.session_state.last_data = None # Store the last extracted table

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

                mining_prompt = (
                    "You are a Mining Analyst. "
                    "If asked to 'extract data for Excel', output ONLY valid JSON without extra text. "
                    "Format the JSON as a list of dictionaries, where each dictionary is a row."
                )
                
                st.session_state.query_engine = index.as_query_engine(
                    system_prompt=mining_prompt,
                    similarity_top_k=5
                )
                
                st.session_state.last_uploaded = uploaded_file.name
                st.success("Analysis Ready!")
                os.remove(tmp_file_path)

            except Exception as e:
                st.error(f"Error: {e}")

# --- CHAT & EXPORT INTERFACE ---
if st.session_state.query_engine:

    # 1. Display History
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 2. Input Area
    col1, col2 = st.columns([3, 1])
    
    # Preset Buttons
    with col1:
        st.write("Targeted Extraction:")
        btn_eco = st.button("📊 Extract Economics to Excel")
        btn_drill = st.button("💎 Extract Drill Results to Excel")
    
    # Chat Input
    prompt = st.chat_input("Ask a question...")

    # Logic
    search_query = None
    is_export_task = False

    if btn_eco:
        search_query = (
            "Extract the Project Economics into a JSON table. "
            "Columns: Metric (e.g., NPV 5%, IRR, Initial Capex, AISC), Unit, Value, Notes. "
            "Do not add conversational text, just the JSON."
        )
        is_export_task = True
    elif btn_drill:
        search_query = (
            "Extract the top 10 drill intercepts into a JSON table. "
            "Columns: Hole_ID, From_m, To_m, Interval_m, Grade_Au_gpt, Grade_Ag_gpt. "
            "Do not add conversational text, just the JSON."
        )
        is_export_task = True
    elif prompt:
        search_query = prompt

    # Execution
    if search_query:
        # Append user message to history
        st.session_state.messages.append({"role": "user", "content": search_query})
        with st.chat_message("user"):
            st.markdown(search_query)

        # Generate Response
        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                response = st.session_state.query_engine.query(search_query)
                response_text = str(response)
                
                # Check if this was an export task
                if is_export_task:
                    df = extract_json(response_text)
                    if df is not None:
                        st.session_state.last_data = df # Save for download button
                        st.dataframe(df) # Show preview
                        st.success("Data extracted! Use the download button below.")
                    else:
                        st.warning("Could not format data as JSON. Showing raw text instead.")
                        st.markdown(response_text)
                else:
                    st.markdown(response_text)
                
                st.session_state.messages.append({"role": "assistant", "content": response_text})

    # 3. Persistent Download Button
    # This button appears whenever there is valid data in the 'buffer'
    if st.session_state.last_data is not None:
        st.divider()
        st.subheader("📥 Download Area")
        
        # Convert DataFrame to Excel in memory
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            st.session_state.last_data.to_excel(writer, index=False, sheet_name='Mining_Data')
        excel_data = output.getvalue()

        st.download_button(
            label="Download Data as Excel (.xlsx)",
            data=excel_data,
            file_name="mining_analysis.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )