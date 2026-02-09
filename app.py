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
st.set_page_config(page_title="Mining Analyst Pro", layout="wide")
st.title("⛏️ Mining Analyst Pro (Enterprise Extraction)")

# --- SIDEBAR ---
with st.sidebar:
    st.header("Configuration")
    openai_key = st.text_input("OpenAI API Key", type="password")
    llama_cloud_key = st.text_input("LlamaCloud API Key", type="password")
    st.divider()
    model_choice = st.selectbox("Select Model", ["gpt-4o", "gpt-4o-mini"])
    st.info("Tip: GPT-4o is required for this level of depth.")

# --- THE 11-STEP SCHEMA CONFIGURATION ---
# This dictionary controls the entire extraction logic.
# Edit the 'prompt' here to change what the AI looks for.
SCHEMA_CONFIG = {
    "1_Identity": {
        "description": "Project Identity",
        "prompt": "Extract project identity. Return JSON: { 'proj_name': '', 'company': '', 'prop_type': 'Exploration/Dev/Prod', 'loc_country': '', 'loc_province': '', 'dep_type': 'Porphyry/VMS/etc' }"
    },
    "2_Resources": {
        "description": "Mineral Resources (M&I + Inferred)",
        "prompt": "Extract Resource Totals. Return JSON: { 'res_total_tonnes_mi': 'Total M&I', 'res_grade_au_mi': null, 'res_grade_cu_mi': null, 'res_contained_au_oz': null, 'res_category': 'Measured+Indicated' }"
    },
    "3_Mining": {
        "description": "Mining Method & Production",
        "prompt": "Extract Mining Metrics. Return JSON: { 'mining_method': 'Open Pit/UG', 'strip_ratio': null, 'mining_rate_tpd': null, 'lom_years': null, 'avg_prod_au_pa': null }"
    },
    "4_Metallurgy": {
        "description": "Processing & Recovery",
        "prompt": "Extract Metallurgy. Return JSON: { 'process_method': 'CIL/Flotation/Heap', 'recovery_au_pct': null, 'recovery_cu_pct': null, 'throughput_tpd': null, 'reagent_consumption': '' }"
    },
    "5_Costs": {
        "description": "Capex & Opex",
        "prompt": "Extract Costs. Return JSON: { 'capex_initial_m': null, 'capex_sustaining_m': null, 'opex_mining_t': null, 'opex_process_t': null, 'opex_ga_t': null, 'aisc_oz': null }"
    },
    "6_Economics": {
        "description": "Economic Outputs (Base Case)",
        "prompt": "Extract Base Case Economics. Return JSON: { 'price_assumed_au': null, 'discount_rate_pct': null, 'npv_post_tax_m': null, 'irr_post_tax_pct': null, 'payback_years': null }"
    },
    "7_Enviro": {
        "description": "Environmental & Permitting",
        "prompt": "Extract Environmental status. Return JSON: { 'permit_status': 'Approved/Pending', 'tailings_type': 'Dry Stack/Slurry', 'closure_cost_m': null, 'major_enviro_risk': '' }"
    },
    "8_Risks": {
        "description": "Risks & Constraints",
        "prompt": "Extract Key Risks. Return JSON: { 'risk_resource_confidence': '', 'risk_metallurgy': '', 'risk_infrastructure': '', 'risk_social_permitting': '' }"
    },
    "9_Infra": {
        "description": "Infrastructure",
        "prompt": "Extract Infrastructure details. Return JSON: { 'power_source': 'Grid/Diesel/Hydro', 'water_source': '', 'access_road_rail': '', 'camp_requirements': '' }"
    },
    "10_QP": {
        "description": "Qualified Persons",
        "prompt": "Extract QP Info. Return JSON: { 'qp_names': ['Name1', 'Name2'], 'qp_firms': ['Firm1'] }"
    },
    "11_Compliance": {
        "description": "Compliance & Dates",
        "prompt": "Extract Report Details. Return JSON: { 'effective_date': '', 'filing_date': '', 'drill_meters_total': null }"
    }
}

# --- HELPER: ROBUST JSON PARSER ---
def extract_json_from_text(text):
    """Finds JSON objects { ... } in text, robust to conversational noise."""
    try:
        match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
        if match: text = match.group(1)
        start, end = text.find('{'), text.rfind('}')
        if start != -1 and end != -1:
            return json.loads(text[start:end+1])
        return {}
    except:
        return {}

# --- MAIN LOGIC ---

if not openai_key or not llama_cloud_key:
    st.warning("Please enter API Keys.")
    st.stop()

os.environ["OPENAI_API_KEY"] = openai_key
os.environ["LLAMA_CLOUD_API_KEY"] = llama_cloud_key
Settings.llm = OpenAI(model=model_choice, temperature=0)

# File Upload
uploaded_file = st.file_uploader("Upload Mining Report (PDF)", type=['pdf'])

if "index" not in st.session_state: st.session_state.index = None
if "last_data" not in st.session_state: st.session_state.last_data = None

if uploaded_file:
    if "last_uploaded" not in st.session_state or st.session_state.last_uploaded != uploaded_file.name:
        with st.spinner(f"Indexing {uploaded_file.name}..."):
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name
                
                # Instruction for LlamaParse to focus on tables
                parser = LlamaParse(result_type="markdown", verbose=True, language="en")
                docs = SimpleDirectoryReader(input_files=[tmp_path], file_extractor={".pdf": parser}).load_data()
                st.session_state.index = VectorStoreIndex.from_documents(docs)
                st.session_state.last_uploaded = uploaded_file.name
                st.success("Report Indexed!")
                os.remove(tmp_path)
            except Exception as e:
                st.error(f"Error: {e}")

# --- EXECUTION INTERFACE ---
if st.session_state.index:
    
    col1, col2 = st.columns([3, 1])
    with col1:
        if st.button("🚀 Run 11-Step Deep Extraction"):
            
            # Setup
            engine = st.session_state.index.as_query_engine(similarity_top_k=10, response_mode="tree_summarize")
            master_data = {}
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # THE LOOP: Iterate through the SCHEMA_CONFIG
            total_steps = len(SCHEMA_CONFIG)
            
            for i, (key, config) in enumerate(SCHEMA_CONFIG.items()):
                # Update UI
                step_num = i + 1
                status_text.markdown(f"**Step {step_num}/{total_steps}:** Analyzing {config['description']}...")
                progress_bar.progress(int((step_num / total_steps) * 100))
                
                # Run Query
                # We append "Return ONLY valid JSON" to ensure compliance
                full_prompt = f"{config['prompt']} Return ONLY valid JSON."
                response = engine.query(full_prompt)
                
                # Extract Data
                data = extract_json_from_text(str(response))
                if data:
                    master_data.update(data) # Merge into master dict
                else:
                    print(f"Failed on {key}") # Debug in terminal
            
            # Finalize
            status_text.markdown("✅ **Extraction Complete!**")
            progress_bar.empty()
            
            # Display & Save
            if master_data:
                df = pd.json_normalize([master_data])
                st.session_state.last_data = df
                st.dataframe(df)
            else:
                st.error("Extraction failed. No data found.")

    # Download
    if st.session_state.last_data is not None:
        st.divider()
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            st.session_state.last_data.to_excel(writer, index=False, sheet_name='Deep_Dive')
        st.download_button("Download Full Analysis (.xlsx)", output.getvalue(), "mining_deep_dive.xlsx")