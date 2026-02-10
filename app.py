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
st.title("⛏️ Mining Analyst - NI 43-101 Deep Dive")

# --- SIDEBAR: CONFIGURATION ---
with st.sidebar:
    st.header("Configuratie")
    
    # 1. Probeer keys uit Streamlit Secrets te halen
    try:
        openai_key = st.secrets["OPENAI_API_KEY"]
        st.success("✅ OpenAI Key geladen uit veilige opslag")
    except (FileNotFoundError, KeyError):
        # Als ze niet in secrets staan, vraag erom (voor manueel gebruik)
        openai_key = st.text_input("OpenAI API Key", type="password")

    try:
        llama_cloud_key = st.secrets["LLAMA_CLOUD_API_KEY"]
        st.success("✅ LlamaCloud Key geladen uit veilige opslag")
    except (FileNotFoundError, KeyError):
        llama_cloud_key = st.text_input("LlamaCloud API Key", type="password")
    
    st.divider()
    
    model_choice = st.selectbox("Select Model", ["gpt-4o", "gpt-4o-mini"])
    if model_choice == "gpt-4o":
        st.info("Tip: GPT-4o wordt aanbevolen voor complexe tabellen.")

# --- API SETUP ---
if not openai_key or not llama_cloud_key:
    st.warning("Voer je API keys in om te beginnen.")
    st.stop()

# Zet de keys in de omgeving zodat LlamaIndex ze kan vinden
os.environ["OPENAI_API_KEY"] = openai_key
os.environ["LLAMA_CLOUD_API_KEY"] = llama_cloud_key

# --- THE MASTER SCHEMA (18 STEPS) ---
SCHEMA_CONFIG = {
    "1_Identity": {
        "description": "Project Identity",
        "prompt": """
        Extract project identity. 
        Return JSON: { 
            'proj_name': '', 
            'company': '', 
            'prop_type': 'Exploration/Dev/Prod', 
            'loc_country': '', 
            'source_page': 'Page number where this is found' 
        }
        """
    },
    "2_Resources": {
        "description": "Mineral Resources",
        "prompt": """
        Locate the 'Mineral Resource Estimate' table. 
        Focus on the 'Total Measured and Indicated' row (or 'M&I'). 
        Extract the following:
        - Tonnage (k tonnes or tonnes)
        - Gold Grade (Au g/t)
        - Contained Gold (oz)
        If multiple cut-offs are shown, select the 'Base Case' or the highlighted row.
        Return JSON: { 'res_total_tonnes': '', 'res_grade_au': '', 'source_table': '' }
        """
    },
    "3_Reserves": {
        "description": "Mineral Reserves",
        "prompt": """
        Locate the 'Mineral Reserve Estimate' table. 
        Focus on the 'Proven and Probable' totals. 
        Extract:
        - Tonnage (k tonnes or tonnes)
        - Gold Grade (Au g/t)
        - Contained Gold (oz)
        Return JSON: { 'resv_total_tonnes': '', 'resv_grade_au': '', 'source_table': '' }
        """
    },
    "4_Mining": {
        "description": "Mining Method",
        "prompt": """
        Extract Mining Metrics. 
        Return JSON: { 
            'mining_method': 'Open Pit/UG', 
            'strip_ratio': null, 
            'mining_rate_tpd': null, 
            'source_context': 'Quote the sentence describing the method' 
        }
        """
    },
    "5_Production": {
        "description": "Production Profile",
        "prompt": """
        Locate the production schedule (often in 'Mining' or 'Economic Analysis').
        Extract:
        - Annual gold production (oz/year) for the first full year of commercial production.
        - LOM average annual production (oz/year).
        - Mine life (years).
        Return JSON: { 
            'prod_year1_oz': null,
            'prod_lom_avg_oz': null,
            'mine_life_years': null,
            'source_table': ''
        }
        """
    },
    "6_Metallurgy": {
        "description": "Processing",
        "prompt": """
        Extract Metallurgy. 
        Return JSON: { 
            'process_method': 'CIL/Flotation/Heap', 
            'recovery_au_pct': null, 
            'throughput_tpd': null, 
            'source_page': 'page #'
        }
        """
    },
    "7_Grades": {
        "description": "Head Grades / Multi-Metal",
        "prompt": """
        Identify the section describing head grades or ROM grades. 
        Extract grades for all relevant metals:
        - Au (g/t)
        - Ag (g/t)
        - Cu (%)
        - Any additional listed metal
        Return JSON: {
            'grade_au': null,
            'grade_ag': null,
            'grade_cu': null,
            'other_grades': {},
            'source_page': ''
        }
        """
    },
    "8_Recoveries": {
        "description": "Metallurgical Recoveries",
        "prompt": """
        Find metallurgical testwork or processing sections.
        Extract metal recoveries:
        - Gold recovery (%)
        - Silver recovery (%)
        - Copper recovery (%)
        Return JSON: {
            'rec_au_pct': null,
            'rec_ag_pct': null,
            'rec_cu_pct': null,
            'source_page': ''
        }
        """
    },
    "9_Costs": {
        "description": "Capital Costs (Capex)",
        "prompt": """
        Locate the 'Capital Cost Summary' table (often Section 21).
        Extract:
        - 'Initial Capital' (or Pre-Production Capital).
        - 'Sustaining Capital'.
        - 'Total LOM Capital'.
        Return JSON: { 'capex_initial_m': '', 'capex_sustaining_m': '', 'source_table': '' }
        """
    },
    "10_Opex": {
        "description": "Operating Costs (Opex)",
        "prompt": """
        Locate the operating cost breakdown.
        Extract:
        - Mining cost ($/t mined or $/t ore)
        - Processing cost ($/t processed)
        - G&A ($/t)
        - Total operating cost ($/t)
        Return JSON: {
            'opex_mining_per_t': null,
            'opex_processing_per_t': null,
            'opex_ga_per_t': null,
            'opex_total_per_t': null,
            'source_table': ''
        }
        """
    },
    "11_AISC": {
        "description": "All-in Sustaining Costs",
        "prompt": """
        Locate AISC values. 
        Extract:
        - AISC ($/oz)
        - AISC breakdown if provided
        Return JSON: {
            'aisc_per_oz': null,
            'aisc_breakdown': {},
            'source_page': ''
        }
        """
    },
    "12_Economics": {
        "description": "Base Case Economics",
        "prompt": """
        Extract Base Case Economics. 
        Return JSON: { 
            'price_assumed_au': null, 
            'npv_post_tax_m': null, 
            'irr_post_tax_pct': null, 
            'payback_years': null, 
            'source_page': 'Page #',
            'confidence': 'High/Low (Low if you had to calculate it)' 
        }
        """
    },
    "13_Infrastructure": {
        "description": "Infrastructure",
        "prompt": """
        Identify infrastructure statements.
        Extract:
        - Power source
        - Water source
        - Roads / access
        - Nearby infrastructure (ports, rail)
        Return JSON: {
            'power_source': '',
            'water_source': '',
            'access_roads': '',
            'infrastructure_notes': '',
            'source_page': ''
        }
        """
    },
    "14_Environment": {
        "description": "Environmental",
        "prompt": """
        Extract environmental data:
        - Tailings type (e.g., conventional, dry-stack)
        - Water management issues
        - Key environmental risks cited
        Return JSON: {
            'tailings_type': '',
            'env_risks': '',
            'water_mgmt': '',
            'source_page': ''
        }
        """
    },
    "15_Permitting": {
        "description": "Permitting Status",
        "prompt": """
        Identify permitting stage:
        - Environmental permits
        - Mining licence status
        - Any pending approvals
        Return JSON: {
            'permit_status': '',
            'critical_permits': '',
            'source_page': ''
        }
        """
    },
    "16_Risks": {
        "description": "Project Risks",
        "prompt": """
        Extract any explicit risk statements made by Qualified Persons:
        - Geological risks
        - Metallurgical risks
        - Operational risks
        - ESG or social risks
        Return JSON: {
            'geo_risks': '',
            'meta_risks': '',
            'operational_risks': '',
            'esg_risks': '',
            'source_context': ''
        }
        """
    },
    "17_QP": {
        "description": "Qualified Persons",
        "prompt": """
        Extract Qualified Person information:
        - Name(s)
        - Area of responsibility (geology, mining, metallurgy, economic analysis)
        - Affiliation (consulting firm, independent QP, etc.)
        Return JSON: {
            'qp_list': [],
            'source_page': ''
        }
        """
    },
    "18_Compliance": {
        "description": "Compliance Details",
        "prompt": """
        Identify compliance information:
        - Effective date of the report
        - Report type (PFS, FS, PEA)
        - Data sources and QA/QC notes
        Return JSON: {
            'effective_date': '',
            'report_type': '',
            'qa_qc_summary': '',
            'source_page': ''
        }
        """
    }
}

# --- HELPER: ROBUST JSON PARSER ---
def extract_json_from_text(text):
    """
    Finds JSON objects { ... } in text, robust to conversational noise.
    """
    try:
        # Try to find text between ```json and ``` tags first
        match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
        if match: 
            text = match.group(1)
            
        # Locate the first '{' and the last '}'
        start = text.find('{')
        end = text.rfind('}')
        
        if start != -1 and end != -1:
            json_str = text[start:end+1]
            return json.loads(json_str)
        return {}
    except:
        return {}

# --- MAIN APP LOGIC ---

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
if "last_data" not in st.session_state:
    st.session_state.last_data = None

if uploaded_file:
    if "last_uploaded" not in st.session_state or st.session_state.last_uploaded != uploaded_file.name:
        with st.spinner(f"Indexing {uploaded_file.name}... (This may take a minute)"):
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_file_path = tmp_file.name

                # LlamaParse Configuration
                parser = LlamaParse(
                    result_type="markdown", 
                    verbose=True, 
                    language="en",
                    parsing_instruction="Extract tables containing Resources, Reserves, Economics, and Costs."
                )
                
                file_extractor = {".pdf": parser}
                documents = SimpleDirectoryReader(input_files=[tmp_file_path], file_extractor=file_extractor).load_data()
                
                # Build Index
                st.session_state.index = VectorStoreIndex.from_documents(documents)
                st.session_state.last_uploaded = uploaded_file.name
                st.success("Report Indexed Successfully!")
                
                # Cleanup
                os.remove(tmp_file_path)

            except Exception as e:
                st.error(f"Error during indexing: {e}")

# --- EXTRACTION INTERFACE ---
if st.session_state.index:
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.write(f"**Ready to analyze:** {st.session_state.last_uploaded}")
        
        if st.button("🚀 Run Full 18-Step Extraction"):
            
            # Initialize Engine
            engine = st.session_state.index.as_query_engine(
                similarity_top_k=8,  # Balanced for speed/accuracy
                response_mode="tree_summarize"
            )
            
            master_data = {}
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # ITERATE THROUGH SCHEMA
            total_steps = len(SCHEMA_CONFIG)
            
            for i, (key, config) in enumerate(SCHEMA_CONFIG.items()):
                # Update UI
                step_num = i + 1
                status_text.markdown(f"**Step {step_num}/{total_steps}:** Analyzing {config['description']}...")
                progress_bar.progress(int((step_num / total_steps) * 100))
                
                # Query AI
                try:
                    full_prompt = f"{config['prompt']} Return ONLY valid JSON."
                    response = engine.query(full_prompt)
                    
                    # Parse Data
                    data = extract_json_from_text(str(response))
                    if data:
                        master_data.update(data)
                        print(f"✅ {key} Success") # Logs to terminal
                    else:
                        print(f"⚠️ {key} Returned Empty")
                        
                except Exception as e:
                    print(f"❌ Error on {key}: {e}")
            
            # Finish
            progress_bar.progress(100)
            status_text.markdown("✅ **Analysis Complete!**")
            
            if master_data:
                # Flatten and display
                df = pd.json_normalize([master_data])
                st.session_state.last_data = df
                st.dataframe(df)
            else:
                st.error("Extraction failed. Please check the logs.")

    # Download Button
    if st.session_state.last_data is not None:
        st.divider()
        st.subheader("📥 Export Results")
        
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            st.session_state.last_data.to_excel(writer, index=False, sheet_name='Deep_Analysis')
        
        st.download_button(
            label="Download Full Excel Report (.xlsx)",
            data=output.getvalue(),
            file_name="mining_deep_dive_export.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )