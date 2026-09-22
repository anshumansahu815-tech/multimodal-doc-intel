import streamlit as st
from PIL import Image
import os
import json
import time
import textstat
from google import genai
from google.genai import types
from google.genai.errors import ServerError, APIError
from schemas import InvoiceSchema, ChartAnalysisSchema

st.set_page_config(page_title="Multimodal Doc Intel", layout="wide")

# API Configuration via sidebar
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    api_key = st.sidebar.text_input("Gemini API Key", type="password")

if not api_key:
    st.info("Enter your Gemini API key in the sidebar to begin.")
    st.stop()

client = genai.Client(api_key=api_key)

# Resilient request handler with dynamic model fallback and exponential backoff
def execute_gemini_call(contents, config=None, max_retries=3):
    preferred_models = ["gemini-3.6-flash", "gemini-2.0-flash", "gemini-1.5-flash"]
    
    # Query active models on this key to avoid 404 errors
    try:
        available_models = [m.name.split("/")[-1] for m in client.models.list()]
        candidate_models = [m for m in preferred_models if m in available_models]
        if not candidate_models:
            candidate_models = ["gemini-3.6-flash"]
    except Exception:
        candidate_models = ["gemini-3.6-flash"]

    last_error = None
    for model_name in candidate_models:
        for attempt in range(max_retries):
            try:
                if config:
                    return client.models.generate_content(
                        model=model_name,
                        contents=contents,
                        config=config
                    )
                else:
                    return client.models.generate_content(
                        model=model_name,
                        contents=contents
                    )
            except ServerError as e:
                last_error = e
                wait = 4 * (attempt + 1)
                st.warning(f"Server busy on {model_name} (503). Retrying in {wait}s...")
                time.sleep(wait)
            except APIError as e:
                last_error = e
                break  # If endpoint is invalid or unsupported, fail over to next model

    raise last_error

st.title("Multimodal Document Intelligence & Jargon-Free Summarizer")
st.caption("Extract structured data and generate plain-language explanations from visual documents.")

col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("1. Ingestion")
    doc_type = st.selectbox("Document Type", ["Invoice / Receipt", "Chart / Technical Plot"])
    uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Document Preview", use_container_width=True)

if uploaded_file and st.button("Process Document", type="primary"):
    with col_right:
        st.subheader("2. Results & Metrics")
        with st.spinner("Analyzing document and extracting structured schema..."):
            target_schema = InvoiceSchema if doc_type == "Invoice / Receipt" else ChartAnalysisSchema
            
            try:
                # Step A: Multimodal Extraction via Schema Enforcement
                extract_prompt = (
                    "Extract all fields visible in this document strictly adhering to the schema. "
                    "Do not invent missing data."
                )
                res = execute_gemini_call(
                    contents=[image, extract_prompt],
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=target_schema,
                    )
                )
                extracted_json = json.loads(res.text)

                # Step B: Plain-Language Translation (Targeted to 8th-Grade Level)
                summary_prompt = f"""
                You are a plain-language communicator. Explain the following extracted document information to an 8th grader.
                
                Rules:
                1. Replace technical terminology, accounting jargon, and statistical metrics with simple words.
                2. Grounding Rule: State ONLY facts and numbers present in the JSON below. Never invent details.
                3. Structure your response into: Snapshot, Key Numbers, and Meaning.
                
                Extracted JSON:
                {json.dumps(extracted_json, indent=2)}
                """
                summary_res = execute_gemini_call(contents=summary_prompt)
                summary_text = summary_res.text

                # Step C: Readability Verification
                ease = textstat.flesch_reading_ease(summary_text)
                grade = textstat.flesch_kincaid_grade(summary_text)

                # Output Tabs
                tab1, tab2, tab3 = st.tabs(["Plain Summary", "Structured JSON", "Quality Metrics"])
                with tab1:
                    st.markdown(summary_text)
                with tab2:
                    st.json(extracted_json)
                with tab3:
                    m1, m2 = st.columns(2)
                    m1.metric("Flesch Reading Ease", f"{ease:.1f}", delta="Target: > 60")
                    m2.metric("Reading Grade Level", f"Grade {grade:.1f}", delta="Target: <= Grade 8", delta_color="inverse")

            except ServerError:
                st.error("Google servers are temporarily experiencing high peak traffic. Please wait 10–15 seconds and click 'Process Document' again.")
            except Exception as err:
                st.error(f"Error during document processing: {err}")