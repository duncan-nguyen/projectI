import streamlit as st
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.pipeline import InformationExtractionPipeline

st.set_page_config(page_title="Medical IE Demo", layout="wide")

st.title("Medical Information Extraction Demo")
st.markdown("Extract entities and relations from Vietnamese medical texts.")

st.sidebar.header("Configuration")
ner_method = st.sidebar.selectbox(
    "Select NER Method",
    options=["gliner", "standard"],
    format_func=lambda x: "GLiNER (General)" if x == "gliner" else "Standard NER (PhoBERT)"
)

device_option = st.sidebar.selectbox(
    "Device",
    options=["cpu", "cuda"],
    index=0
)

default_text = "Bệnh nhân Nguyễn Văn A, 45 tuổi, trú tại Quận Cầu Giấy, Hà Nội. Ngày 20/05, bệnh nhân có biểu hiện sốt cao, ho khan và đau họng. Bệnh nhân đã đi đến Bệnh viện Bạch Mai để khám."
input_text = st.text_area("Input Text", value=default_text, height=150)

@st.cache_resource(show_spinner="Loading models...")
def get_pipeline(method, device):
    return InformationExtractionPipeline(ner_method=method, device=device)

if st.button("Process Extraction", type="primary"):
    if not input_text.strip():
        st.warning("Please enter some text.")
    else:
        try:
            with st.spinner("Processing..."):
                pipeline = get_pipeline(ner_method, device_option)
                result = pipeline.process(input_text)

            col1, col2 = st.columns([1, 1])

            with col1:
                st.subheader("Extracted Entities")
                if result.entities:
                    ent_data = [
                        {
                            "Text": e.text, 
                            "Label": e.label, 
                            "Confidence": f"{e.confidence:.4f}" if e.confidence else "N/A"
                        }
                        for e in result.entities
                    ]
                    st.dataframe(ent_data, use_container_width=True)
                else:
                    st.info("No entities found.")

                st.subheader("Extracted Relations")
                if result.relations:
                    rel_data = [
                        {
                            "Subject": next((e.text for e in result.entities if e.id == r.source_id), r.source_id),
                            "Relation": r.relation_type,
                            "Object": next((e.text for e in result.entities if e.id == r.target_id), r.target_id),
                            "Evidence": r.evidence
                        }
                        for r in result.relations
                    ]
                    st.table(rel_data)
                else:
                    st.info("No relations found.")

            with col2:
                st.subheader("Knowledge Graph")
                if result.entities:
                    from src.visualization import Visualizer
                    mermaid_code = Visualizer.generate_knowledge_graph(result.entities, result.relations)
                    st.markdown(f"```mermaid\n{mermaid_code}\n```")
                else:
                    st.markdown("No graph to display.")
                    
            with st.expander("Show Preprocessing Details"):
                st.markdown("**Cleaned Text:**")
                st.write(result.cleaned_text)
                st.markdown("**Tokens:**")
                st.write(result.tokens)

        except Exception as e:
            st.error(f"An error occurred: {e}")
            import traceback
            st.code(traceback.format_exc())
