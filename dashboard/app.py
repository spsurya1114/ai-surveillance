
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import json
import pandas as pd
import plotly.express as px

from rag.text_converter import convert_logs_to_text
from rag.embedder import RAGEmbedder
from rag.llm import ask_llm

# -------------------------------
# CONFIG
# -------------------------------
st.set_page_config(page_title="AI Surveillance Dashboard", layout="wide")
st.title("🎥 AI Surveillance Dashboard")

# -------------------------------
# SIDEBAR
# -------------------------------
st.sidebar.header("Controls")

camera = st.sidebar.selectbox("Select Camera", ["CAM_01", "CAM_02"])

person_filter = st.sidebar.text_input("Filter by Person ID")

# -------------------------------
# LOAD EVENTS
# -------------------------------
def load_events(camera):
    path = f"data/logs/{camera}.json"
    try:
        with open(path, "r") as f:
            return json.load(f)
    except:
        return []

events = load_events(camera)

# Apply filter
if person_filter:
    events = [e for e in events if str(e.get("person_id")) == person_filter]

# -------------------------------
# KPI METRICS
# -------------------------------
st.subheader("📊 Overview")

col1, col2, col3 = st.columns(3)

total_events = len(events)
total_persons = len(set(e.get("person_id") for e in events))
anomalies = sum(1 for e in events if e.get("event_type") == "loitering")

col1.metric("Total Events", total_events)
col2.metric("Unique Persons", total_persons)
col3.metric("Anomalies", anomalies)

# -------------------------------
# EVENTS TABLE
# -------------------------------
st.subheader("📋 Event Log")

if events:
    df = pd.DataFrame(events)

    # Highlight anomalies
    def highlight(row):
        if row["event_type"] == "loitering":
            return ["background-color: red"] * len(row)
        return [""] * len(row)

    st.dataframe(df.style.apply(highlight, axis=1), use_container_width=True)
else:
    st.warning("No events found")

# -------------------------------
# TIMELINE
# -------------------------------
st.subheader("📈 Event Timeline")

if events:
    df = pd.DataFrame(events)

    fig = px.scatter(
        df,
        x="timestamp",
        y="person_id",
        color="event_type",
        title="Movement Timeline",
    )

    st.plotly_chart(fig, use_container_width=True)

# -------------------------------
# RAG SETUP
# -------------------------------
@st.cache_resource
def load_rag(camera):
    log_path = f"data/logs/{camera}.json"
    texts = convert_logs_to_text(log_path)

    rag = RAGEmbedder()
    rag.build_index(texts)

    return rag

rag = load_rag(camera)

# -------------------------------
# CHAT SECTION
# -------------------------------
st.subheader("💬 Ask Questions")

query = st.text_input("Enter your question")

if st.button("Ask"):
    if query.strip() == "":
        st.warning("Enter a question")
    else:
        results = rag.search(query)
        context = "\n".join(results)

        with st.expander("📄 Retrieved Context"):
            for r in results:
                st.write("-", r)

        answer = ask_llm(query, context)

        st.subheader("🤖 Answer")
        st.success(answer)

# -------------------------------
# DOWNLOAD REPORT
# -------------------------------
st.subheader("⬇️ Export")

if st.button("Download Report"):
    report = json.dumps(events, indent=2)
    st.download_button("Download JSON", report, file_name="report.json")

# -------------------------------
# FOOTER
# -------------------------------
st.markdown("---")
st.markdown("🚀 AI Surveillance System | YOLO + Re-ID + RAG + LLM")

