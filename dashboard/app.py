import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import json
import pandas as pd
import plotly.express as px
from datetime import datetime
from pathlib import Path

from rag.temporal_reasoning import TemporalRetriever

# -------------------------------
# EVIDENCE PATH RESOLVER
# -------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

def resolve_evidence_path(path):
    if not path:
        return None

    p = Path(path)

    if p.is_absolute():
        return p if p.exists() else None

    resolved = PROJECT_ROOT / p
    return resolved if resolved.exists() else None


# Initialize Temporal Retriever
retriever = TemporalRetriever()

# -------------------------------
# PREMIUM PAGE CONFIG
# -------------------------------
st.set_page_config(
    page_title="AI Surveillance Command Center",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sleek Dark/Custom Theme Injection
st.markdown("""
<style>
    .reportview-container {
        background-color: #0e1117;
    }
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border: 1px solid rgba(255,255,255,0.1);
        text-align: center;
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: bold;
        color: #00ffcc;
        margin-bottom: 0.2rem;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #8892b0;
        text-transform: uppercase;
        letter-spacing: 0.1rem;
    }
    h1, h2, h3 {
        color: #ffffff !important;
        font-family: 'Outfit', sans-serif;
    }
</style>
""", unsafe_allow_html=True)

st.title("🎥 AI Surveillance Command Center")
st.markdown("---")

# -------------------------------
# LOAD ALL LOGS DYNAMICALLY
# -------------------------------
def load_all_events_from_disk():
    events = retriever.load_all_events()
    return events

all_events = load_all_events_from_disk()

# -------------------------------
# SIDEBAR FILTERS
# -------------------------------
st.sidebar.header("🎛️ Control Panel")

# Camera selection
cameras = sorted(list(set(e["camera_id"] for e in all_events))) if all_events else []
camera_filter = st.sidebar.multiselect("Select Cameras", options=["All"] + cameras, default="All")

# Person ID selection
person_ids = []
for e in all_events:
    pid = e.get("person_id")
    if pid is not None and pid != "":
        person_ids.append(str(pid))
person_ids = sorted(list(set(person_ids)))
person_filter = st.sidebar.selectbox("Filter by Person ID", ["All"] + person_ids)

# Anomaly/Event Type selection
event_types = sorted(list(set(e.get("anomaly_type") if e.get("event_type") == "anomaly" else e["event_type"] for e in all_events))) if all_events else []
type_filter = st.sidebar.selectbox("Event/Anomaly Type", ["All"] + event_types)

# Risk Level selection
risk_levels = ["All", "LOW", "MEDIUM", "HIGH", "CRITICAL"]
risk_filter = st.sidebar.selectbox("Risk Level", risk_levels)

# Date filter
dates = sorted(list(set(e.get("date", datetime.now().strftime("%Y-%m-%d")) for e in all_events))) if all_events else []
date_filter = st.sidebar.selectbox("Date", ["All"] + dates)

st.sidebar.markdown("---")

# -------------------------------
# FILTER LOGIC
# -------------------------------
filtered_events = []
for e in all_events:
    # Camera filter
    if "All" not in camera_filter and e["camera_id"] not in camera_filter:
        continue
    # Person filter
    if person_filter != "All" and str(e.get("person_id")) != person_filter:
        continue
    # Event type filter
    e_type = e.get("anomaly_type") if e.get("event_type") == "anomaly" else e["event_type"]
    if type_filter != "All" and e_type != type_filter:
        continue
    # Risk Level filter
    if risk_filter != "All" and e.get("severity", "LOW") != risk_filter:
        continue
    # Date filter
    e_date = e.get("date", datetime.now().strftime("%Y-%m-%d"))
    if date_filter != "All" and e_date != date_filter:
        continue
    
    filtered_events.append(e)

# -------------------------------
# KPI TOP METRICS
# -------------------------------
st.subheader("📊 Operational Analytics")

col1, col2, col3, col4, col5 = st.columns(5)

# Calculations
unique_persons = len(set(e.get("person_id") for e in filtered_events if e.get("person_id") is not None))
total_events_count = len(filtered_events)
anomalies_count = sum(1 for e in filtered_events if e["event_type"] not in ["entry", "movement", "exit"])
high_risk_count = sum(1 for e in filtered_events if e.get("severity") == "HIGH")
critical_count = sum(1 for e in filtered_events if e.get("severity") == "CRITICAL")

with col1:
    st.markdown(f'<div class="metric-card"><div class="metric-value">{unique_persons}</div><div class="metric-label">Total People</div></div>', unsafe_allow_html=True)
with col2:
    st.markdown(f'<div class="metric-card"><div class="metric-value">{total_events_count}</div><div class="metric-label">Total Events</div></div>', unsafe_allow_html=True)
with col3:
    st.markdown(f'<div class="metric-card"><div class="metric-value" style="color: #ffcc00;">{anomalies_count}</div><div class="metric-label">Anomalies</div></div>', unsafe_allow_html=True)
with col4:
    st.markdown(f'<div class="metric-card"><div class="metric-value" style="color: #ff3300;">{high_risk_count}</div><div class="metric-label">High Risk</div></div>', unsafe_allow_html=True)
with col5:
    st.markdown(f'<div class="metric-card"><div class="metric-value" style="color: #ff0000; text-shadow: 0 0 10px rgba(255,0,0,0.5);">{critical_count}</div><div class="metric-label">Critical Incidents</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# -------------------------------
# CHARTS ROW
# -------------------------------
if filtered_events:
    df = pd.DataFrame(filtered_events)
    
    row1_col1, row1_col2, row1_col3 = st.columns(3)
    
    with row1_col1:
        st.subheader("🔥 Risk Distribution")
        risk_counts = df["severity"].value_counts().reset_index()
        fig_risk = px.pie(
            risk_counts, 
            names="severity", 
            values="count",
            color="severity",
            color_discrete_map={"LOW": "#00ffcc", "MEDIUM": "#ffcc00", "HIGH": "#ff5500", "CRITICAL": "#ff0000"},
            hole=0.4
        )
        fig_risk.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#fff")
        st.plotly_chart(fig_risk, use_container_width=True)

    with row1_col2:
        st.subheader("⚠️ Anomaly Categories")
        display_types = df.apply(lambda row: row["anomaly_type"] if row.get("event_type") == "anomaly" and "anomaly_type" in row and pd.notna(row["anomaly_type"]) else row["event_type"], axis=1)
        type_counts = display_types.value_counts().reset_index()
        type_counts.columns = ["event_type", "count"]
        fig_type = px.bar(
            type_counts, 
            x="event_type", 
            y="count",
            labels={"event_type": "Event Type", "count": "Frequency"},
            color="event_type",
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig_type.update_layout(showlegend=False, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#fff")
        st.plotly_chart(fig_type, use_container_width=True)

    with row1_col3:
        st.subheader("📹 Camera Anomaly Stats")
        # count anomalies per camera
        anom_df = df[~df["event_type"].isin(["entry", "movement", "exit"])]
        if not anom_df.empty:
            cam_counts = anom_df["camera_id"].value_counts().reset_index()
            fig_cam = px.bar(
                cam_counts, 
                x="camera_id", 
                y="count",
                labels={"camera_id": "Camera", "count": "Anomalies"},
                color="camera_id",
                color_discrete_sequence=["#ff5500"]
            )
            fig_cam.update_layout(showlegend=False, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#fff")
            st.plotly_chart(fig_cam, use_container_width=True)
        else:
            st.info("No anomalies detected in selected cameras.")

# -------------------------------
# EVIDENCE VIEWER & DETAILS
# -------------------------------
st.markdown("---")
st.subheader("🕵️ Selected Incident Details & Evidence")

if filtered_events:
    event_options = [
        f"{e.get('abs_time', e['timestamp'])} - {e['camera_id']} - {e['event_type']} (PID: {e.get('person_id', 'N/A')})"
        for e in filtered_events
    ]
    selected_option = st.selectbox("Select an Event to View Evidence:", event_options)
    selected_idx = event_options.index(selected_option)
    selected_event = filtered_events[selected_idx]

    detail_col, evidence_col = st.columns([1, 1.2])

    with detail_col:
        st.write("### Event Information")
        details_data = {
            "Field": ["Event ID", "Camera ID", "Timestamp (Absolute)", "Timestamp (Video Sec)", "Event/Anomaly Type", "Risk Score", "Severity", "Person ID", "Object ID", "Zone", "Description"],
            "Value": [
                str(selected_event.get("event_id", "N/A")),
                str(selected_event["camera_id"]),
                str(selected_event.get("abs_time", "N/A")),
                f"{selected_event['timestamp']} sec",
                str(selected_event["event_type"]),
                str(selected_event.get("risk_score", 0)),
                str(selected_event.get("severity", "LOW")),
                str(selected_event.get("person_id", "N/A")),
                str(selected_event.get("object_id", "N/A")),
                str(selected_event.get("zone", "N/A")),
                str(selected_event["description"])
            ]
        }
        st.table(pd.DataFrame(details_data))

    with evidence_col:
        st.write("### Captured Evidence")
        img_path = selected_event.get("evidence_path")
        clip_path = selected_event.get("evidence_clip_path")

        if img_path and os.path.exists(img_path):
            st.image(img_path, caption=f"Evidence Capture - {selected_event.get('abs_time')}", use_container_width=True)
        else:
            st.warning("No still image evidence file found on disk.")

        if clip_path and os.path.exists(clip_path):
            st.video(clip_path)
        elif clip_path:
            st.info("Evidence video clip registered but not found on disk.")



# -------------------------------
# PERSON MOVEMENT PATH
# -------------------------------
if person_filter != "All" and filtered_events:
    st.markdown("---")
    st.subheader(f"🚶 Person P{person_filter} Activity Path")
    
    person_df = df[df["person_id"].astype(str) == str(person_filter)].copy()
    if not person_df.empty:
        # Build path coordinates
        path_list = []
        for idx, row in person_df.iterrows():
            path_list.append(f"{row['abs_time']} ({row['camera_id']} - {row['zone']})")
        
        st.markdown("**Movement Path Sequence:**")
        st.write(" ➡️ ".join(path_list))
    else:
        st.info("No path records found.")

# -------------------------------
# TEMPORAL GROUNDED RAG CHAT
# -------------------------------
st.markdown("---")
st.subheader("💬 Temporal Intelligence Chat (RAG + LLM)")
st.markdown("*Ask temporal questions such as 'What did person 1 do today?' or 'Show timeline of P014 between 10:00 and 10:10'. Answers are strictly grounded in retrieved evidence.*")

query = st.text_input("Enter your command / question:")
if st.button("Ask Analyst"):
    if not query.strip():
        st.warning("Please type a question.")
    else:
        with st.spinner("Analyzing event records..."):
            answer, context = retriever.query(query)
            
            st.subheader("🤖 Surveillance Analyst")
            if "Insufficient evidence" in answer:
                st.error(answer)
            else:
                st.success(answer)

            with st.expander("🔎 Show grounding evidence"):
                events_to_show = retriever.filter_evidence_by_query(query, getattr(retriever, "last_filtered_events", []))
                
                if not events_to_show:
                    st.info("No grounding evidence events found for this query.")
                else:
                    for idx, e in enumerate(events_to_show):
                        # Construct a header for each card
                        time_str = e.get("abs_time", str(e.get("timestamp", "00:00:00")))
                        cam_id = e.get("camera_id", "N/A")
                        etype = e.get("event_type", "unknown")
                        
                        # Anomaly visual distinction
                        is_anomaly = etype == "anomaly" or etype in ["loitering", "intrusion", "wrong_direction", "sudden_movement", "crowd_anomaly", "abandoned_object"]
                        
                        # Person/Object ID
                        pid = e.get("person_id")
                        oid = e.get("object_id")
                        entity_str = ""
                        if pid is not None and pid != "":
                            entity_str = f"Person P{pid}"
                        elif oid is not None and oid != "":
                            entity_str = f"Object ID {oid}"
                            
                        # Anomaly type
                        atype = e.get("anomaly_type")
                        atype_str = f" — {atype.replace('_', ' ').capitalize()}" if (is_anomaly and atype) else ""
                        
                        # Visual styling / emoji
                        prefix = "⚠️ " if is_anomaly else "🔹 "
                        
                        # Title
                        title = f"{time_str} | {cam_id}"
                        card_header = f"{prefix}{entity_str or 'System'}{atype_str or (' — ' + etype.capitalize())}"
                        
                        # Display card
                        with st.container(border=True):
                            st.markdown(f"**{title}**")
                            st.markdown(f"### {card_header}")
                            st.markdown(
                                f"**Risk Score:** {e.get('risk_score', 0)} | **Severity:** {e.get('severity', 'LOW')}"
                            )
                            st.write(e.get('description', 'No description provided.'))
                            
                            # Check evidence path
                            img_path = resolve_evidence_path(e.get("evidence_path"))
                            clip_path = resolve_evidence_path(e.get("evidence_clip_path"))
                            
                            has_evidence = False
                            
                            if img_path or clip_path:
                                col_img, col_vid = st.columns(2)
                                
                                if img_path:
                                    has_evidence = True
                                    with col_img:
                                        st.image(str(img_path), caption="Captured Still Frame", use_container_width=True)
                                        # Download button for image
                                        try:
                                            with open(img_path, "rb") as file:
                                                st.download_button(
                                                    label="Download Image",
                                                    data=file,
                                                    file_name=img_path.name,
                                                    mime="image/jpeg",
                                                    key=f"dl_img_{idx}_{e.get('event_id', idx)}"
                                                )
                                        except Exception as ex:
                                            st.error(f"Error loading image download: {ex}")
                                            
                                if clip_path:
                                    has_evidence = True
                                    with col_vid:
                                        st.video(str(clip_path))
                                        # Download button for video
                                        try:
                                            with open(clip_path, "rb") as file:
                                                st.download_button(
                                                    label="Download Video",
                                                    data=file,
                                                    file_name=clip_path.name,
                                                    mime="video/mp4",
                                                    key=f"dl_vid_{idx}_{e.get('event_id', idx)}"
                                                )
                                        except Exception as ex:
                                            st.error(f"Error loading video download: {ex}")
                                            
                            if not has_evidence:
                                st.markdown("*No evidence captured for this event.*")

# -------------------------------
# DATA EXPORT
# -------------------------------
st.markdown("---")
st.subheader("⬇️ Command Center Export")

if st.button("Generate System JSON Report"):
    report = json.dumps(filtered_events, indent=2)
    st.download_button(
        label="Download System JSON Report",
        data=report,
        file_name=f"surveillance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json"
    )

# -------------------------------
# FOOTER
# -------------------------------
st.markdown("---")
st.markdown("<p style='text-align: center; color: #8892b0;'>🚀 AI Command Center | YOLOv8 + Re-ID + Grounded RAG + LLM | Surya SP</p>", unsafe_allow_html=True)


