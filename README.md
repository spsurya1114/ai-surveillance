# 🎥 AI Surveillance Command Center (RAG + LLM Enabled)

An intelligent multi-camera security surveillance system that analyzes video feeds, detects and tracks individuals using deep learning Re-ID, monitors for security anomalies, calculates dynamic risk scores, dispatches real-time alerts with evidence capture, and enables natural language temporal reasoning queries (grounded RAG + LLM).

---

## 🏗️ System Architecture

```text
       CCTV Video Feed (Multi-Camera)
                    │
                    ▼
          Detection & Tracking (YOLOv8)
                    │
                    ▼
          Person Re-ID Feature Extraction (MobileNetV3 / OSNet)
                    │
                    ▼
          Feature Matching & Identity Linking (Matcher)
                    │
                    ▼
          State Update & Zone-Based Logging (EventLogger)
                    │
                    ▼
  ┌─────────────────┴─────────────────┐
  ▼                                   ▼
Anomaly Detection (Per Frame)    Basic Transitions (Entry/Movement/Exit)
  │                                   │
  ▼                                   ▼
Risk Engine (Profile Accumulation) ──► Unified Camera Log (data/logs/CAM_XX.json)
  │
  ▼
Alert Manager (Cooldown Check) ──► Local Alerts (alerts.json) & Email (SMTP)
  │
  ▼
Evidence Capture (JPG Frame + MP4 Sliding Clip in evidence/YYYY-MM-DD/)
                    │
                    ▼
  ┌─────────────────┴─────────────────┐
  ▼                                   ▼
Temporal Retriever (RAG Pipeline)    Interactive Dashboard (Streamlit)
  │                                   │
  ▼                                   ▼
LLM Answer Generation ───────────────► User Interface
(Groq Llama-3.3 / Offline Simulator)
```

---

## 🧠 Key Capabilities

### 1. Advanced Anomaly Detection
The anomaly detection suite evaluates tracking inputs frame-by-frame using clean object-oriented detectors:
* **Loitering Detector**: Flags if a tracked person remains active in a camera feed longer than `loitering_threshold` seconds.
* **Restricted Zone Intrusion**: Detects if a person's bounding-box center falls inside any camera-specific restricted polygon coordinates (evaluated using OpenCV `cv2.pointPolygonTest`).
* **Wrong-Direction Movement**: Records a running trajectory window (last 15 points) for each person, calculates their net movement vector, and flags if it consistently violates the camera's `allowed_direction` angle.
* **Sudden/Running Movement**: Computes instantaneous velocity (pixels/sec) from consecutive frames and flags sudden accelerations or running behavior.
* **Crowd Anomaly**: Counts active people in the scene and triggers an incident if count exceeds `crowd_threshold`.
* **Abandoned Object Detector**: Monitors YOLO classes like `backpack`, `handbag`, `umbrella`, and `suitcase`. If an object remains stationary and no person is detected within a proximity radius (150 pixels) for more than `abandoned_object_threshold` seconds, it flags an abandoned object incident.

### 2. Configurable Risk Scoring Engine
* Combines active anomalies for the same person within a sliding time window (`risk_combination_window` default 30s) to build a unified behavioral risk profile.
* Sums the risk weights of active unique anomalies (e.g. Loitering = +25, Intrusion = +40, Wrong Direction = +20) and clamps the score to `0–100`.
* Classifies the overall risk into four tiers:
  * **LOW**: 0–39
  * **MEDIUM**: 40–69
  * **HIGH**: 70–89
  * **CRITICAL**: 90–100

### 3. Real-Time Alerts & Cooldown
* Dispatches alerts to a local JSON log (`data/logs/alerts.json`) and SMTP Email whenever the risk level exceeds the configured `alert_threshold` or a `CRITICAL` anomaly occurs.
* Incorporates a cooldown/de-duplication mechanism based on `(person_id/object_id, anomaly_type, camera_id)` to prevent flooding the logs/email across consecutive frames.

### 4. Automatic Evidence Capture
When a significant incident occurs:
* Saves the exact frame containing the event as a `.jpg` image.
* Commits a short `.mp4` video clip (capturing 3 seconds of pre-buffered history before the event and 2 seconds of subsequent frames) using a sliding buffer.
* Directory structure:
  ```text
  evidence/
      2026-08-23/
          CAM_01/
              intrusion/
                  event_intrusion_abc123.jpg
                  event_intrusion_abc123.mp4
  ```

### 5. Grounded Temporal Reasoning
* The RAG layer does not just do semantic search on isolated sentences. It uses a **Temporal Retriever** that parses natural language questions to extract filtering parameters (`person_id`, `camera_id`, `start_time`, `end_time`, `anomaly_type`, `risk_level`, `min_risk_score`).
* Resolves relative video times to absolute times (e.g., `10:02:15`) using the camera start times in the configuration.
* Dynamically extracts and filters events from logs, sorts them chronologically, constructs a timeline, and feeds it as grounding context to the LLM.
* **Groundedness Guarantee**: If no records match, the pipeline immediately returns `"Insufficient evidence: no event records were found matching your query."` to prevent LLM hallucinations.
* **Offline Simulator**: If no `GROQ_API_KEY` is configured, a high-quality rule-based local simulation engine answers the temporal queries directly based on the chronological timeline context.

---

## ⚙️ Configuration (`config.json`)

The system is configuration-driven. Values are loaded from `config.json` in the root:

```json
{
  "loitering_threshold": 5.0,
  "restricted_zones": {
    "CAM_01": [
      {
        "name": "zone_A",
        "polygon": [[0, 0], [250, 0], [250, 1080], [0, 1080]]
      }
    ]
  },
  "allowed_directions": {
    "CAM_01": {
      "angle": 90,
      "tolerance": 45
    }
  },
  "velocity_threshold": 100.0,
  "crowd_threshold": 3,
  "abandoned_object_threshold": 10.0,
  "risk_weights": {
    "loitering": 25,
    "intrusion": 40,
    "wrong_direction": 20,
    "sudden_movement": 20,
    "crowd_anomaly": 15,
    "abandoned_object": 30
  },
  "risk_combination_window": 30.0,
  "alert_threshold": "HIGH",
  "alert_cooldown": 60.0,
  "evidence": {
    "save_dir": "evidence"
  },
  "camera_start_times": {
    "CAM_01": "2026-08-23T10:00:00"
  }
}
```

*Secrets such as `GROQ_API_KEY`, `SMTP_USER`, `SMTP_PASS` should be placed in a `.env` file at the root.*

---

## ▶️ Running the Project

### 1️⃣ Process Camera Feeds
Place your video feeds in `data/videos/cam1.mp4`, etc., and run the pipeline:
```bash
python run_pipeline.py
```
This runs the YOLO tracker, assigns Re-ID global IDs (with MobileNetV3 fallback), runs frame-by-frame anomaly detection, logs unified events in `data/logs/`, and saves evidence clips.

### 2️⃣ Launch Command Center Dashboard
```bash
streamlit run dashboard/app.py
```
The Streamlit command center features:
* Top operational metrics (People, Events, Anomalies, High/Critical Risk).
* Visual statistics (risk distributions, anomaly categories, camera stats).
* Chronological event timeline plots.
* Person Movement Path builder (traces selected person across cameras/zones).
* Selectable Event Evidence Viewer (loads JPG/MP4 clips directly).
* Interactive Temporal Intelligence Chat panel.

---

## 🧪 Verification and Testing

The unit test suite uses synthetic tracking coordinates to run instantly without requiring video files or GPU resources.

To run the automated tests:
```bash
python tests/test_surveillance.py
```

The suite verifies:
1. Loitering time accumulator.
2. Polygon restricted-zone calculations (`cv2.pointPolygonTest`).
3. Movement direction vector differences.
4. Consecutive velocity speed computations.
5. Crowd size checks.
6. Profile-based risk score calculations, clamping, and classifications.
7. Cooldown de-duplication.
8. Grounded RAG temporal queries and filter extractions.
9. Insufficient evidence responses.

---

## 💬 Example Temporal Queries

You can ask the temporal intelligence analyst:
* *"What did person 1 do today?"*
* *"Show all suspicious activities that happened today."*
* *"Which camera had the most anomalies?"*
* *"Who entered restricted areas?"*
* *"What did person P014 do between 10:00:00 and 10:05:00?"*
* *"Show the timeline of P1"*
* *"What happened to P999?"* (returns *Insufficient evidence...*)

---

## 📌 Technical Limitations & Future Work

* **Camera Synchronization**: Assumes camera clocks are aligned to absolute timestamps. In future releases, network time protocol (NTP) sync will be integrated.
* **Complex Multi-Camera Linking**: Re-ID matching threshold (cosine similarity 0.6) is robust but can benefit from visual context learning in complex lighting.
* **Real-Time RTSP feeds**: Currently reads from video files; future work includes frame queues for live RTSP streaming streams.
