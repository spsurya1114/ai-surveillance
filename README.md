# 🎥 AI Surveillance System (RAG + LLM Enabled)

An intelligent multi-camera surveillance system that analyzes CCTV footage and allows users to **query events using natural language**, similar to ChatGPT.

---

## 🚀 Overview

Traditional CCTV systems require manual monitoring, which is inefficient and time-consuming.
This project transforms passive video feeds into an **interactive AI system** that can:

* Detect and track people
* Identify individuals across frames (Re-ID)
* Log structured events (entry, movement, exit)
* Detect anomalies (e.g., loitering)
* Answer user queries using **RAG + LLM**

---

## 🧠 Key Features

* 🎯 **Object Detection & Tracking** (YOLOv8)
* 🧍 **Person Re-Identification (Re-ID)** using deep learning
* 📝 **Event Logging**

  * Entry / Movement / Exit
  * Zone-based tracking
* ⚠️ **Anomaly Detection**

  * Loitering detection based on time threshold
* 💬 **Natural Language Querying**

  * Ask questions like:

    * *“Where did person 1 go?”*
    * *“What happened at 3 seconds?”*
* 📊 **Interactive Dashboard (Streamlit)**

  * Event table
  * Timeline visualization
  * Chat interface

---

## 🏗️ System Architecture

```text
Video Input
   ↓
Detection (YOLOv8)
   ↓
Tracking + Re-ID
   ↓
Event Logging (JSON)
   ↓
Text Conversion
   ↓
RAG (Vector Search)
   ↓
LLM (Answer Generation)
   ↓
Streamlit Dashboard
```

---

## 📁 Project Structure

```text
dsa-surveillance/
│
├── anomaly/           # Anomaly detection logic
├── ingestion/         # Video processing pipeline
├── tracking/          # Detection + tracking + logging
├── reid/              # Person Re-ID model
├── rag/               # Retrieval + LLM integration
├── dashboard/         # Streamlit UI
├── data/              # Logs and sample videos
├── outputs/           # Processed frames
│
├── run_pipeline.py    # Run full pipeline
├── requirements.txt   # Dependencies
└── README.md
```

---

## ⚙️ Installation

```bash
git clone https://github.com/YOUR_USERNAME/ai-surveillance.git
cd ai-surveillance

python -m venv venv
venv\Scripts\activate   # Windows

pip install -r requirements.txt
```

---

## ▶️ How to Run

### 1️⃣ Run the pipeline

```bash
python run_pipeline.py
```

This will:

* Process videos from `data/videos/`
* Generate event logs in `data/logs/`
* Save output frames

---

### 2️⃣ Launch dashboard

```bash
streamlit run dashboard/app.py
```

---

## 💬 Example Queries

* Where did person 1 go?
* What happened in CAM_01?
* Who exited the scene?
* Were there any anomalies?

---

## 🛠️ Tech Stack

* Python
* OpenCV
* PyTorch
* YOLOv8 (Ultralytics)
* TorchReID
* FAISS / Vector Search
* Streamlit
* LLM (Groq / OpenAI-compatible)

---

## 📌 Key Highlights

* Combines **Computer Vision + NLP + LLM**
* Multi-camera surveillance simulation
* Real-time reasoning over video events
* Modular and scalable design

---

## 🚀 Future Improvements

* Real-time streaming (RTSP cameras)
* Advanced tracking (ByteTrack / DeepSORT)
* Multi-camera identity linking
* Improved anomaly detection (behavioral analysis)
* Deployment on GPU servers (DGX)

---

## 👨‍💻 Author

Surya SP

---

## ⭐ If you like this project

Give it a star ⭐ and share your feedback!
