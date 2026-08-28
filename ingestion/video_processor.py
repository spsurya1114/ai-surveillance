
import os
import cv2
import uuid
from collections import deque
from datetime import datetime
from reid.reid_model import ReID
from reid.matcher import Matcher
from tracking.detector import Detector
from tracking.logger import EventLogger
from anomaly.detector import AnomalyDetector
from anomaly.alert_manager import RiskEngine, AlertManager


def process_video(path, camera_id):
    cap = cv2.VideoCapture(path)

    detector = Detector()
    reid = ReID()
    matcher = Matcher()
    anomaly_detector = AnomalyDetector()
    risk_engine = RiskEngine()
    alert_manager = AlertManager()

    logger = EventLogger(save_path=f"data/logs/{camera_id}.json")

    # Create output folder
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0  # default fallback

    frame_id = 0
    print("VIDEO PATH:", path)
    print("CAP OPEN:", cap.isOpened())

    # Sliding buffer to capture clips (3 seconds at 30 fps = 90 frames)
    frame_buffer = deque(maxlen=90)
    active_writers = []

    # Statistics counters
    total_frames = 0
    total_detections = 0
    total_anomalies_detected = 0
    total_anomalies_logged = 0
    total_alerts = 0

    # Cooldown registry for logging anomalies
    logged_anomalies = {}
    alert_cooldown = alert_manager.cooldown

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Progress reporting every 300 frames
        if frame_id > 0 and frame_id % 300 == 0:
            print(f"Frames processed: {frame_id}")

        detections = detector.track(frame)
        total_detections += len(detections)
        
        # Keep YOLO track IDs, fallback to list indices only if ID is missing
        for i, d in enumerate(detections):
            if d.get("id") is None:
                d["id"] = i
                
        frame_width = frame.shape[1]

        # -----------------------------
        # Re-ID
        # -----------------------------
        for d in detections:
            # Re-ID is only run on people (class 0)
            if d.get("class", 0) != 0:
                continue

            x1, y1, x2, y2 = map(int, d["bbox"])
            person_img = frame[y1:y2, x1:x2]

            if person_img.size == 0:
                continue

            features = reid.extract_features(person_img)
            global_id = matcher.match(features)
            d["global_id"] = global_id

        # -----------------------------
        # Logging standard events (Entry/Movement/Exit)
        # -----------------------------
        new_events = logger.update(frame_id, fps, detections, frame_width, camera_id)

        # -----------------------------
        # Anomaly detection (run every frame)
        # -----------------------------
        timestamp = frame_id / fps
        frame_anomalies = anomaly_detector.update(frame_id, fps, detections, camera_id, frame)
        
        if frame_anomalies:
            total_anomalies_detected += len(frame_anomalies)

        for a in frame_anomalies:
            # 1. Update risk profile
            person_or_obj_id = a.get("person_id") or a.get("object_id") or "global"
            risk_score, risk_level = risk_engine.update_profile(person_or_obj_id, a, timestamp)

            # 2. De-duplicate logging of anomalies to prevent duplicates on every frame
            key = (person_or_obj_id, a["anomaly_type"], camera_id)
            if key in logged_anomalies:
                if timestamp - logged_anomalies[key] < alert_cooldown:
                    continue

            logged_anomalies[key] = timestamp
            total_anomalies_logged += 1

            # 3. Determine if this anomaly triggers an alert (check without updating cooldown)
            is_alert = alert_manager.should_alert(a, risk_score, risk_level, timestamp, update_cooldown=False)

            evidence_img_path = None
            evidence_clip_path = None

            if is_alert:
                total_alerts += 1

                # Setup evidence directories
                date_str = datetime.now().strftime("%Y-%m-%d")
                evidence_dir = os.path.join("evidence", date_str, camera_id, a["anomaly_type"])
                os.makedirs(evidence_dir, exist_ok=True)

                event_uuid = str(uuid.uuid4())[:8]
                img_name = f"event_{a['anomaly_type']}_{event_uuid}.jpg"
                clip_name = f"event_{a['anomaly_type']}_{event_uuid}.mp4"

                evidence_img_path = os.path.join(evidence_dir, img_name)
                evidence_clip_path = os.path.join(evidence_dir, clip_name)

                # Save still frame image
                cv2.imwrite(evidence_img_path, frame)

                # Save video clip starting with buffered frames
                if len(frame_buffer) > 0:
                    try:
                        h, w = frame.shape[:2]
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        writer = cv2.VideoWriter(evidence_clip_path, fourcc, fps, (w, h))

                        # Write history
                        for f_buf in frame_buffer:
                            writer.write(f_buf)

                        # Track writer task to record subsequent 60 frames (2 seconds)
                        active_writers.append({
                            "writer": writer,
                            "frames_left": 60,
                            "path": evidence_clip_path
                        })
                    except Exception as e:
                        print(f"Failed to initialize clip writer: {e}")
                        evidence_clip_path = None
                else:
                    evidence_clip_path = None

            # 4. Log anomaly (for all unique non-duplicate detections)
            logged_event = logger.log_anomaly(a, risk_score, risk_level, evidence_img_path, evidence_clip_path)

            # 5. Dispatch Alert if eligible
            if is_alert:
                alert_manager.process_alert(logged_event, risk_score, risk_level, timestamp)
                print(f"⚠️ ANOMALY DETECTED & ALERTED: {a['anomaly_type']} (Risk: {risk_score})")

        # -----------------------------
        # Write current frame to any active recording writers
        # -----------------------------
        for w_task in active_writers[:]:
            try:
                w_task["writer"].write(frame)
                w_task["frames_left"] -= 1
                if w_task["frames_left"] <= 0:
                    w_task["writer"].release()
                    active_writers.remove(w_task)
            except Exception as e:
                print(f"Error writing to clip writer: {e}")
                active_writers.remove(w_task)

        # Update sliding history frame buffer (copy to avoid mutation)
        frame_buffer.append(frame.copy())

        # -----------------------------
        # Draw boxes
        # -----------------------------
        for d in detections:
            x1, y1, x2, y2 = map(int, d["bbox"])
            gid = d.get("global_id", -1)
            cls_id = d.get("class", 0)

            label = f"GID: {gid}" if cls_id == 0 else f"Obj: {cls_id}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # -----------------------------
        # Save output frame (every 30 frames / ~1 sec)
        # -----------------------------
        if frame_id % 30 == 0:
            cv2.imwrite(f"{output_dir}/{camera_id}_frame_{frame_id}.jpg", frame)

        frame_id += 1
        total_frames = frame_id

    # Cleanup remaining clip writers
    for w_task in active_writers:
        try:
            w_task["writer"].release()
        except:
            pass

    cap.release()
    logger.save()

    print("\n==========================================")
    print(f"{camera_id} PROCESSING SUMMARY")
    print("==========================================")
    print(f"Frames processed: {total_frames}")
    print(f"Total detections: {total_detections}")
    print(f"Total anomalies: {total_anomalies_detected}")
    print(f"Anomalies logged: {total_anomalies_logged}")
    print(f"Alerts generated: {total_alerts}")
    print("==========================================")

    print(f"{camera_id} processing completed.")


